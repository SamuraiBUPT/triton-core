#include "inflight_batch_scheduler.h"

#ifdef _WIN32
  #include <sys/resource.h>
  #include <sys/syscall.h>
  #include <unistd.h>
#endif

#include "constants.h"
#incldue "server.h"
#include "triton/common/logging.h"
#include "triton/common/model_config.h"
#include "triton/common/nvtx.h"

namespace triton { namespace core {


InflightBatchScheduler(
  TritonModel* model, TritonModelInstance* model_instance,
  const bool inflight_batching_enabled, const int32_t max_batch_size,
  const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
  const bool preserve_ordering,
  const std::set<int32_t>& preferred_batch_sizes,
  const uint64_t max_queue_delay_microseconds,
  const inference::ModelQueuePolicy& default_queue_policy,
  const uint64_t priority_levels,
  const ModelQueuePolicyMap& queue_policy_map)
  : model_(model), model_instance_(model_instance),
    model_name_(model->Name()), 
    inflight_batching_enabled_(inflight_batching_enabled),
    queue_(default_queue_policy, priority_levels, queue_policy_map),
    stop_(false), max_batch_size_((size_t)std::max(1, max_batch_size)),
    preferred_batch_sizes_(preferred_batch_sizes),
    pending_batch_delay_ns_(max_queue_delay_microseconds * 1000),
    pending_batch_size_(0), pending_batch_queue_(0),
    enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
    has_optional_input(false),
    preserve_ordering_(preserve_ordering)
{
  rate_limiter_ = model_->Server()->GetRateLimiter();
  response_cache_enabled_ = model_->ResponseCacheEnabled() &&
                            model_->Server()->ResponseCacheEnabled();
  max_preferred_batch_size_ = 0;

  for (const auto& input : model_->Config().input()) {
    if (input.optional()) {
      has_optional_input_ = true;
      break;
    }
  }
}

Status
InflightBatchScheduler::Create(
    TritonModel* model, TritonModelInstance* model_instance, const int nice,
    const bool inflight_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    std::unique_ptr<Scheduler>* scheduler)
{
  inference::ModelDynamicBatching batcher_config;
  batcher_config.set_preserve_ordering(preserve_ordering);
  batcher_config.set_max_queue_delay_microseconds(max_queue_delay_microseconds);

  return Create(
      model, model_instance, nice, inflight_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config, scheduler);
}

Status
InflightBatchScheduler::Create(
      TritonModel* model, TritonModelInstance* model_instance, const int nice,
      const bool inflight_batching_enabled, const int32_t max_batch_size,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const inference::ModelDynamicBatching& batcher_config,
      std::unique_ptr<Scheduler>* scheduler)
{
  // call the construct func 
  InflightBatchScheduler* inflight_sched = new InflightBatchScheduler(
      model, model_instance, inflight_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config.preserve_ordering(),
      preferred_batch_sizes, batcher_config.max_queue_delay_microseconds(),
      batcher_config.default_queue_policy(), batcher_config.priority_levels(),
      batcher_config.priority_queue_policy());

  // unique_ptr to hold the resource.
  std::unique_ptr<InflightBatchScheduler> sched(inflight_sched);

  sched->scheduler_thread_exit_.store(false);
  if (inflight_batching_enabled) {
    sched->UpdatePayload();
    sched->scheduler_thread_ = std::thread(
      [inflight_sched, nice]() { inflight_sched->BatcherThread(nice); });
  }

  scheduler->reset(sched.release());
  return Status::Success;
}

InflightBatchScheduler::~InflightBatchScheduler()
{
  // Signal the scheduler thread to exit and then wait for it..
  scheduler_thread_exit_.store(true);
  cv_.notify_one();
  if (scheduler_thread_.joinable()) {
    scheduler_thread_.join();
  }
}

Status 
InflightBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  if (stop_) {
    return Status(
        Status::Code::UNAVAILABLE,
        request->LogRequest() +
            "Server is stopping, scheduler for model has stopped accepting new "
            "inference requests");
  }
  // If queue start timestamp hasn't been set, queue timer starts at
  // the beginning of the queueing and scheduling process. Otherwise,
  // dynamic batcher is used as component of another batcher and should not
  // overwrite the queue start timestamp.
  if (request->QueueStartNs() == 0) {
    request->CaptureQueueStartNs();
    INFER_TRACE_ACTIVITY(
        request->TraceProxy(), TRITONSERVER_TRACE_QUEUE_START,
        request->QueueStartNs());
#ifdef TRITON_ENABLE_TRACING
    request->TraceInputTensors(
        TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT, "DynamicBatchScheduler Enqueue");
#endif  // TRITON_ENABLE_TRACING
  }

  // Record time at the beginning of the batcher queueing. In the case of
  // oldest sequence batcher, this will overwrite the value that was previously
  // set by sequence batcher, which is okay as by this point, the previous
  // batcher won't be needing this value and it can be safely reused by
  // the dynamic batcher.
  request->CaptureBatcherStartNs();

  std::unique_ptr<InferenceResponse> cached_response;

  if (response_cache_enabled_) {
    CacheLookUp(request, cached_response);
  }

  if (cached_response != nullptr) {
    // If there was a cache hit then try sending the cached response
    // and release the request.
    if (preserve_ordering_) {
      // In order to preserve the order, the response send must be
      // delegated.
      DelegateResponse(request);
    }

    // Send cached response and release request
    InferenceResponse::Send(
        std::move(cached_response), TRITONSERVER_RESPONSE_COMPLETE_FINAL);
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);

    return Status::Success;
  }

  // judge if the exec thread can launch.
  bool wake_batcher = true;

  {
    std::lock_guard<std::mutex> lock(mu_);

    queued_batch_size_ += std::max(1U, request->BatchSize());

    // Assuming no error is returned, this call takes ownership of
    // 'request' and so we can't use it after this point.
    RETURN_IF_ERROR(queue_.Enqueue(request->Priority(), request));

    // If there are any idle runners and the queued batch size is greater or
    // equal to next preferred batch size, then wake batcher up to service
    // this request. We do the actual wake outside of the lock to avoid
    // having the woken thread immediately block on the lock
    // Explicitly force non-blocking to prevent waiting for the slot to
    // be available.
    wake_batcher = model_->Server()->GetRateLimiter()->PayloadSlotAvailable(
        model_, model_instance_, queue_.SupportPrefetching(),
        true /*force_non_blocking*/);

    // We may wake up runner less often if we don't enforce equal shape
    // within a batch, otherwise must always wake up runner to check it
    if (enforce_equal_shape_tensors_.empty()) {
      std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
      auto payload_state = curr_payload_->GetState();
      wake_batcher &=
          (payload_saturated_ || IsStaleState(payload_state) ||
            (queued_batch_size_ >= next_preferred_batch_size_));
    }
  }

  // when the situation is good, launch the stuck thread directly.
  if (wake_batcher) {
    cv_.notify_one();
  }
  return Status::Success;
}

// use the latest payload as the current payload, not created.
void 
InflightBatchScheduler::UpdatePayload()
{
  // GetPayload function returns a new payload. Not 
  // inserted in the payload queue, yet.
  curr_payload_ = model_->Server()->GetRateLimiter()->GetPayload(
      Payload::Operation::INFER_RUN, model_instance_);
  payload_saturated_ = false;
}

void 
InflightBatchScheduler::BatcherThread(const int nice)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting dynamic-batcher thread for " << model_name_
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting dynamic-batcher thread for " << model_name_
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting dynamic-batcher thread for " << model_name_
                 << " at default nice...";
#endif

  // ============== Debug Zone ==============
  // For debugging/testing, delay the execution of threads (hang) 
  // until the request queue contains the specified number of entries.
  size_t delay_cnt = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
    if (dstr != nullptr) {
      delay_cnt = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying batcher thread for " << model_name_
                     << " until " << delay_cnt << " queued requests...";
    }
  }
  // ============== Debug Zone ==============

  auto wait_for_slots = [this]() {
    return model_->Server()->GetRateLimiter()->PayloadSlotAvailable(
        model_, model_instance_, queue_.SupportPrefetching());
  };
  const uint64_t default_wait_microseconds = 500 * 1000;


  // now we will step into a loop for execution,
  // if the thread_exit flag atomic var isn't false, the loop keeps running.
  while(!scheduler_thread_exit_.load()) {
    NVTX_RANGE(nvtx_, "DynamicBatcher " + model_name_);

    std::vector<std::deque<std::unique_ptr<InferenceRequest>>> 
        rejected_requests, cancelled_requests;
    uint64_t wait_microseconds = 0;

    // lock the mutex
    {
      std::unique_lock<std::mutex> lock(mu_);
      {
        std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));  // lock the payload.
        auto payload_state = curr_payload_->GetState();
        // If the payload is saturated, or the payload is stale, check
        // if there is any new payload created in the rate limiter.
        // And update it as the current payload used by the scheduler.
        if (payload_saturated_ || IsStaleState(payload_state)) {
          UpdatePayload();
          next_preferred_batch_size_ = 0;
        }
      }
      if (delay_cnt > 0) {
        // ================= Debug Zone =============
        // only the cnt was set > 0, the code will executed to this branch.
        // ================= Debug Zone =============
        wait_microseconds = 10 * 1000;
        if (queue_.Size() >= delay_cnt) {
          delay_cnt = 0;
        }
        LOG_VERBOSE(1) << "Delaying batcher thread " << model_name_ << " until "
                       << delay_cnt
                       << " queued requests, current total = " << queue_.Size();
      } else if (queue_.Empty()) {
        wait_microseconds = default_wait_microseconds;
      } else {
        if (payload_saturated_) continue; // skip the loop.
        // wait for payload slots, and the thread will get stuck here.
        cv_.wait(lock, wait_for_slots); 
        {
          // lock the payload.
          std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
          auto payload_state = curr_payload_->GetState();
          if (IsStaleState(payload_state)) continue;  // in the next loop, the curr will be
          // updated.

          // Use the GetInflightBatch to get several requests from the 
          // request pool to execute.
          wait_microseconds = GetInflightBatch();

          queue_.ReleaseSkippedRequests(&rejected_requests, &cancelled_requests);

          // NOTICE: The logic may change:Extract batch only if there is pending batch
          auto pending_batch_queue_cnt = queue_.PendingBatchCount();
          if ((wait_microseconds == 0) && (pending_batch_queue_cnt != 0)) {
            curr_payload_->ReserveRequests(pending_batch_queue_cnt);

            // we will add all pending requests to curr_payload
            for (size_t idx = 0; idx < pending_batch_queue_cnt; idx++) {
              std::unique_ptr<InferenceRequest> request;
              auto status = queue_.Dequeue(&request);
              if (status.IsOk()) {
                if (preserve_ordering_ || response_cache_enabled_) {
                  DelegateResponse(request);
                }

                // ============= push to payload =============
                curr_payload_->AddRequest(std::move(request));
                // ===========================================
              } else {
                // If the queue is empty then will step into this brach
                // Send the current batch
                LOG_ERROR << request -> LogRequest()
                          << "Failed to retrieve request from scheduler queue: "
                          << status.Message();
                queue_.ResetCursor();
                queued_batch_size_ = 0;
                pending_batch_size_ = 0;
                break;
              }
            }

            if (curr_payload_->GetState() == Payload::State::UNINITIALIZED) {
              curr_payload_->SetState(Payload::State::READY);
            }

            queued_batch_size_ -= pending_batch_size_;
            pending_batch_size_ = 0;
          } // add request logic end
        } // process payload logic end
      } // form batch end

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
      }
    } // end lock, still in loop

    // now we will enqueue the curr_payload.
    // note the payload was created after check the available
    // slot and generated a new one. (UpdatePayload()), but this
    // newly generated payload wasn't into the payload queue yet.
    if (curr_payload_->GetState() == Payload::State::READY) {
      auto callback = [this]() { cv_.notify_one(); };
      curr_payload_->SetCallback(callback);
      // {
      //   std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
      //   CustomBatchFini();
      // }
      model_->Server()->GetRateLimiter()->EnqueuePayload(model_, curr_payload_);
    }

    // Finish rejected and cancelled requests if any
    const static Status rejected_status =
        Status(Status::Code::UNAVAILABLE, "Request timeout expired");
    const static Status cancelled_status = Status(Status::Code::CANCELLED);
    FinishSkippedRequests(std::move(rejected_requests), rejected_status);
    FinishSkippedRequests(std::move(cancelled_requests), cancelled_status);
    
  } // running loop end.
  LOG_VERBOSE(1) << "Stopping dynamic-batcher thread for " << model_name_
                 << "...";
} // batcher thread end

uint64_t 
InflightBatchScheduler::GetInflightBatch()
{
  // 'mu_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Examine the new requests. 
  // If adding these new requests to the pending batch 
  // allows a preferred batch size then execute it immediately. 

  // Stop examining requests [if the maximum preferred batch size 
  // would be exceeded] OR [if the shape of the next request does not 
  // match the shape of the pending batch].
  bool send_now = false;

  // If the PREVIOUS was not executed, reset the cursor to the start
  // of the queue to re-iterate over it and find the ideal batch.
  if (!queue_.IsCursorValid()) {
    // The cursor is not valid, and the pending batch is changed.
    queue_.ResetCursor();
    pending_batch_size_ = 0;
  }
  size_t best_preferred_batch_size = 0;
  queued_batch_size_ -= queue_.ApplyPolicyAtCursor();

}









} // namespace triton::core
} // namespace triton