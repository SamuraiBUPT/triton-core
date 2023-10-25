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

#pragma message("============== Inflight Batch Scheduler is included =============")

namespace triton { namespace core {
uint64_t
CaptureTimeNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool
IsStaleState(Payload::State payload_state)
{
  return (
      (payload_state == Payload::State::EXECUTING) ||
      (payload_state == Payload::State::RELEASED));
}

void
FinishSkippedRequests(
    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>&& requests,
    const Status& response_status)
{
  for (auto& queue : requests) {
    for (auto& request : queue) {
      InferenceRequest::RespondIfError(request, response_status, true);
    }
  }
}

InflightBatchScheduler::DynamicBatchScheduler(
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

              // TODO: here is not dequeue, need to modify
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

  // reduce the rejected or cancelled requests.
  queued_batch_size_ -= queue_.ApplyPolicyAtCursor();

  // When there is optional input or input shape must be enforced,
  // the inputs in the requests must be examined for forming a batch
  const bool check_input = !enforce_equal_shape_tensors_.empty() ||
                           has_optional_input_;
  auto payload_batch_size = curr_payload_->BatchSize(); 
  while(!queue_CursorEnd()) {
    // now we are adding each request.

    const auto batch_size = std::max(1U, queue_.RequestAtCursor()->BatchSize());

    if ((payload_batch_size + queue_.PendingBatchCount()) == 0) {
      // If there is no pending batch, then this request is starting a
      // new batch.
      // Get the shape of the new batch that is being started...
      if (check_input) {
        if (!curr_payload_->MutableRequiredEqualInputs()
                 ->Initialize(
                     queue_.RequestAtCursor(), enforce_equal_shape_tensors_,
                     has_optional_input_)
                 .IsOk()) {
          send_now = true;
          break;
        }
      }
    } else {
      // There is a pending batch, situation 1: the batch is full.
      // 
      // adding this request would make the batch size larger than all of 
      // the preferred batch sizes,
      // so mark the cursor at this point. Not sending the pending batch so
      // that we can examine the queue delay of requests that fits in a batch.
      if (((payload_batch_size + pending_batch_size_ + batch_size) >
           max_preferred_batch_size_) && (best_preferred_batch_size == 0)) {
        best_preferred_batch_size = pending_batch_size_;
        queue_.MarkCursor();
        payload_saturated_ = true;
      }
      if((payload_batch_size + pending_batch_size_ + batch_size) >
          max_batch_size_) {
        send_now = true;
        break;
      }

      // There is a pending batch, situation 2: shape change.
      //
      // the batch has a different shape if adding this request, so send 
      // the pending batch as it is.
      if (check_input &&
          !curr_payload_->MutableRequiredEqualInputs()->HasEqualInputs(
              queue_.RequestAtCursor())) {
        curr_payload_->MarkSaturated();
        send_now = true;
        break;
      }
    }

    pending_batch_size_ += batch_size;   // add this request count to bs
    queue_.AdvanceCursor();             // move the cursor to next request
    queued_batch_size_ -= queue_.ApplyPolicyAtCursor();  

    if (preferred_batch_sizes_.find(pending_batch_size_ + payload_batch_size) !=
        preferred_batch_sizes_.end()) {
      best_preferred_batch_size = pending_batch_size_;
      queue_.MarkCursor();
    }
  } // examine request loop end

  // Time Computing
  // Obtain the age of the oldest pending request to compare with the maximum
  // batch queuing delay.
  uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  uint64_t delay_ns = now_ns - queue_.OldestEnqueueTime();
  bool delay_is_exceeded =
      (pending_batch_delay_ns_ != 0) && (delay_ns >= pending_batch_delay_ns_);

  // If we found a preferred batch size and the queue delay hasn't
  // been exceeded, then execute that.
  if ((best_preferred_batch_size != 0) && !delay_is_exceeded) {
    if (pending_batch_delay_ns_ == 0) {
      payload_saturated_ = true;
    }
    pending_batch_size_ = best_preferred_batch_size;
    queue_.SetCursorToMark();
    return 0;
  }

  // No request in pending batch happens when all queued requests have expired
  // timeout and the policies are REJECT
  if (queue_.PendingBatchCount() == 0) {
    return 0;
  }

  // if the current batch can't grow any larger then just immediately 
  // execute whatever is pending.
  if (send_now || ((payload_batch_size + pending_batch_size_) >=
                   max_preferred_batch_size_)) {
    payload_saturated_ = true;
    return 0;
  }

  // If the delay has been exceeded
  if (delay_is_exceeded || (pending_batch_delay_ns_ == 0)) {
    return 0;
  }

  // Set the next preferred batch size given the pending batch size
  auto next_preferred_batch_size_it = preferred_batch_sizes_.upper_bound(
      pending_batch_size_ + payload_batch_size);

  if (next_preferred_batch_size_it != preferred_batch_sizes_.end()) {
    next_preferred_batch_size_ = *next_preferred_batch_size_it;
  } else {
    next_preferred_batch_size_ =
        preferred_batch_sizes_.empty() ? 0 : *preferred_batch_sizes_.begin();
  }

  if (next_preferred_batch_size_ != 0) {
    next_preferred_batch_size_ -= payload_batch_size;
  }


  // By this point, we have not seen the pending batch that should be executed
  // immediately. However, if we have scheduled a payload that can be grown and
  // not yet in preferred batch size, we should move the pending batch over to
  // ensure the model instance will pick up largest available batch even if it
  // is not the preferred batch.
  if (!payload_saturated_ && (payload_batch_size != 0) &&
      (preferred_batch_sizes_.find(payload_batch_size) ==
       preferred_batch_sizes_.end())) {
    return 0;
  }

  uint64_t wait_ns = pending_batch_delay_ns_ - delay_ns;
  // Note that taking request timeout into consideration allows us to reset
  // pending batch as soon as it is invalidated. But the cost is that in edge
  // case where the timeout will be expired one by one, the thread will be
  // waken frequently.
  if (queue_.ClosestTimeout() != 0) {
    if (now_ns <= queue_.ClosestTimeout()) {
      wait_ns = std::min(queue_.ClosestTimeout() - now_ns, wait_ns);
    } else {
      // A request in pending batch is timed-out, wait for 1 us to force the
      // thread to reset the pending batch right the way.
      wait_ns = 1000;
    }
  }

  // Return non-zero wait microseconds to cause this thread to wait
  // until the queue delay or the closest timeout has expired.
  // Another thread may be awaken due to incoming request to handle the
  // pending batch before this thread wakes and that is ok. But if no other
  // request comes in then this thread will wake and revisit the pending batch
  // (and at that time will then see the delay has been exceeded and will send
  // the batch).
  return wait_ns / 1000;
}

void
InflightBatchScheduler::DelegateResponse(
    std::unique_ptr<InferenceRequest>& request)
{
  std::lock_guard<std::mutex> lock(completion_queue_mtx_);
  completion_queue_.emplace_back();
  auto queue_slot = &completion_queue_.back();
  // Cache plumbing
  const std::string& key = request->CacheKey();
  const bool is_key_set = request->CacheKeyIsSet();
  const uint64_t lookup_end_ns = request->CacheLookupEndNs();
  const uint64_t lookup_start_ns = request->CacheLookupStartNs();

  request->SetResponseDelegator(
      [this, queue_slot, key, is_key_set, lookup_end_ns, lookup_start_ns](
          std::unique_ptr<InferenceResponse>&& response, const uint32_t flags) {
        if (response_cache_enabled_) {
          // Logical error, the key should be set if caching is enabled
          // for this model
          if (!is_key_set) {
            LOG_ERROR << "Request cache key was not set correctly.";
          }

          // Cache insertion happens here because we need the backend to have
          // computed the inference response first in the case of cache miss
          auto cache = model_->Server()->CacheManager()->Cache();

#ifdef TRITON_ENABLE_STATS
          const uint64_t insert_start_ns = CaptureTimeNs();
#endif  // TRITON_ENABLE_STATS

          auto status = cache->Insert(response.get(), key);

#ifdef TRITON_ENABLE_STATS
          const uint64_t insert_end_ns = CaptureTimeNs();
#endif  // TRITON_ENABLE_STATS

          bool cache_miss =
              (status.StatusCode() != Status::Code::ALREADY_EXISTS);
          if (cache_miss) {
#ifdef TRITON_ENABLE_STATS
            uint64_t lookup_ns = lookup_end_ns - lookup_start_ns;
            // Logical error, this shouldn't happen
            if (lookup_start_ns > lookup_end_ns) {
              lookup_ns = 0;
              LOG_ERROR << "Request lookup duration was not set correctly.";
            }

            uint64_t insert_ns = insert_end_ns - insert_start_ns;
            uint64_t cache_miss_ns = lookup_ns + insert_ns;
            // Use model_ to update stats directly because request object can be
            // released by the backend before getting to this callback.
            model_->MutableStatsAggregator()->UpdateSuccessCacheMiss(
                model_->MetricReporter().get(), cache_miss_ns);
#endif  // TRITON_ENABLE_STATS
            if (!status.IsOk()) {
              LOG_ERROR << "Failed to insert key [" << key
                        << "] into response cache: " << status.Message();
            }
          }  // Otherwise do nothing; we update cache hit statistics on Lookup
        }

        if (preserve_ordering_) {
          {
            std::lock_guard<std::mutex> lock(completion_queue_mtx_);
            queue_slot->emplace_back(std::move(response), flags);
          }
          FinalizeResponses();
        } else {
          InferenceResponse::Send(std::move(response), flags);
        }
      });
}

void
InflightBatchScheduler::CacheLookUp(
    std::unique_ptr<InferenceRequest>& request,
    std::unique_ptr<InferenceResponse>& cached_response)
{
  Status status;
  auto cache = model_->Server()->CacheManager()->Cache();
  std::unique_ptr<InferenceResponse> local_response;
  request->ResponseFactory()->CreateResponse(&local_response);
  // Hash request into cache key
  std::string key = "";
  if (!request->CacheKeyIsSet()) {
    status = cache->Hash(*request, &key);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to hash request: " << status.Message();
      return;
    }
    request->SetCacheKey(key);
  } else {
    key = request->CacheKey();
  }

  // Lookup and capture timestamps
  {
    request->CaptureCacheLookupStartNs();
    status = cache->Lookup(local_response.get(), key);
    request->CaptureCacheLookupEndNs();
  }

  if (status.IsOk() && (local_response != nullptr)) {
    cached_response = std::move(local_response);
#ifdef TRITON_ENABLE_STATS
    // Update model metrics/stats on cache hits
    // Backends will update metrics as normal on cache misses
    request->ReportStatisticsCacheHit(model_->MetricReporter().get());
#endif  // TRITON_ENABLE_STATS
  }
}

void
InflightBatchScheduler::FinalizeResponses()
{
  // Need exclusive access of the function to ensure responses are sent
  // in order
  std::lock_guard<std::mutex> lock(finalize_mtx_);
  // Finalize the completed payloads in-order as far as possible
  std::deque<std::pair<std::unique_ptr<InferenceResponse>, const uint32_t>>
      responses;
  {
    std::lock_guard<std::mutex> queue_lock(completion_queue_mtx_);
    while (!completion_queue_.empty() && !completion_queue_.front().empty()) {
      bool response_complete = false;
      for (auto& response_pair : completion_queue_.front()) {
        // Assuming FINAL flag is set only in the last response of the request
        response_complete =
            ((response_pair.second & TRITONSERVER_RESPONSE_COMPLETE_FINAL) !=
             0);
        responses.emplace_back(std::move(response_pair));
      }
      if (response_complete) {
        completion_queue_.pop_front();
      } else {
        completion_queue_.front().clear();
      }
    }
  }

  for (auto& response : responses) {
    InferenceResponse::Send(std::move(response.first), response.second);
  }
}

size_t
InflightBatchScheduler::InflightInferenceCount()
{
  std::unique_lock<std::mutex> lock(mu_);
  if (curr_payload_ != nullptr) {
    return queue_.Size() + curr_payload_->RequestCount();
  }
  return queue_.Size();
}






} // namespace triton::core
} // namespace triton