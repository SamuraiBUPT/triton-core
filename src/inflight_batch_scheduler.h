#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

#include "backend_model.h"
#include "backend_model_instance.h"
#include "model_config.pb.h"
#include "rate_limiter.h"
#include "scheduler.h"
#include "scheduler_utils.h"
#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

class InflightBatchScheduler : public Scheduler {
private:
  /// the construction cannot be executed directly, if a scheduler
  /// is needed, use the Create function.
  InflightBatchScheduler(
    TritonModel* model, TritonModelInstance* model_instance,
    const bool inflight_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    const inference::ModelQueuePolicy& default_queue_policy,
    const uint64_t priority_levels,
    const ModelQueuePolicyMap& queue_policy_map);

public:
  /// @brief when desconstruct the scheduler object, the thread 
  /// will be executed and wait for it.
  ~InflightBatchScheduler();

  /// Create a scheduler to support a given number of runners 
  /// and a run function to call when a request is scheduled.
  /// The last param will be Output scheduler.
  static Status Create(
      TritonModel* model, TritonModelInstance* model_instance, const int nice,
      const bool inflight_batching_enabled, const int32_t max_batch_size,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool preserve_ordering,
      const std::set<int32_t>& preferred_batch_sizes,
      const uint64_t max_queue_delay_microseconds,
      std::unique_ptr<Scheduler>* scheduler);

  /// Create a scheduler through the batcher config.
  /// The last param will be Output scheduler.
  static Status Create(
      TritonModel* model, TritonModelInstance* model_instance, const int nice,
      const bool inflight_batching_enabled, const int32_t max_batch_size,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const inference::ModelDynamicBatching& batcher_config,
      std::unique_ptr<Scheduler>* scheduler);

  /// Override: `Enqueue` method from parent class "Scheduler".
  Status Enqueue(std::unique_ptr<InferenceRequest>& request) override;

  /// Override: `InflightInferenceCount` method from parent class "Scheduler".
  size_t InflightInferenceCount() override;

  /// Override
  bool stop_;
  void Stop() override { stop_ = true; }


private:
  void BatcherThread(const int nice);

  void UpdatePayload();

  /// @brief Iterate the queue and move the cursor inside the queue
  /// until one batch was formed. The unfinished request will still
  /// remain in the queue, and the state is "in-flight", these requests
  /// will be processed in the next batch until the <eos> was generated
  /// by any epoch. And the request will be sent.
  uint64_t GetInflightBatch();

  void DelegateResponse(std::unique_ptr<InferenceRequest>& request);

  void CacheLookUp(std::unique_ptr<InferenceRequest>& request,
                   std::unique_ptr<InferenceResponse>& cached_response);

  void FinalizeResponses();

  /// ================================================
  ///                    Attributes  
  /// ================================================
  TritonModel* model_;
  TritonModelInstance* model_instance_;

  // name of the model.
  std::string model_name_;

  // The flag for inflight batching.
  const bool inflight_batching_enabled_;

  // request pool, stores the requests not finished processing.
  // Maybe raw or in-flight.
  // I think the request should add the state "in-flight" for
  // further management.
  PriorityQueue queue_;

  // =========== variables of batcher thread ===========
  std::thread scheduler_thread_;  // one object hold one thread.
  std::atomic<bool> scheduler_thread_exit_;
  std::mutex mu_; // lock the object attribute.
  std::condition_variable cv_; // for thread synchronization.

  // =========== variables of payload ===========
  std::shared_ptr<RateLimiter> rate_limiter_;
  std::shared_ptr<Payload> curr_payload_; // the payload holding requests.

  /// the flag for current payload saturation. If the current payload is saturated,
  /// a new payload will be created through NewPayload().
  bool payload_saturated_;

  // =========== variables of batch info ===========
  size_t max_batch_size_;
  uint64_t pending_batch_delay_ns_;
  size_t pending_batch_size_;

  size_t queued_batch_size_;
  size_t next_preferred_batch_size_;

  // The input tensors that require shape checking before being
  // allowed in a batch. As a map from the tensor name to a bool. If
  // tensor is in map then its shape must match shape of same tensor
  // in requests already in the batch. If value is "true" then
  // additional tensor is treated as a shape tensor and the values
  // contained in the shape tensor must match same tensor already in
  // the batch.
  const std::unordered_map<std::string, bool> enforce_equal_shape_tensors_;

  // Store information on whether the model contains optional inputs.
  bool has_optional_input_;

  // If true the ordering of responses matches the order of requests
  // even when there are multiple scheduler threads.
  const bool preserve_ordering_;

  // If true, the scheduler will try to retrieve responses from cache.
  bool response_cache_enabled_;

  // Per completion-id queues to store the ready responses
  std::deque<
      std::vector<std::pair<std::unique_ptr<InferenceResponse>, uint32_t>>>
      completion_queue_;
  // Lock to protect the completion_queues_
  std::mutex completion_queue_mtx_;

  // Preserves the order in which responses are finalized
  std::mutex finalize_mtx_;

}

}}  // namespace triton::core