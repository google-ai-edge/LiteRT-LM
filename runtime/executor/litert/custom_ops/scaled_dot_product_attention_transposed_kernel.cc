/* Copyright 2026 The LiteRT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "runtime/executor/litert/custom_ops/scaled_dot_product_attention_transposed_kernel.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/litert/custom_ops/attention_utils.h"

namespace litert {
namespace lm {

ScaledDotProductAttentionTransposedKernel::
    ScaledDotProductAttentionTransposedKernel()
    : pool_("sdpa_transposed", 4) {}

Expected<void> ScaledDotProductAttentionTransposedKernel::Init(
    const void* init_data, size_t init_data_size) {
  if (init_data_size > 0) {
    auto map = flexbuffers::GetRoot(static_cast<const uint8_t*>(init_data),
                                    init_data_size)
                   .AsMap();
    if (!map["softcap"].IsNull()) {
      softcap_ = map["softcap"].AsFloat();
    } else if (!map["logit_cap"].IsNull()) {
      softcap_ = map["logit_cap"].AsFloat();
    }
    if (!map["k_ts_idx"].IsNull()) {
      k_ts_idx_ = map["k_ts_idx"].AsInt32();
    }
    if (!map["v_ts_idx"].IsNull()) {
      v_ts_idx_ = map["v_ts_idx"].AsInt32();
    }
  }
  return {};
}

Expected<void> ScaledDotProductAttentionTransposedKernel::GetOutputLayouts(
    const std::vector<Layout>& input_layouts,
    std::vector<Layout>& output_layouts) {
  if ((input_layouts.size() != 4 && input_layouts.size() != 5) ||
      output_layouts.size() != 1) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid number of inputs/outputs");
  }
  output_layouts[0] = input_layouts[0];
  return {};
}

Expected<void> ScaledDotProductAttentionTransposedKernel::Run(
    const std::vector<TensorBuffer>& inputs,
    std::vector<TensorBuffer>& outputs) {
  if ((inputs.size() != 4 && inputs.size() != 5) || outputs.size() != 1) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Invalid number of inputs/outputs");
  }

  LITERT_ASSIGN_OR_RETURN(auto q_lock,
                          TensorBufferScopedLock::Create<float>(
                              inputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto k_lock,
                          TensorBufferScopedLock::Create<float>(
                              inputs[1], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto v_lock,
                          TensorBufferScopedLock::Create<float>(
                              inputs[2], TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(auto out_lock,
                          TensorBufferScopedLock::Create<float>(
                              outputs[0], TensorBuffer::LockMode::kWrite));

  LITERT_ASSIGN_OR_RETURN(auto q_type, inputs[0].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto k_type, inputs[1].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto v_type, inputs[2].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto mask_type, inputs[3].TensorType());
  LITERT_ASSIGN_OR_RETURN(auto out_type, outputs[0].TensorType());

  std::vector<float> mask_float_buffer;
  const float* mask_data_ptr = nullptr;
  std::optional<std::pair<TensorBufferScopedLock, const float*>>
      mask_lock_float;

  if (mask_type.ElementType() == ElementType::Bool ||
      mask_type.ElementType() == ElementType::UInt8 ||
      mask_type.ElementType() == ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(auto mask_lock,
                            TensorBufferScopedLock::Create<uint8_t>(
                                inputs[3], TensorBuffer::LockMode::kRead));
    auto num_elements_or = mask_type.Layout().NumElements();
    size_t num_elements =
        num_elements_or.HasValue() ? num_elements_or.Value() : 0;
    mask_float_buffer.resize(num_elements);
    const uint8_t* bool_data = mask_lock.second;
    for (size_t i = 0; i < num_elements; ++i) {
      mask_float_buffer[i] = (bool_data[i] != 0) ? 0.0f : -1e30f;
    }
    mask_data_ptr = mask_float_buffer.data();
  } else {
    LITERT_ASSIGN_OR_RETURN(auto lock_float,
                            TensorBufferScopedLock::Create<float>(
                                inputs[3], TensorBuffer::LockMode::kRead));
    mask_lock_float.emplace(std::move(lock_float));
    mask_data_ptr = mask_lock_float->second;
  }

  std::optional<int32_t> active_seq_len = std::nullopt;
  if (inputs.size() == 5) {
    LITERT_ASSIGN_OR_RETURN(auto param_type, inputs[4].TensorType());
    auto num_elements_or = param_type.Layout().NumElements();
    size_t num_elements =
        num_elements_or.HasValue() ? num_elements_or.Value() : 0;
    if (param_type.ElementType() == ElementType::Int32) {
      LITERT_ASSIGN_OR_RETURN(auto param_lock,
                              TensorBufferScopedLock::Create<int32_t>(
                                  inputs[4], TensorBuffer::LockMode::kRead));
      if (num_elements >= 3) {
        active_seq_len = param_lock.second[2];
      } else if (num_elements >= 1) {
        active_seq_len = param_lock.second[0];
      }
    } else if (param_type.ElementType() == ElementType::Int64) {
      LITERT_ASSIGN_OR_RETURN(auto param_lock,
                              TensorBufferScopedLock::Create<int64_t>(
                                  inputs[4], TensorBuffer::LockMode::kRead));
      if (num_elements >= 3) {
        active_seq_len = static_cast<int32_t>(param_lock.second[2]);
      } else if (num_elements >= 1) {
        active_seq_len = static_cast<int32_t>(param_lock.second[0]);
      }
    }
  }

  auto get_shape = [](const RankedTensorType& type) {
    auto span = type.Layout().Dimensions();
    return std::vector<int32_t>(span.begin(), span.end());
  };

  TensorRef query_ref = {q_lock.second, get_shape(q_type)};
  TensorRef key_ref = {k_lock.second, get_shape(k_type)};
  TensorRef value_ref = {v_lock.second, get_shape(v_type)};
  TensorRef mask_ref = {mask_data_ptr, get_shape(mask_type)};
  MutableTensorRef output_ref = {out_lock.second, get_shape(out_type)};

  const int32_t bk = query_ref.shape[1];
  const int32_t gt = query_ref.shape[2];

  std::atomic<bool> success{true};

  if (gt <= 4) {
    const int32_t num_threads = std::min(2, bk);
    const int32_t heads_per_thread = (bk + num_threads - 1) / num_threads;

    int32_t num_tasks = 0;
    for (int32_t t_id = 0; t_id < num_threads; ++t_id) {
      if (t_id * heads_per_thread < bk) {
        ++num_tasks;
      }
    }

    if (num_tasks > 0) {
      absl::Notification done;
      std::atomic<int32_t> counter{num_tasks};

      for (int32_t t_id = 0; t_id < num_threads; ++t_id) {
        int32_t h_start = t_id * heads_per_thread;
        int32_t h_end = std::min(h_start + heads_per_thread, bk);

        if (h_start >= bk) break;

        auto status = pool_.Schedule([h_start, h_end, &query_ref, &key_ref,
                                      &value_ref, &mask_ref, this, &output_ref,
                                      active_seq_len, &success, &counter,
                                      &done]() {
          for (int32_t h_idx = h_start; h_idx < h_end; ++h_idx) {
            if (!ComputeTransposedAttentionSingleHead(
                    h_idx, query_ref, key_ref, value_ref, mask_ref, softcap_,
                    k_ts_idx_, v_ts_idx_, output_ref, active_seq_len,
                    std::nullopt, 1, nullptr, nullptr, nullptr)) {
              success.store(false, std::memory_order_relaxed);
              break;
            }
          }
          if (counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            done.Notify();
          }
        });
        if (!status.ok()) {
          return Unexpected(Status::kErrorRuntimeFailure,
                            "Failed to schedule decode head task.");
        }
      }

      done.WaitForNotification();
    }

    if (!success.load(std::memory_order_relaxed)) {
      return Unexpected(
          Status::kErrorRuntimeFailure,
          "ComputeTransposedAttentionSingleHead failed in decode head task.");
    }
  } else {
    const int32_t num_threads = 4;
    const int32_t chunk_size = (gt + num_threads - 1) / num_threads;

    int32_t num_tasks = 0;
    for (int32_t t_id = 0; t_id < num_threads; ++t_id) {
      if (t_id * chunk_size < gt) {
        ++num_tasks;
      }
    }

    if (num_tasks > 0) {
      absl::Notification done;
      std::atomic<int32_t> counter{num_tasks};

      for (int32_t t_id = 0; t_id < num_threads; ++t_id) {
        int32_t start_gt = t_id * chunk_size;
        int32_t end_gt = std::min(start_gt + chunk_size, gt);

        if (start_gt >= gt) break;

        auto status = pool_.Schedule([h_idx_start = 0, h_idx_end = bk, start_gt,
                                      end_gt, &query_ref, &key_ref, &value_ref,
                                      &mask_ref, this, &output_ref,
                                      active_seq_len, &success, &counter,
                                      &done]() {
          for (int32_t h_idx = h_idx_start; h_idx < h_idx_end; ++h_idx) {
            if (!ComputeTransposedAttentionSingleHead(
                    h_idx, query_ref, key_ref, value_ref, mask_ref, softcap_,
                    k_ts_idx_, v_ts_idx_, output_ref, active_seq_len,
                    std::make_pair(start_gt, end_gt), 1, nullptr, nullptr,
                    nullptr)) {
              success.store(false, std::memory_order_relaxed);
              break;
            }
          }
          if (counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            done.Notify();
          }
        });
        if (!status.ok()) {
          return Unexpected(Status::kErrorRuntimeFailure,
                            "Failed to schedule prefill task.");
        }
      }

      done.WaitForNotification();
    }

    if (!success.load(std::memory_order_relaxed)) {
      return Unexpected(
          Status::kErrorRuntimeFailure,
          "ComputeTransposedAttentionSingleHead failed in prefill task.");
    }
  }

  return {};
}

Expected<void> ScaledDotProductAttentionTransposedKernel::Destroy() {
  return {};
}

}  // namespace lm
}  // namespace litert
