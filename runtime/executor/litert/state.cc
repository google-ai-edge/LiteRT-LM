// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/executor/litert/state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/common_utils.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/state_interface.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

namespace {

constexpr int kDynamicDimValue = -1;

absl::StatusOr<std::optional<int>> GetDynamicDimIndex(
    const CompiledModel& compiled_model, absl::string_view signature,
    absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  std::optional<int> dynamic_dim_index;
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      RET_CHECK(!dynamic_dim_index.has_value())
          << "Multiple dynamic dimensions are not supported.";
      dynamic_dim_index = i;
    }
  }
  return dynamic_dim_index;
}

absl::Status ResolveDynamicShape(CompiledModel& compiled_model,
                                 absl::string_view signature,
                                 absl::string_view tensor_name, int new_value) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();

  bool has_dynamic_dim = false;
  std::vector<int> new_shape;
  new_shape.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      has_dynamic_dim = true;
      new_shape.push_back(new_value);
    } else {
      new_shape.push_back(dimensions[i]);
    }
  }

  if (has_dynamic_dim) {
    LITERT_RETURN_IF_ERROR(
        compiled_model.ResizeInputTensor(signature, tensor_name, new_shape));
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> ResizeTensorBuffer(Environment& env,
                                                TensorBuffer& tensor_buffer,
                                                int dynamic_dim_index,
                                                int num_entries_to_insert) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType& tensor_type,
                          tensor_buffer.TensorType());
  RET_CHECK(!tensor_type.Layout().HasStrides());
  auto dimensions = tensor_type.Layout().Dimensions();
  std::vector<int> new_dimensions;
  new_dimensions.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (i == dynamic_dim_index) {
      new_dimensions.push_back(dimensions[i] + num_entries_to_insert);
    } else {
      new_dimensions.push_back(dimensions[i]);
    }
  }

  LITERT_ASSIGN_OR_RETURN(TensorBufferType buffer_type,
                          tensor_buffer.BufferType());
  Layout new_layout(Dimensions(new_dimensions.begin(), new_dimensions.end()));
  auto new_out_type =
      RankedTensorType(tensor_type.ElementType(), std::move(new_layout));
  LITERT_ASSIGN_OR_RETURN(size_t new_size, new_out_type.Bytes());

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer new_tensor_buffer,
      TensorBuffer::CreateManaged(env, buffer_type, new_out_type, new_size));
  LITERT_RETURN_IF_ERROR(new_tensor_buffer.Clear());

  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  auto* tensor_buffer_ptr =
      static_cast<uint8_t*>(tensor_buffer_lock_and_addr.second);
  LITERT_ASSIGN_OR_RETURN(
      auto new_tensor_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(new_tensor_buffer,
                                     TensorBuffer::LockMode::kWrite));
  auto* new_tensor_buffer_ptr =
      static_cast<uint8_t*>(new_tensor_buffer_lock_and_addr.second);
  std::optional<size_t> element_size = GetByteWidth(tensor_type.ElementType());
  RET_CHECK(element_size.has_value());

  ABSL_RETURN_IF_ERROR(ExpandBuffer(tensor_buffer_ptr, dimensions,
                                    new_tensor_buffer_ptr, new_dimensions,
                                    element_size.value()));

  return new_tensor_buffer;
}

absl::Status SelectAndCopyBuffer(TensorBuffer& dst, const TensorBuffer& src,
                                 int batch_index) {
  LITERT_ASSIGN_OR_RETURN(
      auto src_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  const char* src_buffer_ptr =
      static_cast<const char*>(src_buffer_lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(
      auto dst_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t dst_buffer_size, dst.PackedSize());
  char* dst_buffer_ptr =
      static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));
  // This copy is based on the assumption that the KV cache buffers are in the
  // layout of [batch * X, ...] or [1, batch * X, ...] where X could be 1 or
  // more and X doesn't make values interleaved across batches which is true
  // for the current LLM models of all backends.
  src_buffer_ptr += batch_index * dst_buffer_size;
  memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
  return absl::OkStatus();
}

absl::Status BroadcastAndCopyBuffer(TensorBuffer& dst, int dst_batch_size,
                                    const TensorBuffer& src) {
  LITERT_ASSIGN_OR_RETURN(
      auto src_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(size_t src_buffer_size, src.PackedSize());
  const char* src_buffer_ptr =
      static_cast<const char*>(src_buffer_lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(
      auto dst_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  char* dst_buffer_ptr =
      static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));

  for (int i = 0; i < dst_batch_size; ++i) {
    memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
    dst_buffer_ptr += src_buffer_size;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::Create(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata* executor_metadata,
    AllocationPolicy allocation_policy, int batch_size,
    bool clear_kv_cache_before_prefill) {
  if (executor_metadata == nullptr) {
    return HeuristicBasedCreate(env, compiled_model, signature_name,
                                allocation_policy, batch_size,
                                clear_kv_cache_before_prefill);
  }

  return MetadataBasedCreate(env, compiled_model, signature_name,
                             *executor_metadata, allocation_policy, batch_size,
                             clear_kv_cache_before_prefill);
}

absl::Status LitertState::SelectAndCopyFrom(StateInterface& other,
                                            int batch_index) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_state_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_state_buffers_.has_value());
  RET_CHECK_GT(other_litert->batch_size_, batch_size_);
  RET_CHECK_LT(batch_index, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    auto it = other_litert->bank_1_state_buffers_.find(input_name);
    RET_CHECK(it != other_litert->bank_1_state_buffers_.end());
    ABSL_RETURN_IF_ERROR(SelectAndCopyBuffer(state_buffer.buffer,
                                             it->second.buffer, batch_index));
  }
  return absl::OkStatus();
}

absl::Status LitertState::BroadcastAndCopyFrom(StateInterface& other) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_state_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_state_buffers_.has_value());
  RET_CHECK_EQ(other_litert->batch_size_, 1);
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    auto it = other_litert->bank_1_state_buffers_.find(input_name);
    RET_CHECK(it != other_litert->bank_1_state_buffers_.end());
    ABSL_RETURN_IF_ERROR(BroadcastAndCopyBuffer(
        state_buffer.buffer, batch_size_, it->second.buffer));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DeepCopy() const {
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto buf_copy,
                            CopyTensorBuffer(env_, state_buffer.buffer));
    bank_1_state_buffers[name] = StateBuffer{
        .buffer = std::move(buf_copy),
        .type = state_buffer.type,
        .dynamic_dim = state_buffer.dynamic_dim,
    };
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (bank_2_state_buffers_.has_value()) {
    bank_2_state_buffers.emplace();
    auto& map = *bank_2_state_buffers;
    for (const auto& [name, state_buffer] : *bank_2_state_buffers_) {
      LITERT_ASSIGN_OR_RETURN(auto buf_copy,
                              CopyTensorBuffer(env_, state_buffer.buffer));
      map[name] = StateBuffer{
          .buffer = std::move(buf_copy),
          .type = state_buffer.type,
          .dynamic_dim = state_buffer.dynamic_dim,
      };
    }
  }

  auto copy = absl::WrapUnique(new LitertState(
      batch_size_, num_entries_, env_, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy_));
  copy->bank_1_is_input_ = bank_1_is_input_;

  return copy;
}

bool LitertState::Contains(absl::string_view tensor_name) const {
  return bank_1_state_buffers_.contains(tensor_name);
}

absl::Status LitertState::Clear() {
  for (auto& [_, state_buffer] : bank_1_state_buffers_) {
    LITERT_RETURN_IF_ERROR(state_buffer.buffer.Clear());
  }
  if (bank_2_state_buffers_.has_value()) {
    for (auto& [_, state_buffer] : *bank_2_state_buffers_) {
      LITERT_RETURN_IF_ERROR(state_buffer.buffer.Clear());
    }
  }
  return absl::OkStatus();
}

absl::Status LitertState::Resize(CompiledModel& compiled_model,
                                 absl::string_view signature_name,
                                 int num_entries) {
  RET_CHECK(!bank_2_state_buffers_.has_value())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Out of place KV cache cannot be resized.";

  bool has_dynamic = false;
  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      has_dynamic = true;
      break;
    }
  }
  if (!has_dynamic) {
    return absl::InvalidArgumentError(
        "KV cache is not dynamic and cannot be resized.");
  }

  int entries_to_add = num_entries - num_entries_;
  if (entries_to_add <= 0) {
    return absl::OkStatus();
  }

  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, num_entries));
    }
  }

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          state_buffer.buffer,
          ResizeTensorBuffer(env_, state_buffer.buffer,
                             state_buffer.dynamic_dim.value(), entries_to_add));
    }
  }
  num_entries_ = num_entries;
  return absl::OkStatus();
}

absl::StatusOr<LitertState::StateBuffers> LitertState::GetStateBuffers(
    CompiledModel& compiled_model, absl::string_view signature_name) {
  LITERT_RETURN_IF_ERROR(SyncShapes(compiled_model, signature_name));
  auto* input_bank = &bank_1_state_buffers_;
  auto* output_bank = &bank_1_state_buffers_;

  if (bank_2_state_buffers_.has_value()) {
    if (bank_1_is_input_) {
      output_bank = &bank_2_state_buffers_.value();
    } else {
      input_bank = &bank_2_state_buffers_.value();
    }
    bank_1_is_input_ = !bank_1_is_input_;
  }

  StateBuffers buffers;
  const bool should_skip_inputs =
      allocation_policy_ == AllocationPolicy::kGpuOptimizedInplace;
  const bool is_prefill = absl::StartsWith(signature_name, "prefill");

  for (const auto& [input_name, state_buffer] : *input_bank) {
    const bool is_local_kv_cache =
        state_buffer.type == proto::StateBuffer::TYPE_LOCAL_KEY_CACHE ||
        state_buffer.type == proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE;
    if (should_skip_inputs && (!is_prefill || !is_local_kv_cache)) {
      // For GPU optimized in place updates, we are required to pass the local
      // KV cache buffers as inputs too in the prefill stage.
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto duplicated, state_buffer.buffer.Duplicate());
    buffers.input_buffers[input_name] = std::move(duplicated);
  }

  for (const auto& [input_name, state_buffer] : *output_bank) {
    LITERT_ASSIGN_OR_RETURN(auto duplicated, state_buffer.buffer.Duplicate());
    buffers.output_buffers[input_name] = std::move(duplicated);
  }
  return buffers;
}

absl::Status LitertState::SyncShapes(CompiledModel& compiled_model,
                                     absl::string_view signature_name) {
  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, num_entries_));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::HeuristicBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name, AllocationPolicy allocation_policy,
    int batch_size, bool clear_kv_cache_before_prefill) {
  LITERT_ASSIGN_OR_RETURN(
      auto sig_input_names,
      compiled_model.GetSignatureInputNames(signature_name));
  LITERT_ASSIGN_OR_RETURN(
      auto sig_output_names,
      compiled_model.GetSignatureOutputNames(signature_name));

  std::string k_root_name;
  std::string v_root_name;
  ABSL_RETURN_IF_ERROR(GetKVCacheRootNames(sig_input_names, sig_output_names,
                                           k_root_name, v_root_name));

  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;
  std::string mask_input_name;
  for (auto input_name : sig_input_names) {
    if (absl::StartsWith(input_name, k_root_name) ||
        absl::StartsWith(input_name, "kv_cache_c_")) {
      key_cache_input_names.push_back(std::string(input_name));
    }
    if (absl::StartsWith(input_name, v_root_name) ||
        absl::StartsWith(input_name, "kv_cache_c_")) {
      value_cache_input_names.push_back(std::string(input_name));
    }
    if (absl::StrContains(input_name, "mask")) {
      mask_input_name = input_name;
    }
  }
  if (key_cache_input_names.empty() || value_cache_input_names.empty()) {
    return absl::FailedPreconditionError("No KV cache inputs found.");
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<int> k_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           key_cache_input_names[0]));
  ABSL_ASSIGN_OR_RETURN(std::optional<int> v_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           value_cache_input_names[0]));
  RET_CHECK(k_dynamic_dim.has_value() == v_dynamic_dim.has_value());

  auto create_and_init_buffer =
      [&](absl::string_view input_name, proto::StateBuffer::Type type,
          const std::optional<int>& dynamic_dim, bool is_output,
          bool clear_buffer) -> absl::StatusOr<StateBuffer> {
    if (dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, /*new_value=*/1));
    }

    // LiteRT macros don't support trinary operators. Hence the following block.
    std::optional<TensorBuffer> buffer;
    if (is_output) {
      LITERT_ASSIGN_OR_RETURN(buffer, compiled_model.CreateOutputBuffer(
                                          signature_name, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          buffer, compiled_model.CreateInputBuffer(signature_name, input_name));
    }

    if (clear_buffer) {
      LITERT_RETURN_IF_ERROR(buffer->Clear());
    }
    return StateBuffer{
        .buffer = std::move(buffer.value()),
        .type = type,
        .dynamic_dim = dynamic_dim,
    };
  };

  const bool is_gpu_optimized =
      allocation_policy == AllocationPolicy::kGpuOptimizedInplace;
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;

  for (const auto& input_name : key_cache_input_names) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[input_name],
        create_and_init_buffer(input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
                               k_dynamic_dim, /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }
  for (const auto& input_name : value_cache_input_names) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[input_name],
        create_and_init_buffer(input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
                               v_dynamic_dim, /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    auto& bank_2 = bank_2_state_buffers.emplace();
    for (const auto& input_name : key_cache_input_names) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[input_name],
          create_and_init_buffer(
              input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
              /*dynamic_dim=*/std::nullopt, /*is_output=*/true,
              /*clear_buffer=*/false));
    }
    for (const auto& input_name : value_cache_input_names) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[input_name],
          create_and_init_buffer(
              input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
              /*dynamic_dim=*/std::nullopt, /*is_output=*/true,
              /*clear_buffer=*/false));
    }
  }

  int context_size = 1;
  const bool is_dynamic_kv_cache = k_dynamic_dim.has_value();
  if (!is_dynamic_kv_cache) {
    if (!mask_input_name.empty()) {
      // Mask is our best bet for inferring context size. Key and value tensors
      // have different layouts and as such cannot be used directly.
      LITERT_ASSIGN_OR_RETURN(
          const RankedTensorType mask_tensor_type,
          compiled_model.GetInputTensorType(signature_name, mask_input_name));
      auto dims = mask_tensor_type.Layout().Dimensions();
      // Expect [1, 1, Sequence, KV Length]
      RET_CHECK_EQ(dims.size(), 4);
      context_size = dims[3];
    } else {
      // Fallback: get capacity from key cache tensor layout.
      // Usually key cache layout is [batch, num_heads, sequence_length,
      // head_dim]
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType k_tensor_type,
                              compiled_model.GetInputTensorType(
                                  signature_name, key_cache_input_names[0]));
      auto dims = k_tensor_type.Layout().Dimensions();
      RET_CHECK_GE(dims.size(), 3);
      context_size = dims[2];
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, env, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy));
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::MetadataBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata& executor_metadata,
    AllocationPolicy allocation_policy, int batch_size,
    bool clear_kv_cache_before_prefill) {
  struct StateBufferMeta {
    std::string name;
    proto::StateBuffer::Type type;
    std::optional<int> dynamic_dim;
    bool is_global;
  };

  std::vector<StateBufferMeta> state_buffer_metas;
  std::optional<int> max_supported_sequence_size;

  for (const auto& state_buffer :
       executor_metadata.llm_executor_metadata().state_buffers()) {
    std::vector<absl::string_view> names;
    if (!state_buffer.prefill_input_name().empty()) {
      names.push_back(state_buffer.prefill_input_name());
    }
    if (!state_buffer.prefill_output_name().empty()) {
      names.push_back(state_buffer.prefill_output_name());
    }
    if (!state_buffer.decode_input_name().empty()) {
      names.push_back(state_buffer.decode_input_name());
    }
    if (!state_buffer.decode_output_name().empty()) {
      names.push_back(state_buffer.decode_output_name());
    }
    for (size_t i = 1; i < names.size(); ++i) {
      RET_CHECK_EQ(names[0], names[i])
          << "Current implementation requires all state names in StateBuffer"
             " to be the same: "
          << names[0] << " vs " << names[i];
    }

    RET_CHECK(!names.empty()) << "At least one state name must be defined";
    std::string input_name = std::string(names[0]);

    bool is_global = false;
    switch (state_buffer.type()) {
      case proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_GLOBAL_VALUE_CACHE:
        is_global = true;
        break;
      case proto::StateBuffer::TYPE_LOCAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE:
      case proto::StateBuffer::TYPE_LINEAR_ATTENTION:
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported state buffer type: ", state_buffer.type()));
    }

    LITERT_ASSIGN_OR_RETURN(
        const RankedTensorType ranked_tensor_type,
        compiled_model.GetInputTensorType(signature_name, input_name));
    auto dimensions = ranked_tensor_type.Layout().Dimensions();

    std::optional<int> dynamic_dim;
    if (state_buffer.has_sequence_axis()) {
      int axis = state_buffer.sequence_axis();
      RET_CHECK_GE(axis, 0);
      RET_CHECK_LT(axis, dimensions.size());
      if (dimensions[axis] == kDynamicDimValue) {
        dynamic_dim = axis;
      } else if (is_global) {
        int seq_size = dimensions[axis];
        if (max_supported_sequence_size.has_value()) {
          max_supported_sequence_size =
              std::min(*max_supported_sequence_size, seq_size);
        } else {
          max_supported_sequence_size = seq_size;
        }
      }
    } else {
      RET_CHECK(state_buffer.type() ==
                proto::StateBuffer::TYPE_LINEAR_ATTENTION)
          << "Sequence axis must be defined for state buffers in the current "
             "implementation.";
    }

    state_buffer_metas.push_back(StateBufferMeta{
        .name = std::move(input_name),
        .type = state_buffer.type(),
        .dynamic_dim = dynamic_dim,
        .is_global = is_global,
    });
  }

  if (state_buffer_metas.empty()) {
    return absl::InvalidArgumentError(
        "No state buffers found for the current signature");
  }

  auto create_and_init_buffer =
      [&](absl::string_view input_name, proto::StateBuffer::Type type,
          const std::optional<int>& dynamic_dim, bool is_output,
          bool clear_buffer) -> absl::StatusOr<StateBuffer> {
    if (dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, /*new_value=*/1));
    }

    // LiteRT macros don't support trinary operators. Hence the following block.
    std::optional<TensorBuffer> buffer;
    if (is_output) {
      LITERT_ASSIGN_OR_RETURN(buffer, compiled_model.CreateOutputBuffer(
                                          signature_name, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          buffer, compiled_model.CreateInputBuffer(signature_name, input_name));
    }
    if (clear_buffer) {
      LITERT_RETURN_IF_ERROR(buffer->Clear());
    }
    return StateBuffer{
        .buffer = std::move(buffer.value()),
        .type = type,
        .dynamic_dim = dynamic_dim,
    };
  };

  const bool is_gpu_optimized =
      allocation_policy == AllocationPolicy::kGpuOptimizedInplace;
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;
  for (const auto& meta : state_buffer_metas) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[meta.name],
        create_and_init_buffer(meta.name, meta.type, meta.dynamic_dim,
                               /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    auto& bank_2 = bank_2_state_buffers.emplace();
    for (const auto& meta : state_buffer_metas) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[meta.name],
          create_and_init_buffer(meta.name, meta.type,
                                 /*dynamic_dim=*/std::nullopt,
                                 /*is_output=*/true,
                                 /*clear_buffer=*/false));
    }
  }

  int context_size = 1;
  bool is_dynamic = false;
  for (const auto& meta : state_buffer_metas) {
    if (meta.dynamic_dim.has_value()) {
      is_dynamic = true;
      break;
    }
  }
  if (!is_dynamic) {
    if (max_supported_sequence_size.has_value()) {
      context_size = *max_supported_sequence_size;
    } else {
      context_size = std::numeric_limits<int>::max();
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, env, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy));
}

}  // namespace litert::lm
