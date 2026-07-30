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
    AllocationPolicy allocation_policy, int batch_size) {
  if (executor_metadata == nullptr) {
    return HeuristicBasedCreate(env, compiled_model, signature_name,
                                allocation_policy, batch_size);
  }

  return MetadataBasedCreate(env, compiled_model, signature_name,
                             *executor_metadata, allocation_policy, batch_size);
}

absl::Status LitertState::SelectAndCopyFrom(StateInterface& other,
                                            int batch_index) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_key_cache_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_key_cache_buffers_.has_value());
  RET_CHECK_GT(other_litert->batch_size_, batch_size_);
  RET_CHECK_LT(batch_index, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);
  RET_CHECK_EQ(k_dynamic_dim_.has_value(),
               other_litert->k_dynamic_dim_.has_value());
  RET_CHECK_EQ(v_dynamic_dim_.has_value(),
               other_litert->v_dynamic_dim_.has_value());

  for (auto& [input_name, key_cache_buffer] : bank_1_key_cache_buffers_) {
    RET_CHECK(other_litert->bank_1_key_cache_buffers_.contains(input_name));
    ABSL_RETURN_IF_ERROR(SelectAndCopyBuffer(
        key_cache_buffer, other_litert->bank_1_key_cache_buffers_[input_name],
        batch_index));
  }
  for (auto& [input_name, value_cache_buffer] : bank_1_value_cache_buffers_) {
    RET_CHECK(other_litert->bank_1_value_cache_buffers_.contains(input_name));
    ABSL_RETURN_IF_ERROR(SelectAndCopyBuffer(
        value_cache_buffer,
        other_litert->bank_1_value_cache_buffers_[input_name], batch_index));
  }
  return absl::OkStatus();
}

absl::Status LitertState::BroadcastAndCopyFrom(StateInterface& other) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_key_cache_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_key_cache_buffers_.has_value());
  RET_CHECK_EQ(other_litert->batch_size_, 1);
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);
  RET_CHECK_EQ(k_dynamic_dim_.has_value(),
               other_litert->k_dynamic_dim_.has_value());
  RET_CHECK_EQ(v_dynamic_dim_.has_value(),
               other_litert->v_dynamic_dim_.has_value());

  for (auto& [input_name, key_cache_buffer] : bank_1_key_cache_buffers_) {
    RET_CHECK(other_litert->bank_1_key_cache_buffers_.contains(input_name));
    ABSL_RETURN_IF_ERROR(BroadcastAndCopyBuffer(
        key_cache_buffer, batch_size_,
        other_litert->bank_1_key_cache_buffers_[input_name]));
  }
  for (auto& [input_name, value_cache_buffer] : bank_1_value_cache_buffers_) {
    RET_CHECK(other_litert->bank_1_value_cache_buffers_.contains(input_name));
    ABSL_RETURN_IF_ERROR(BroadcastAndCopyBuffer(
        value_cache_buffer, batch_size_,
        other_litert->bank_1_value_cache_buffers_[input_name]));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DeepCopy() const {
  absl::flat_hash_map<std::string, TensorBuffer> bank_1_key_cache_buffers;
  for (const auto& [name, buffer] : bank_1_key_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(bank_1_key_cache_buffers[name],
                            CopyTensorBuffer(env_, buffer));
  }
  absl::flat_hash_map<std::string, TensorBuffer> bank_1_value_cache_buffers;
  for (const auto& [name, buffer] : bank_1_value_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(bank_1_value_cache_buffers[name],
                            CopyTensorBuffer(env_, buffer));
  }

  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_key_cache_buffers;
  if (bank_2_key_cache_buffers_.has_value()) {
    bank_2_key_cache_buffers.emplace();
    auto& map = *bank_2_key_cache_buffers;
    for (const auto& [name, buffer] : *bank_2_key_cache_buffers_) {
      LITERT_ASSIGN_OR_RETURN(map[name], CopyTensorBuffer(env_, buffer));
    }
  }

  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_value_cache_buffers;
  if (bank_2_value_cache_buffers_.has_value()) {
    bank_2_value_cache_buffers.emplace();
    auto& map = *bank_2_value_cache_buffers;
    for (const auto& [name, buffer] : *bank_2_value_cache_buffers_) {
      LITERT_ASSIGN_OR_RETURN(map[name], CopyTensorBuffer(env_, buffer));
    }
  }

  auto copy = absl::WrapUnique(new LitertState(
      batch_size_, num_entries_, k_dynamic_dim_, v_dynamic_dim_, env_,
      std::move(bank_1_key_cache_buffers),
      std::move(bank_1_value_cache_buffers),
      std::move(bank_2_key_cache_buffers),
      std::move(bank_2_value_cache_buffers), allocation_policy_));
  copy->bank_1_is_input_ = bank_1_is_input_;

  return copy;
}

bool LitertState::Contains(absl::string_view tensor_name) const {
  return bank_1_key_cache_buffers_.contains(tensor_name) ||
         bank_1_value_cache_buffers_.contains(tensor_name);
}

absl::Status LitertState::Clear() {
  for (auto& [_, buffer] : bank_1_key_cache_buffers_) {
    LITERT_RETURN_IF_ERROR(buffer.Clear());
  }
  for (auto& [_, buffer] : bank_1_value_cache_buffers_) {
    LITERT_RETURN_IF_ERROR(buffer.Clear());
  }
  if (bank_2_key_cache_buffers_.has_value()) {
    for (auto& [_, buffer] : *bank_2_key_cache_buffers_) {
      LITERT_RETURN_IF_ERROR(buffer.Clear());
    }
  }
  if (bank_2_value_cache_buffers_.has_value()) {
    for (auto& [_, buffer] : *bank_2_value_cache_buffers_) {
      LITERT_RETURN_IF_ERROR(buffer.Clear());
    }
  }
  return absl::OkStatus();
}

absl::Status LitertState::Resize(CompiledModel& compiled_model,
                                 absl::string_view signature_name,
                                 int num_entries) {
  RET_CHECK(!bank_2_key_cache_buffers_.has_value())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Out of place KV cache cannot be resized.";
  if (!k_dynamic_dim_.has_value() || !v_dynamic_dim_.has_value()) {
    return absl::InvalidArgumentError(
        "KV cache is not dynamic and cannot be resized.");
  }

  int entries_to_add = num_entries - num_entries_;
  if (entries_to_add <= 0) {
    return absl::OkStatus();
  }

  for (const auto& [input_name, _] : bank_1_key_cache_buffers_) {
    ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                             input_name, num_entries));
  }
  for (const auto& [input_name, _] : bank_1_value_cache_buffers_) {
    ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                             input_name, num_entries));
  }

  for (auto& [input_name, key_cache_buffer] : bank_1_key_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(
        key_cache_buffer,
        ResizeTensorBuffer(env_, key_cache_buffer, k_dynamic_dim_.value(),
                           entries_to_add));
  }
  for (auto& [input_name, value_cache_buffer] : bank_1_value_cache_buffers_) {
    LITERT_ASSIGN_OR_RETURN(
        value_cache_buffer,
        ResizeTensorBuffer(env_, value_cache_buffer, v_dynamic_dim_.value(),
                           entries_to_add));
  }
  num_entries_ = num_entries;
  return absl::OkStatus();
}

absl::StatusOr<LitertState::StateBuffers> LitertState::GetStateBuffers(
    CompiledModel& compiled_model, absl::string_view signature_name) {
  LITERT_RETURN_IF_ERROR(SyncShapes(compiled_model, signature_name));
  auto* input_bank_key = &bank_1_key_cache_buffers_;
  auto* input_bank_value = &bank_1_value_cache_buffers_;
  auto* output_bank_key = &bank_1_key_cache_buffers_;
  auto* output_bank_value = &bank_1_value_cache_buffers_;

  if (bank_2_key_cache_buffers_.has_value()) {
    if (bank_1_is_input_) {
      output_bank_key = &bank_2_key_cache_buffers_.value();
      output_bank_value = &bank_2_value_cache_buffers_.value();
    } else {
      input_bank_key = &bank_2_key_cache_buffers_.value();
      input_bank_value = &bank_2_value_cache_buffers_.value();
    }
    bank_1_is_input_ = !bank_1_is_input_;
  }

  StateBuffers buffers;
  const bool skip_inputs =
      allocation_policy_ == AllocationPolicy::kGpuOptimizedInplace;
  if (!skip_inputs) {
    for (const auto& [input_name, key_cache_buffer] : *input_bank_key) {
      LITERT_ASSIGN_OR_RETURN(auto duplicated, key_cache_buffer.Duplicate());
      buffers.input_buffers[input_name] = std::move(duplicated);
    }
    for (const auto& [input_name, value_cache_buffer] : *input_bank_value) {
      LITERT_ASSIGN_OR_RETURN(auto duplicated, value_cache_buffer.Duplicate());
      buffers.input_buffers[input_name] = std::move(duplicated);
    }
  }

  for (const auto& [input_name, key_cache_buffer] : *output_bank_key) {
    LITERT_ASSIGN_OR_RETURN(auto duplicated, key_cache_buffer.Duplicate());
    buffers.output_buffers[input_name] = std::move(duplicated);
  }
  for (const auto& [input_name, value_cache_buffer] : *output_bank_value) {
    LITERT_ASSIGN_OR_RETURN(auto duplicated, value_cache_buffer.Duplicate());
    buffers.output_buffers[input_name] = std::move(duplicated);
  }
  return buffers;
}

absl::Status LitertState::SyncShapes(CompiledModel& compiled_model,
                                     absl::string_view signature_name) {
  bool is_static = !k_dynamic_dim_.has_value() && !v_dynamic_dim_.has_value();
  if (is_static) {
    return absl::OkStatus();
  }
  for (const auto& [input_name, _] : bank_1_key_cache_buffers_) {
    ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                             input_name, num_entries_));
  }
  for (const auto& [input_name, _] : bank_1_value_cache_buffers_) {
    ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                             input_name, num_entries_));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::HeuristicBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name, AllocationPolicy allocation_policy,
    int batch_size) {
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  LITERT_ASSIGN_OR_RETURN(
      auto sig_input_names,
      compiled_model.GetSignatureInputNames(signature_name));
  LITERT_ASSIGN_OR_RETURN(
      auto sig_output_names,
      compiled_model.GetSignatureOutputNames(signature_name));
  ABSL_RETURN_IF_ERROR(GetKVCacheRootNames(sig_input_names, sig_output_names,
                                           kv_cache_k_root_name,
                                           kv_cache_v_root_name));

  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;
  std::string mask_input_name;
  for (auto input_name : sig_input_names) {
    bool is_key_cache_input =
        absl::StartsWith(input_name, kv_cache_k_root_name);
    if (is_key_cache_input) {
      key_cache_input_names.push_back(std::string(input_name));
    }

    bool is_value_cache_input =
        absl::StartsWith(input_name, kv_cache_v_root_name);
    if (is_value_cache_input) {
      value_cache_input_names.push_back(std::string(input_name));
    }

    if (absl::StrContains(input_name, "mask")) {
      mask_input_name = input_name;
    }
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<int> k_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           key_cache_input_names[0]));
  ABSL_ASSIGN_OR_RETURN(std::optional<int> v_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           value_cache_input_names[0]));
  RET_CHECK(k_dynamic_dim.has_value() == v_dynamic_dim.has_value());

  auto create_and_init_buffers =
      [&](const std::vector<std::string>& input_names,
          const std::optional<int>& dynamic_dim, bool clear_buffer)
      -> absl::StatusOr<absl::flat_hash_map<std::string, TensorBuffer>> {
    absl::flat_hash_map<std::string, TensorBuffer> buffers;
    for (const auto& input_name : input_names) {
      if (dynamic_dim.has_value()) {
        ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                                 input_name,
                                                 /*new_value=*/1));
      }
      LITERT_ASSIGN_OR_RETURN(auto buffer, compiled_model.CreateInputBuffer(
                                               signature_name, input_name));
      if (clear_buffer) {
        LITERT_RETURN_IF_ERROR(buffer.Clear());
      }
      buffers[input_name] = std::move(buffer);
    }
    return buffers;
  };

  auto create_and_init_output_buffers =
      [&](const std::vector<std::string>& input_names,
          const std::optional<int>& dynamic_dim, bool clear_buffer)
      -> absl::StatusOr<absl::flat_hash_map<std::string, TensorBuffer>> {
    absl::flat_hash_map<std::string, TensorBuffer> buffers;
    for (const auto& input_name : input_names) {
      if (dynamic_dim.has_value()) {
        ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                                 input_name,
                                                 /*new_value=*/1));
      }
      LITERT_ASSIGN_OR_RETURN(auto buffer, compiled_model.CreateOutputBuffer(
                                               signature_name, input_name));
      if (clear_buffer) {
        LITERT_RETURN_IF_ERROR(buffer.Clear());
      }
      buffers[input_name] = std::move(buffer);
    }
    return buffers;
  };

  absl::flat_hash_map<std::string, TensorBuffer> bank_1_key_cache_buffers;
  absl::flat_hash_map<std::string, TensorBuffer> bank_1_value_cache_buffers;
  if (allocation_policy == AllocationPolicy::kGpuOptimizedInplace) {
    ABSL_ASSIGN_OR_RETURN(
        bank_1_key_cache_buffers,
        create_and_init_output_buffers(key_cache_input_names, k_dynamic_dim,
                                       /*clear_buffer=*/true));
    ABSL_ASSIGN_OR_RETURN(
        bank_1_value_cache_buffers,
        create_and_init_output_buffers(value_cache_input_names, v_dynamic_dim,
                                       /*clear_buffer=*/true));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        bank_1_key_cache_buffers,
        create_and_init_buffers(key_cache_input_names, k_dynamic_dim,
                                /*clear_buffer=*/true));
    ABSL_ASSIGN_OR_RETURN(
        bank_1_value_cache_buffers,
        create_and_init_buffers(value_cache_input_names, v_dynamic_dim,
                                /*clear_buffer=*/true));
  }

  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_key_cache_buffers;
  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_value_cache_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    // Bank 2 buffers are created after Bank 1 shapes are resolved, so no need
    // to pass dynamic_dim.
    ABSL_ASSIGN_OR_RETURN(bank_2_key_cache_buffers.emplace(),
                          create_and_init_buffers(key_cache_input_names,
                                                  /*dynamic_dim=*/std::nullopt,
                                                  /*clear_buffer=*/true));
    ABSL_ASSIGN_OR_RETURN(bank_2_value_cache_buffers.emplace(),
                          create_and_init_buffers(value_cache_input_names,
                                                  /*dynamic_dim=*/std::nullopt,
                                                  /*clear_buffer=*/false));
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
      batch_size, context_size, k_dynamic_dim, v_dynamic_dim, env,
      std::move(bank_1_key_cache_buffers),
      std::move(bank_1_value_cache_buffers),
      std::move(bank_2_key_cache_buffers),
      std::move(bank_2_value_cache_buffers), allocation_policy));
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::MetadataBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata& executor_metadata,
    AllocationPolicy allocation_policy, int batch_size) {
  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;

  std::optional<int> k_dynamic_dim;
  std::optional<int> v_dynamic_dim;
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

    bool is_key = false;
    bool is_value = false;
    bool is_global = false;

    switch (state_buffer.type()) {
      case proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE:
        is_key = true;
        is_global = true;
        break;
      case proto::StateBuffer::TYPE_LOCAL_KEY_CACHE:
        is_key = true;
        break;
      case proto::StateBuffer::TYPE_GLOBAL_VALUE_CACHE:
        is_value = true;
        is_global = true;
        break;
      case proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE:
        is_value = true;
        break;
      case proto::StateBuffer::TYPE_LINEAR_ATTENTION:
        is_key = true;
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

    if (is_key) {
      key_cache_input_names.push_back(input_name);
      if (dynamic_dim.has_value()) {
        if (k_dynamic_dim.has_value() &&
            k_dynamic_dim.value() != dynamic_dim.value()) {
          return absl::InvalidArgumentError(
              "Mismatched sequence axis for key cache buffers");
        }
        k_dynamic_dim = dynamic_dim;
      }
    } else if (is_value) {
      value_cache_input_names.push_back(input_name);
      if (dynamic_dim.has_value()) {
        if (v_dynamic_dim.has_value() &&
            v_dynamic_dim.value() != dynamic_dim.value()) {
          return absl::InvalidArgumentError(
              "Mismatched sequence axis for value cache buffers");
        }
        v_dynamic_dim = dynamic_dim;
      }
    }
  }

  if (key_cache_input_names.empty() && value_cache_input_names.empty()) {
    return absl::InvalidArgumentError(
        "No state buffers found for the current signature");
  }

  auto create_and_init_buffers =
      [&](const std::vector<std::string>& input_names,
          const std::optional<int>& dynamic_dim, bool clear_buffer)
      -> absl::StatusOr<absl::flat_hash_map<std::string, TensorBuffer>> {
    absl::flat_hash_map<std::string, TensorBuffer> buffers;
    for (const auto& input_name : input_names) {
      if (dynamic_dim.has_value()) {
        ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                                 input_name,
                                                 /*new_value=*/1));
      }
      LITERT_ASSIGN_OR_RETURN(auto buffer, compiled_model.CreateInputBuffer(
                                               signature_name, input_name));
      if (clear_buffer) {
        LITERT_RETURN_IF_ERROR(buffer.Clear());
      }
      buffers[input_name] = std::move(buffer);
    }
    return buffers;
  };

  auto create_and_init_output_buffers =
      [&](const std::vector<std::string>& input_names,
          const std::optional<int>& dynamic_dim, bool clear_buffer)
      -> absl::StatusOr<absl::flat_hash_map<std::string, TensorBuffer>> {
    absl::flat_hash_map<std::string, TensorBuffer> buffers;
    for (const auto& input_name : input_names) {
      if (dynamic_dim.has_value()) {
        ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                                 input_name,
                                                 /*new_value=*/1));
      }
      LITERT_ASSIGN_OR_RETURN(auto buffer, compiled_model.CreateOutputBuffer(
                                               signature_name, input_name));
      if (clear_buffer) {
        LITERT_RETURN_IF_ERROR(buffer.Clear());
      }
      buffers[input_name] = std::move(buffer);
    }
    return buffers;
  };

  absl::flat_hash_map<std::string, TensorBuffer> bank_1_key_cache_buffers;
  absl::flat_hash_map<std::string, TensorBuffer> bank_1_value_cache_buffers;
  if (allocation_policy == AllocationPolicy::kGpuOptimizedInplace) {
    ABSL_ASSIGN_OR_RETURN(
        bank_1_key_cache_buffers,
        create_and_init_output_buffers(key_cache_input_names, k_dynamic_dim,
                                       /*clear_buffer=*/true));
    ABSL_ASSIGN_OR_RETURN(
        bank_1_value_cache_buffers,
        create_and_init_output_buffers(value_cache_input_names, v_dynamic_dim,
                                       /*clear_buffer=*/true));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        bank_1_key_cache_buffers,
        create_and_init_buffers(key_cache_input_names, k_dynamic_dim,
                                /*clear_buffer=*/true));
    ABSL_ASSIGN_OR_RETURN(
        bank_1_value_cache_buffers,
        create_and_init_buffers(value_cache_input_names, v_dynamic_dim,
                                /*clear_buffer=*/true));
  }

  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_key_cache_buffers;
  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_value_cache_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    ABSL_ASSIGN_OR_RETURN(bank_2_key_cache_buffers.emplace(),
                          create_and_init_buffers(key_cache_input_names,
                                                  /*dynamic_dim=*/std::nullopt,
                                                  /*clear_buffer=*/true));
    ABSL_ASSIGN_OR_RETURN(bank_2_value_cache_buffers.emplace(),
                          create_and_init_buffers(value_cache_input_names,
                                                  /*dynamic_dim=*/std::nullopt,
                                                  /*clear_buffer=*/false));
  }

  int context_size = 1;
  const bool is_dynamic_kv_cache = k_dynamic_dim.has_value();
  if (!is_dynamic_kv_cache) {
    if (max_supported_sequence_size.has_value()) {
      context_size = *max_supported_sequence_size;
    } else {
      context_size = std::numeric_limits<int>::max();
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, k_dynamic_dim, v_dynamic_dim, env,
      std::move(bank_1_key_cache_buffers),
      std::move(bank_1_value_cache_buffers),
      std::move(bank_2_key_cache_buffers),
      std::move(bank_2_value_cache_buffers), allocation_policy));
}

}  // namespace litert::lm
