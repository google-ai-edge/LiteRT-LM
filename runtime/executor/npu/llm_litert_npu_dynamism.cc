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

#include "runtime/executor/npu/llm_litert_npu_dynamism.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert

#if defined(LITERT_ENABLE_FABRIC_INTEGRATION)

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_mask.h"

#endif  // LITERT_ENABLE_FABRIC_INTEGRATION

namespace litert::lm {

#if defined(LITERT_ENABLE_FABRIC_INTEGRATION)

namespace {
inline constexpr absl::string_view kDynamicKvCacheKRootName = "kv_cache_k";
inline constexpr absl::string_view kDynamicKvCacheVRootName = "kv_cache_v";
inline constexpr absl::string_view kDynamicKvCacheCRootName = "kv_cache_c";
}  // namespace

bool NpuDynamismHelper::HasDynamicKVCache(const litert::Model& model) {
  size_t num_sigs = model.GetNumSignatures();
  for (size_t i = 0; i < num_sigs; ++i) {
    auto signature_expected = model.GetSignature(i);
    if (!signature_expected.HasValue()) continue;
    const auto& signature = signature_expected.Value();
    for (auto input_name : signature.InputNames()) {
      if (absl::StartsWith(input_name, kDynamicKvCacheKRootName) ||
          absl::StartsWith(input_name, kDynamicKvCacheVRootName) ||
          absl::StartsWith(input_name, kDynamicKvCacheCRootName)) {
        auto tensor_expected = signature.InputTensor(input_name);
        if (!tensor_expected.HasValue()) continue;
        auto type_expected = tensor_expected->RankedTensorType();
        if (!type_expected.HasValue()) continue;
        auto dims = type_expected->Layout().Dimensions();
        for (int32_t dim : dims) {
          if (dim == -1) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

absl::StatusOr<int> NpuDynamismHelper::ResizeDynamicInputs(
    ::litert::CompiledModel& compiled_model, absl::string_view signature_name,
    int target_context_size) {
  if (target_context_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "target_context_size must be positive, got: ", target_context_size));
  }
  if (signature_name.empty()) {
    return absl::InvalidArgumentError("signature_name cannot be empty");
  }
  auto sig_res = compiled_model.GetSignatureIndex(signature_name);
  if (!sig_res.HasValue()) {
    return absl::NotFoundError(absl::StrCat(
        "Signature not found in compiled model: ", signature_name));
  }
  size_t sig_idx = sig_res.Value();
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          compiled_model.GetSignatureInputNames(sig_idx));
  bool resized_any = false;
  for (size_t i = 0; i < input_names.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto layout,
                            compiled_model.GetInputTensorLayout(sig_idx, i));
    auto dims = layout.Dimensions();
    absl::string_view name = input_names[i];
    std::vector<int32_t> new_dims(dims.begin(), dims.end());
    bool should_resize = false;
    // We assume for K the context/cache size is dimension index 2 (third
    // entry), and for V it is transposed and is dimension index 3 (fourth
    // entry).
    // TODO: In the future, this layout information should be provided and
    // read from model metadata instead of hardcoding these assumptions.
    if (absl::StartsWith(name, kDynamicKvCacheKRootName) &&
        new_dims.size() >= 4) {
      new_dims[2] = target_context_size;
      should_resize = true;
    } else if (absl::StartsWith(name, kDynamicKvCacheVRootName) &&
               new_dims.size() >= 4) {
      new_dims[3] = target_context_size;
      should_resize = true;
    }
    if (should_resize) {
      auto res = compiled_model.ResizeInputTensor(sig_idx, name, new_dims);
      if (!res.HasValue()) {
        ABSL_LOG(WARNING) << "ResizeInputTensor failed for " << name << ": "
                          << res.Error().Message();
      } else {
        resized_any = true;
      }
    }
  }
  if (resized_any) {
    LITERT_ASSIGN_OR_RETURN(auto _layouts,
                            compiled_model.GetOutputTensorLayouts(
                                sig_idx, /*update_allocation=*/true));
  }

  int actual_context_size = target_context_size;
  for (size_t i = 0; i < input_names.size(); ++i) {
    absl::string_view name = input_names[i];
    if (absl::StartsWith(name, kDynamicKvCacheKRootName) ||
        absl::StartsWith(name, kDynamicKvCacheVRootName)) {
      LITERT_ASSIGN_OR_RETURN(auto layout,
                              compiled_model.GetInputTensorLayout(sig_idx, i));
      auto dims = layout.Dimensions();
      if (dims.size() >= 4) {
        // Query the actual context length dimension from the layout (dim index
        // 2 for K, dim index 3 for V).
        int dim = absl::StartsWith(name, kDynamicKvCacheKRootName) ? dims[2]
                                                                   : dims[3];
        if (dim > 0) {
          actual_context_size = std::max(actual_context_size, dim);
        }
      }
    }
  }

  return actual_context_size;
}

absl::Status NpuDynamismHelper::ResizeDynamicMaskInputs(
    ::litert::CompiledModel& aux_compiled_model,
    absl::string_view signature_name, int actual_context_size) {
  if (actual_context_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "actual_context_size must be positive, got: ", actual_context_size));
  }
  if (signature_name.empty()) {
    return absl::InvalidArgumentError("signature_name cannot be empty");
  }
  auto sig_res = aux_compiled_model.GetSignatureIndex(signature_name);
  if (!sig_res.HasValue()) {
    return absl::NotFoundError(absl::StrCat(
        "Signature not found in aux compiled model: ", signature_name));
  }
  size_t sig_idx = sig_res.Value();
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          aux_compiled_model.GetSignatureInputNames(sig_idx));
  bool resized_any = false;
  for (size_t i = 0; i < input_names.size(); ++i) {
    absl::string_view name = input_names[i];
    if (name == MaskSignatures::kMaskLocalContextLength ||
        name == MaskSignatures::kMaskGlobalContextLength) {
      LITERT_ASSIGN_OR_RETURN(
          auto layout, aux_compiled_model.GetInputTensorLayout(sig_idx, i));
      auto dims = layout.Dimensions();
      std::vector<int32_t> new_dims(dims.begin(), dims.end());
      if (new_dims.size() == 1 && new_dims[0] == -1) {
        new_dims[0] = actual_context_size;
        auto res =
            aux_compiled_model.ResizeInputTensor(sig_idx, name, new_dims);
        if (!res.HasValue()) {
          ABSL_LOG(WARNING) << "ResizeInputTensor failed for " << name << ": "
                            << res.Error().Message();
        } else {
          resized_any = true;
        }
      }
    }
  }
  if (resized_any) {
    LITERT_ASSIGN_OR_RETURN(auto _layouts,
                            aux_compiled_model.GetOutputTensorLayouts(
                                sig_idx, /*update_allocation=*/true));
  }
  return absl::OkStatus();
}

#else  // !defined(LITERT_ENABLE_FABRIC_INTEGRATION)

bool NpuDynamismHelper::HasDynamicKVCache(const litert::Model& /*model*/) {
  return false;
}

absl::StatusOr<int> NpuDynamismHelper::ResizeDynamicInputs(
    ::litert::CompiledModel& /*compiled_model*/,
    absl::string_view /*signature_name*/, int target_context_size) {
  return target_context_size;
}

absl::Status NpuDynamismHelper::ResizeDynamicMaskInputs(
    ::litert::CompiledModel& /*aux_compiled_model*/,
    absl::string_view /*signature_name*/, int /*actual_context_size*/) {
  return absl::OkStatus();
}

#endif  // LITERT_ENABLE_FABRIC_INTEGRATION

}  // namespace litert::lm
