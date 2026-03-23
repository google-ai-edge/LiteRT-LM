// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/litert_compiled_model_executor_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/embedding_lookup/embedding_lookup_text.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

namespace {

// The name of the prefill decode model in the task bundle.
constexpr char kPrefilDecodeModelNameInTaskBundle[] = "TF_LITE_PREFILL_DECODE";
// Possible input tokens names:
constexpr std::array<absl::string_view, 2> kInputTokensNames = {"token_ids",
                                                                "tokens"};
// Possible input positions names:
constexpr std::array<absl::string_view, 2> kInputPositionsNames = {"positions",
                                                                   "input_pos"};
// Possible input attention mask names:
constexpr std::array<absl::string_view, 2> kInputAttnMaskNames = {"attn_mask",
                                                                  "mask"};
// Possible embedding names:
constexpr std::array<absl::string_view, 1> kEmbeddingNames = {"embeddings"};
// Possible per layer embedding names:
constexpr std::array<absl::string_view, 1> kPerLayerEmbeddingNames = {
    "per_layer_embeddings"};
// Possible input int32 param names:
constexpr std::array<absl::string_view, 1> kInputInt32ParamNames = {
    "param_tensor"};
// Possible output logits names:
constexpr std::array<absl::string_view, 1> kOutputLogitsNames = {"logits"};

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildModelResourcesFromTaskFormat(const ModelAssets& model_assets) {
  std::unique_ptr<ModelAssetBundleResources> resources;
  if (model_assets.HasMemoryMappedFile()) {
    ASSIGN_OR_RETURN(auto memory_mapped_file,
                     model_assets.GetMemoryMappedFile());
    ASSIGN_OR_RETURN(resources, ModelAssetBundleResources::Create(
                                    /*tag=*/"", memory_mapped_file));
  } else {
    ASSIGN_OR_RETURN(auto scoped_file, model_assets.GetOrCreateScopedFile());
    ASSIGN_OR_RETURN(resources, ModelAssetBundleResources::Create(
                                    /*tag=*/"", scoped_file));
  }
  auto files_list = resources->ListFiles();
  RET_CHECK(std::find(files_list.begin(), files_list.end(),
                      kPrefilDecodeModelNameInTaskBundle) != files_list.end())
      << kPrefilDecodeModelNameInTaskBundle
      << " model file not found in task bundle.";
  return ModelResourcesTask::Create(std::move(resources));
}

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildModelResourcesFromLitertLmFormat(const ModelAssets& model_assets) {
  std::unique_ptr<LitertLmLoader> loader;
  if (model_assets.HasMemoryMappedFile()) {
    ASSIGN_OR_RETURN(auto memory_mapped_file,
                     model_assets.GetMemoryMappedFile());
    loader = std::make_unique<LitertLmLoader>(memory_mapped_file);
  } else {
    // `BuildModelResourcesFromLitertLmFormat` expects a ScopedFile that it
    // takes ownership of, so we need to duplicate the ScopedFile to keep
    // the original alive.
    ASSIGN_OR_RETURN(auto scoped_file, model_assets.GetOrCreateScopedFile());
    ASSIGN_OR_RETURN(auto duplicate_file, scoped_file->Duplicate());
    loader = std::make_unique<LitertLmLoader>(std::move(duplicate_file));
  }
  return ModelResourcesLitertLm::Create(std::move(loader));
}

}  // namespace

absl::StatusOr<ModelSignatures> GetModelSignaturesFromInputOutputNames(
    const std::vector<absl::string_view>& input_names,
    const std::vector<absl::string_view>& output_names, bool strict) {
  ModelSignatures model_signatures;
  for (auto input_name : input_names) {
    if (absl::c_linear_search(kInputTokensNames, input_name)) {
      model_signatures.input_tokens = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kInputPositionsNames, input_name)) {
      model_signatures.input_positions = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kInputAttnMaskNames, input_name)) {
      model_signatures.input_attn_mask = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kEmbeddingNames, input_name)) {
      model_signatures.input_embeddings = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kPerLayerEmbeddingNames, input_name)) {
      model_signatures.input_per_layer_embeddings = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kInputInt32ParamNames, input_name)) {
      model_signatures.input_int32_param = std::string(input_name);
      continue;
    }
  }

  for (auto output_name : output_names) {
    if (absl::c_linear_search(kOutputLogitsNames, output_name)) {
      model_signatures.output_logits = std::string(output_name);
      continue;
    }
  }

  if (strict) {
    RET_CHECK(!model_signatures.input_tokens.empty() ||
              model_signatures.input_embeddings.has_value())
            .SetCode(absl::StatusCode::kFailedPrecondition)
        << "Input tokens or embeddings not found.";
    RET_CHECK(!model_signatures.input_positions.empty())
            .SetCode(absl::StatusCode::kFailedPrecondition)
        << "Input positions not found.";
    RET_CHECK(!model_signatures.output_logits.empty())
            .SetCode(absl::StatusCode::kFailedPrecondition)
        << "Output logits not found.";
  }
  return model_signatures;
}

absl::Status GetKVCacheRootNames(std::vector<absl::string_view> input_names,
                                 std::vector<absl::string_view> output_names,
                                 std::string& k_root_name,
                                 std::string& v_root_name) {
  for (auto input_name : input_names) {
    if (input_name == "kv_cache_k_0") {
      k_root_name = "kv_cache_k_";
      v_root_name = "kv_cache_v_";
      return absl::OkStatus();
    } else if (input_name == "k_cache_0") {
      k_root_name = "k_cache_";
      v_root_name = "v_cache_";
      return absl::OkStatus();
    } else if (input_name == "kv_cache_c_0") {
      k_root_name = "kv_cache_c_";
      v_root_name = "kv_cache_c_";
      return absl::OkStatus();
    }
  }
  for (auto output_name : output_names) {
    if (output_name == "kv_cache_k_0") {
      k_root_name = "kv_cache_k_";
      v_root_name = "kv_cache_v_";
      return absl::OkStatus();
    } else if (output_name == "k_cache_0") {
      k_root_name = "k_cache_";
      v_root_name = "v_cache_";
      return absl::OkStatus();
    } else if (output_name == "kv_cache_c_0") {
      k_root_name = "kv_cache_c_";
      v_root_name = "kv_cache_c_";
      return absl::OkStatus();
    }
  }
  return absl::FailedPreconditionError("No KV cache inputs found.");
}

absl::StatusOr<SortedPrefillSignatureMap> GetPrefillRunnerSetFromModel(
    const ::litert::Model& model, absl::string_view signature_name_base,
    absl::string_view input_positions_name) {
  SortedPrefillSignatureMap prefill_runner_set;
  auto signatures = model.GetSignatures();
  for (auto& signature : *signatures) {
    if (auto signature_key = signature.Key();
        absl::StartsWith(signature_key, signature_name_base)) {
      LITERT_ASSIGN_OR_RETURN(auto input_positions_tensor,
                              signature.InputTensor(input_positions_name));
      LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                              input_positions_tensor.RankedTensorType());
      if (ranked_tensor_type.Layout().Rank() == 2) {
        // [batch_size, max_seq_len]
        prefill_runner_set[ranked_tensor_type.Layout().Dimensions()[1]] =
            std::string(signature_key);
      } else if (ranked_tensor_type.Layout().Rank() == 1) {
        // [max_seq_len]
        prefill_runner_set[ranked_tensor_type.Layout().Dimensions()[0]] =
            std::string(signature_key);
      } else {
        return absl::FailedPreconditionError(
            "Unsupported input tokens tensor dimension.");
      }
    }
  }
  return prefill_runner_set;
}

absl::StatusOr<std::vector<std::pair<std::string, int>>>
GetOptimizedPrefillWorkGroups(
    const SortedPrefillSignatureMap& prefill_runner_set, int input_length) {
  std::vector<std::pair<std::string, int>> work_groups;
  // Current strategy:
  // 1. Use the prefill runner with the largest sequence length, until the
  // remaining length is less than its sequence length.
  // 2. Finish the remaining length with one prefill call, using the runner with
  // the sequence length as small as possible.
  // TODO: b/378772479 - Improve this strategy once we have benchmarked costs.
  int max_seq_len = prefill_runner_set.begin()->first;
  while (input_length >= max_seq_len) {
    work_groups.push_back(
        std::make_pair(prefill_runner_set.begin()->second, max_seq_len));
    input_length -= max_seq_len;
  }
  if (input_length > 0) {
    for (auto it = prefill_runner_set.begin(); it != prefill_runner_set.end();
         ++it) {
      // If the next smaller runner can handle the remaining length, skip the
      // current runner.
      if (std::next(it) != prefill_runner_set.end() &&
          std::next(it)->first >= input_length) {
        continue;
      }
      work_groups.push_back(std::make_pair(it->second, input_length));
      break;
    }
  }
  return work_groups;
}

absl::Status InitializeAttentionMask(litert::TensorBuffer& mask, bool is_f16) {
  LITERT_ASSIGN_OR_RETURN(auto mask_size, mask.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto mask_tensor_type, mask.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto mask_lock_and_addr,
                          litert::TensorBufferScopedLock::Create(
                              mask, litert::TensorBuffer::LockMode::kWrite));

  switch (mask_tensor_type.ElementType()) {
    case litert::ElementType::Bool:
      // Boolean mask: Default value = false.
      memset(mask_lock_and_addr.second, 0, mask_size);
      break;
    case litert::ElementType::Float32: {
      // Float mask: Default value is based on precision.
      // Default value reference:
      // third_party/odml/infra/genai/inference/ml_drift/llm/tasks/apply_attention_mask_test_util.cc
      float* mask_ptr = static_cast<float*>(mask_lock_and_addr.second);
      std::fill(mask_ptr, mask_ptr + mask_size / sizeof(float),
                is_f16 ? -45824 : -0.7f * std::numeric_limits<float>::max());
      break;
    }
    case litert::ElementType::Float16: {
      // Float16 mask: Default value is -45824.
      // This value is approximately -0.7 * MaxFloat16 (65504).
      // 0.7 * 65504 = 45852.8. Truncated to 45824.
      // It provides a margin of ~19680 before overflowing to -inf.
      tflite::half* mask_ptr =
          static_cast<tflite::half*>(mask_lock_and_addr.second);
      std::fill(mask_ptr, mask_ptr + mask_size / sizeof(tflite::half),
                tflite::half(-45824.0f));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Unsupported attention mask data type.");
  }
  return absl::OkStatus();
}

absl::Status FillSingleBufferCacheParamTensor(
    litert::TensorBuffer& param_tensor, int start_index, int update_length) {
  // TODO(sulemanshahid): Local attention optimization is not supported in the
  // OpenCL implementation, enable for WebGPU.
  LITERT_ASSIGN_OR_RETURN(auto packed_size, param_tensor.PackedSize());
  LITERT_ASSIGN_OR_RETURN(
      auto param_tensor_lock_and_addr,
      TensorBufferScopedLock::Create(param_tensor,
                                     TensorBuffer::LockMode::kWrite));
  std::memset(param_tensor_lock_and_addr.second, 0, packed_size);

  // See parameter definition in ml_drift::LlmRuntimeParams.
  // First 2 parameters are used by add_values_to_cache kernel.
  // 3rd parameter is used by runtime_batched_matmul kernel to check the end
  // channel index, which doesn't have to be aligned as the kernel does that.
  int end_index = start_index + update_length;
  int32_t params[] = {start_index, end_index, end_index};
  LITERT_RETURN_IF_ERROR(sizeof(params) <= packed_size);
  std::memcpy(param_tensor_lock_and_addr.second, params, sizeof(params));
  return absl::OkStatus();
}

absl::Status FillAttentionMask(litert::TensorBuffer& mask, int start_timestep,
                               int steps) {
  LITERT_ASSIGN_OR_RETURN(auto mask_tensor_type, mask.TensorType());
  RET_CHECK_EQ(mask_tensor_type.Layout().Rank(), 4)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Attention mask must be 4D.";
  int batch_size = mask_tensor_type.Layout().Dimensions()[0];
  int channel_size = mask_tensor_type.Layout().Dimensions()[3];
  LITERT_ASSIGN_OR_RETURN(auto mask_size, mask.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto mask_lock_and_addr,
                          litert::TensorBufferScopedLock::Create(
                              mask, litert::TensorBuffer::LockMode::kWrite));

  int batch_offset = mask_size / batch_size;
  if (mask_tensor_type.ElementType() == litert::ElementType::Bool) {
    batch_offset /= sizeof(bool);
  } else if (mask_tensor_type.ElementType() == litert::ElementType::Float32) {
    batch_offset /= sizeof(float);
  } else if (mask_tensor_type.ElementType() == litert::ElementType::Float16) {
    batch_offset /= sizeof(tflite::half);
  } else {
    return absl::InvalidArgumentError("Unsupported attention mask data type.");
  }

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < steps; ++i) {
      int current_step = start_timestep + i;
      int offset = b * batch_offset + i * channel_size;
      // For current step = n, we fill (n+1) positions for the mask sequence.
      if (mask_tensor_type.ElementType() == litert::ElementType::Bool) {
        // Boolean mask: Fill value = true.
        bool* bool_ptr = static_cast<bool*>(mask_lock_and_addr.second);
        std::fill(bool_ptr + offset, bool_ptr + offset + current_step + 1,
                  true);
      } else if (mask_tensor_type.ElementType() ==
                 litert::ElementType::Float16) {
        // Float16 mask: Fill value = 0.0f.
        tflite::half* half_ptr =
            static_cast<tflite::half*>(mask_lock_and_addr.second);
        std::fill(half_ptr + offset, half_ptr + offset + current_step + 1,
                  tflite::half(0.0f));
      } else {  // litert::ElementType::Float32, checked above.
        // Float mask: Fill value = 0.0f.
        float* float_ptr = static_cast<float*>(mask_lock_and_addr.second);
        std::fill(float_ptr + offset, float_ptr + offset + current_step + 1,
                  0.0f);
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildLiteRtCompiledModelResources(const ModelAssets& model_assets) {
  ASSIGN_OR_RETURN(auto format, GetFileFormat(model_assets));
  switch (format) {
    case FileFormat::TASK:
      return BuildModelResourcesFromTaskFormat(model_assets);
    case FileFormat::LITERT_LM:
      return BuildModelResourcesFromLitertLmFormat(model_assets);
    default:
      return absl::InvalidArgumentError("Unsupported file format.");
  }
}

absl::Status GenericComputeTokenEmbeddings(
    const TensorBuffer& input_tokens, absl::Span<float> output_embeddings,
    absl::Span<float> output_ple_embeddings,
    EmbeddingLookupManager* embedding_lookup_manager,
    EmbeddingLookupManager* per_layer_embedding_lookup_manager) {
  LITERT_ASSIGN_OR_RETURN(auto input_tokens_span,
                          ReferTensorBufferAsSpan<int32_t>(input_tokens));
  const int num_tokens = input_tokens_span.size();

  if (embedding_lookup_manager == nullptr) {
    return absl::InvalidArgumentError("Embedding lookup manager is missing.");
  }

  const int embedding_dim =
      embedding_lookup_manager->GetTextEmbeddingLookup()->GetFloatsPerToken();
  auto tensor_type = MakeRankedTensorType<float>({num_tokens, embedding_dim});
  LITERT_ASSIGN_OR_RETURN(
      auto wrapped_embeddings,
      WrapOrCreateTensorBufferFromHostMemory(tensor_type, output_embeddings));

  RETURN_IF_ERROR(embedding_lookup_manager->LookupPrefill(
      input_tokens_span, &wrapped_embeddings.buffer, 0));

  if (per_layer_embedding_lookup_manager != nullptr &&
      !output_ple_embeddings.empty()) {
    const int ple_embedding_dim =
        per_layer_embedding_lookup_manager->GetTextEmbeddingLookup()
            ->GetFloatsPerToken();
    auto ple_tensor_type =
        MakeRankedTensorType<float>({num_tokens, ple_embedding_dim});
    LITERT_ASSIGN_OR_RETURN(auto wrapped_ple_embeddings,
                            WrapOrCreateTensorBufferFromHostMemory(
                                ple_tensor_type, output_ple_embeddings));
    RETURN_IF_ERROR(per_layer_embedding_lookup_manager->LookupPrefill(
        input_tokens_span, &wrapped_ple_embeddings.buffer, 0));
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
