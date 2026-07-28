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
#include <filesystem>  // NOLINT: Required for std::filesystem::path.
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/btree_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/embedding_lookup/embedding_lookup_text.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep NOLINT
#include "runtime/components/model_resources_litert_lm.h"  // IWYU pragma: keep
#include "runtime/components/model_resources_task.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/file_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert
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
// Possible local input attention mask names:
constexpr std::array<absl::string_view, 3> kInputLocalAttnMaskNames = {
    "local_mask", "attn_mask_local", "local_attn_mask"};
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
    ABSL_ASSIGN_OR_RETURN(auto memory_mapped_file,
                          model_assets.GetMemoryMappedFile());
    ABSL_ASSIGN_OR_RETURN(resources, ModelAssetBundleResources::Create(
                                         /*tag=*/"", memory_mapped_file));
  } else {
    ABSL_ASSIGN_OR_RETURN(auto scoped_file,
                          model_assets.GetOrCreateScopedFile());
    ABSL_ASSIGN_OR_RETURN(resources, ModelAssetBundleResources::Create(
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
BuildModelResourcesFromLitertLmFormat(const ModelAssets& model_assets,
                                      bool enable_file_backed_model_loading) {
  std::unique_ptr<LitertLmLoader> loader;
  if (model_assets.HasMemoryMappedFile()) {
    ABSL_ASSIGN_OR_RETURN(auto memory_mapped_file,
                          model_assets.GetMemoryMappedFile());
    ABSL_ASSIGN_OR_RETURN(loader, LitertLmLoader::Create(memory_mapped_file));
  } else {
    // `BuildModelResourcesFromLitertLmFormat` expects a ScopedFile that it
    // takes ownership of, so we need to duplicate the ScopedFile to keep
    // the original alive.
    ABSL_ASSIGN_OR_RETURN(auto scoped_file,
                          model_assets.GetOrCreateScopedFile());
    ABSL_ASSIGN_OR_RETURN(auto duplicate_file, scoped_file->Duplicate());
    ABSL_ASSIGN_OR_RETURN(loader,
                          LitertLmLoader::Create(std::move(duplicate_file)));
  }
  return ModelResourcesLitertLm::Create(
      std::move(loader), enable_file_backed_model_loading);
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
    if (absl::c_linear_search(kInputLocalAttnMaskNames, input_name)) {
      model_signatures.input_attn_mask_local = std::string(input_name);
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
    const SortedPrefillSignatureMap& prefill_runner_set, int input_length,
    std::optional<int> max_prefill_sequence_length) {
  SortedPrefillSignatureMap available_runner_set = prefill_runner_set;
  if (max_prefill_sequence_length.has_value()) {
    absl::erase_if(available_runner_set, [&](const auto& pair) {
      return pair.first > max_prefill_sequence_length.value();
    });
    if (available_runner_set.empty()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Chosen prefill work group size exceeds available state entries (",
          max_prefill_sequence_length.value(), ")."));
    }
  }

  // A simple greedy approach can cause performance cliffs for devices that
  // perform much better with longer sequence lengths See
  // go/smarter-prefill-chunking for more details.
  //
  // Instead, we use a "cautious greedy" approach. We greedily fill with the
  // largest sequence length possible. For the remainder, we evaluate whether to
  // pack it into smaller chunks or "cautiously" upgrade it. If the remainder is
  // at least half of the current sequence length, it is upgraded to use an
  // extra full-sized chunk to prevent excessive fragmentation.
  //
  // An exception is made for models that have sequence lengths that are within
  // a factor of 2 of each other (e.g. 128 and 256). In these cases, we will
  // default back to standard greedy stacking, provided the remainder
  // comfortably fits into the smaller sequence length. Otherwise the smaller
  // sequence length will not be used.
  //
  // Examples:
  // 1. input_length = 640, prefill_runner_set = {512, 128, 32}
  //    work_groups = {{"sig_512", 512}, {"sig_128", 128}}
  // 2. input_length = 768, prefill_runner_set = {512, 128, 32}
  //    work_groups = {{"sig_512", 512}, {"sig_512", 256}}
  // 3. input_length = 592, prefill_runner_set = {512, 128, 32}
  //    work_groups = {{"sig_512", 512}, {"sig_128", 80}}
  // 4. input_length = 130, prefill_runner_set = {512, 128}
  //    work_groups = {{"sig_128", 128}, {"sig_128", 2}}
  // 5. input_length = 599, prefill_runner_set = {600, 500}
  //    work_groups = {{"sig_600", 599}}

  std::vector<std::pair<std::string, int>> work_groups;
  int current_remaining_capacity =
      max_prefill_sequence_length.value_or(std::numeric_limits<int>::max());

  // Starting with the largest sequence length available and working our way
  // down, we will add work groups to cover the input length.
  for (auto it = available_runner_set.begin(); it != available_runner_set.end();
       ++it) {
    if (input_length <= 0) {
      break;
    }

    int cur_seq_len = it->first;
    if (cur_seq_len > current_remaining_capacity) {
      continue;
    }

    int full_chunks = input_length / cur_seq_len;
    int max_chunks_capacity = current_remaining_capacity / cur_seq_len;
    full_chunks = std::min(full_chunks, max_chunks_capacity);

    int remainder = input_length - (full_chunks * cur_seq_len);

    // 1. Greedily add any full chunks of the current sequence length.
    for (int i = 0; i < full_chunks; ++i) {
      work_groups.push_back(std::make_pair(it->second, cur_seq_len));
      current_remaining_capacity -= cur_seq_len;
    }
    input_length = remainder;

    if (input_length == 0) {
      break;
    }

    // 2. If there's no smaller sequence length available, we must cover the
    // remainder with this runner if capacity allows.
    auto next_it = std::next(it);
    if (next_it == available_runner_set.end()) {
      if (cur_seq_len <= current_remaining_capacity) {
        work_groups.push_back(std::make_pair(it->second, input_length));
        current_remaining_capacity -= cur_seq_len;
        input_length = 0;
      }
      break;
    }

    int next_seq_len = next_it->first;

    // 3. We need to decide whether to use the current sequence length
    // runner to cover the remainder, or let the smaller runners take care of
    // the remainder.
    if (next_seq_len * 2 >= cur_seq_len && input_length <= next_seq_len) {
      // Check fallback: if next_seq_len >= cur_seq_len / 2 AND remainder <=
      // next_seq_len
      //   Our sequence lengths are too close, AND the remainder fits
      //   comfortably in the next sequence length, so let the smaller runners
      //   handle it.
      continue;
    } else if (input_length * 2 >= cur_seq_len &&
               cur_seq_len <= current_remaining_capacity) {
      // Check cautious threshold rule: if remainder >= cur_seq_len / 2
      //   Cover with the current sequence length runner.
      work_groups.push_back(std::make_pair(it->second, input_length));
      current_remaining_capacity -= cur_seq_len;
      input_length = 0;
      break;
    }
    // Threshold not met or capacity exceeded: let smaller runners handle
    // remainder.
  }

  if (input_length > 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("Prefill input length exceeds available state entries "
                     "(remaining capacity: ",
                     max_prefill_sequence_length.value_or(0), ")."));
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
  LITERT_ASSIGN_OR_RETURN(auto param_tensor_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              param_tensor, TensorBuffer::LockMode::kWrite));
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

// Helper function to fill a range of values in a tensor buffer with the given
// type.
static void FillRange(litert::ElementType type, void* addr, int start,
                      int end) {
  if (type == litert::ElementType::Bool) {
    bool* ptr = static_cast<bool*>(addr);
    std::fill(ptr + start, ptr + end, true);
  } else if (type == litert::ElementType::Float16) {
    tflite::half* ptr = static_cast<tflite::half*>(addr);
    std::fill(ptr + start, ptr + end, tflite::half(0.0f));
  } else {
    float* ptr = static_cast<float*>(addr);
    std::fill(ptr + start, ptr + end, 0.0f);
  }
}

absl::Status FillAttentionMask(litert::TensorBuffer& mask, int start_timestep,
                               int steps,
                               const AttentionMaskPolicy& attention_mask_policy,
                               std::optional<absl::Span<const int>> token_ids,
                               std::optional<int> sliding_window_size) {
  LITERT_ASSIGN_OR_RETURN(auto mask_tensor_type, mask.TensorType());
  RET_CHECK_EQ(mask_tensor_type.Layout().Rank(), 4)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Attention mask must be 4D.";
  int batch_size = mask_tensor_type.Layout().Dimensions()[0];
  int context_size = mask_tensor_type.Layout().Dimensions()[3];
  if (attention_mask_policy == AttentionMaskPolicy::kVisionBidirectional &&
      !token_ids.has_value()) {
    return absl::InvalidArgumentError(
        "Empty token ids with vision bidirectional attention mask policy is "
        "not allowed.");
  }
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
      int q = start_timestep + i;
      int offset = b * batch_offset + i * context_size;
      if (attention_mask_policy == AttentionMaskPolicy::kVisionBidirectional &&
          q < token_ids->size() &&
          (*token_ids)[q] == ExecutorVisionData::kSpecialToken) {
        int start_k = 0;
        if (sliding_window_size.has_value()) {
          start_k = std::max(0, q - sliding_window_size.value() + 1);
        }
        int end_k = token_ids->size();
        if (sliding_window_size.has_value()) {
          end_k = std::min<int>(end_k, q + sliding_window_size.value() + 1);
        }
        int first_past_non_vision_token = q;
        for (int j = q - 1; j >= start_k; --j) {
          if ((*token_ids)[j] != ExecutorVisionData::kSpecialToken) {
            first_past_non_vision_token = j;
            break;
          }
        }
        // Image tokens can attend to all past tokens, but cannot attend to
        // future non-vision tokens.
        const int first_future_non_vision_token =
            std::find_if(token_ids->begin() + q, token_ids->begin() + end_k,
                         [](int token_id) {
                           return token_id != ExecutorVisionData::kSpecialToken;
                         }) -
            token_ids->begin();
        FillRange(mask_tensor_type.ElementType(), mask_lock_and_addr.second,
                  offset + start_k, offset + first_future_non_vision_token);
      } else {
        // Fast path for text queries or other policies.
        int causal_start = 0;
        if (sliding_window_size.has_value()) {
          causal_start = std::max(0, q - sliding_window_size.value() + 1);
        }
        int causal_end = q + 1;
        if (attention_mask_policy == AttentionMaskPolicy::kBidirectional) {
          causal_end = std::min<int>(context_size, start_timestep + steps);
          if (sliding_window_size.has_value()) {
            causal_end =
                std::min<int>(causal_end, q + sliding_window_size.value());
          }
        }
        FillRange(mask_tensor_type.ElementType(), mask_lock_and_addr.second,
                  offset + causal_start, offset + causal_end);
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildLiteRtCompiledModelResources(const ModelAssets& model_assets,
                                  bool enable_file_backed_model_loading) {
  ABSL_ASSIGN_OR_RETURN(auto format, GetFileFormat(model_assets));
  switch (format) {
    case FileFormat::TASK:
      return BuildModelResourcesFromTaskFormat(model_assets);
    case FileFormat::LITERT_LM:
      return BuildModelResourcesFromLitertLmFormat(
          model_assets, enable_file_backed_model_loading);
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
  auto output_buffer_type =
      embedding_lookup_manager->GetTextEmbeddingLookup()->GetOutputBufferType();
  std::vector<int32_t> dims = {num_tokens, embedding_dim};
  if (output_buffer_type.has_value()) {
    auto span_dims = output_buffer_type->Layout().Dimensions();
    dims.assign(span_dims.begin(), span_dims.end());
    dims[0] = 1;
    dims[1] = num_tokens;
  }
  auto tensor_type = MakeRankedTensorType<float>(dims);
  LITERT_ASSIGN_OR_RETURN(
      auto wrapped_embeddings,
      WrapOrCreateTensorBufferFromHostMemory(tensor_type, output_embeddings));

  ABSL_RETURN_IF_ERROR(embedding_lookup_manager->LookupPrefill(
      input_tokens_span, &wrapped_embeddings.buffer, 0 /*token_offset=*/));
  if (!wrapped_embeddings.wrapped) {
    LITERT_RETURN_IF_ERROR(wrapped_embeddings.buffer.Read(output_embeddings));
  }

  if (per_layer_embedding_lookup_manager != nullptr &&
      !output_ple_embeddings.empty()) {
    auto ple_output_buffer_type =
        per_layer_embedding_lookup_manager->GetTextEmbeddingLookup()
            ->GetOutputBufferType();
    const int ple_embedding_dim =
        per_layer_embedding_lookup_manager->GetTextEmbeddingLookup()
            ->GetFloatsPerToken();
    std::vector<int32_t> ple_dims = {num_tokens, ple_embedding_dim};
    if (ple_output_buffer_type.has_value()) {
      auto ple_span_dims = ple_output_buffer_type->Layout().Dimensions();
      ple_dims.assign(ple_span_dims.begin(), ple_span_dims.end());
      ple_dims[0] = 1;
      ple_dims[1] = num_tokens;
    }
    auto ple_tensor_type = MakeRankedTensorType<float>(ple_dims);
    LITERT_ASSIGN_OR_RETURN(auto wrapped_ple_embeddings,
                            WrapOrCreateTensorBufferFromHostMemory(
                                ple_tensor_type, output_ple_embeddings));
    ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_manager->LookupPrefill(
        input_tokens_span, &wrapped_ple_embeddings.buffer,
        0 /*token_offset=*/));
    if (!wrapped_ple_embeddings.wrapped) {
      LITERT_RETURN_IF_ERROR(
          wrapped_ple_embeddings.buffer.Read(output_ple_embeddings));
    }
  }
  return absl::OkStatus();
}

absl::Status SetCpuOptions(litert::CpuOptions& cpu_options, int num_threads) {
  cpu_options.SetNumThreads(num_threads);
  auto default_xnn_options = TfLiteXNNPackDelegateOptionsDefault();
  cpu_options.SetXNNPackFlags(
      default_xnn_options.flags |
      TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED |
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);
  return absl::OkStatus();
}

absl::Status SetCommonGpuOptions(
    const ExecutorSettingsBase& executor_settings,
    litert::GpuOptions& gpu_options,
    std::optional<ActivationDataType> fallback_activation_data_type) {
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.EnableConstantTensorSharing(true);
  ActivationDataType activation_type =
      executor_settings.GetActivationDataType().has_value()
          ? executor_settings.GetActivationDataType().value()
          : fallback_activation_data_type.value_or(ActivationDataType::FLOAT32);
  // Mixed precision setting overrides the activation data type setting. The
  // underlying delegate uses fp32 precision to represent mixed precision, so we
  // set it to fp32 here.
  if (executor_settings.IsMixedPrecisionEnabled() ||
      activation_type == ActivationDataType::FLOAT32) {
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
  } else {
    // For FLOAT16, INT16, and INT8 activation data types, calculation
    // precision of the GPU delegate is set to fp16.
    gpu_options.SetPrecision(GpuOptions::Precision::kFp16);
  }
#if defined(__APPLE__)
  gpu_options.SetPreferTextureWeights(false);
  gpu_options.SetUseMetalArgumentBuffers(true);
#else   // !__APPLE__
  gpu_options.SetPreferTextureWeights(true);
#endif  // !__APPLE__
  gpu_options.SetMadviseOriginalSharedTensors(true);
  gpu_options.SetConvertWeightsOnGpu(true);
  return absl::OkStatus();
}

absl::Status SetCpuCacheOptions(
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        weight_cache_file,
    absl::string_view logging_prefix, litert::CpuOptions& cpu_options) {
  if (!weight_cache_file.ok()) {
    ABSL_VLOG(1) << logging_prefix << " does not use cache.";
    return absl::OkStatus();
  }

  if (std::holds_alternative<std::shared_ptr<litert::lm::ScopedFile>>(
          *weight_cache_file)) {
    auto scoped_cache_file =
        std::get<std::shared_ptr<litert::lm::ScopedFile>>(*weight_cache_file);
    if (scoped_cache_file != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
      ABSL_ASSIGN_OR_RETURN(int fd, duplicated.Release());
      cpu_options.SetXNNPackWeightCacheFileDescriptor(fd);
      ABSL_VLOG(1) << logging_prefix
                   << " use provided cache file descriptor: " << fd;
    }
  } else if (std::holds_alternative<std::string>(*weight_cache_file)) {
    const std::string& weight_cache_path =
        std::get<std::string>(*weight_cache_file);
    cpu_options.SetXNNPackWeightCachePath(weight_cache_path.c_str());
    ABSL_VLOG(1) << logging_prefix << " use cache path: " << weight_cache_path;
  }
  return absl::OkStatus();
}

absl::Status SetGpuCacheOptions(
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        weight_cache_file,
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        program_cache_file,
    absl::string_view cache_key, absl::string_view logging_prefix,
    bool cache_compiled_shaders_only, litert::GpuOptions& gpu_options) {
  if (!cache_key.empty()) {
    gpu_options.SetModelCacheKey(cache_key.data());
  }
  bool serialization_dir_set = false;
  std::string cache_path;
  if (weight_cache_file.ok()) {
    if (std::holds_alternative<std::string>(*weight_cache_file)) {
      cache_path =
          std::filesystem::path(std::get<std::string>(*weight_cache_file))
              .parent_path()
              .string();
      ABSL_VLOG(1) << (logging_prefix.empty()
                           ? ""
                           : absl::StrCat(logging_prefix, ": "))
                   << "Setting serialization dir: " << cache_path;
      gpu_options.SetSerializationDir(cache_path.c_str());
      serialization_dir_set = true;
    } else {
      auto scoped_cache_file =
          std::get<std::shared_ptr<lm::ScopedFile>>(*weight_cache_file);
      ABSL_ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
      ABSL_ASSIGN_OR_RETURN(int fd, duplicated.Release());
      gpu_options.SetWeightCacheFd(fd);
    }
    gpu_options.SetSerializeExternalTensors(true);
  } else {
    gpu_options.SetSerializeExternalTensors(false);
  }

  if (program_cache_file.ok()) {
    if (std::holds_alternative<std::string>(*program_cache_file)) {
      if (!serialization_dir_set) {
        cache_path =
            std::filesystem::path(std::get<std::string>(*program_cache_file))
                .parent_path()
                .string();
        ABSL_VLOG(1) << (logging_prefix.empty()
                             ? ""
                             : absl::StrCat(logging_prefix, ": "))
                     << "Setting program cache dir: " << cache_path;
        gpu_options.SetSerializationDir(cache_path.c_str());
      }
    } else {
      auto scoped_cache_file =
          std::get<std::shared_ptr<lm::ScopedFile>>(*program_cache_file);
      ABSL_ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
      ABSL_ASSIGN_OR_RETURN(int fd, duplicated.Release());
      gpu_options.SetProgramCacheFd(fd);
    }
    gpu_options.CacheCompiledProgramsOnly(cache_compiled_shaders_only);
    gpu_options.SetSerializeProgramCache(true);
  } else {
    gpu_options.SetSerializeProgramCache(false);
  }
  return absl::OkStatus();
}

absl::StatusOr<GpuModelCacheData> GetGpuModelCacheData(
    const ExecutorSettingsBase& executor_settings,
    absl::string_view cache_name) {
  GpuModelCacheData cache_data;
  // Skip if cache is explicitly disabled.
  if (executor_settings.GetCacheDir() != ":nocache") {
    auto model_path = executor_settings.GetModelAssets().GetPath().value_or("");
    std::string model_basename = std::string(Basename(model_path));
    cache_data.program_cache_file = executor_settings.GetProgramCacheFile(
        absl::StrCat(cache_name, ExecutorSettingsBase::kMlDriftCacheSuffix),
        /*check_and_clean=*/true);
    cache_data.weight_cache_file = executor_settings.GetWeightCacheFile(
        absl::StrCat(cache_name,
                     ExecutorSettingsBase::kMlDriftWeightCacheSuffix),
        /*check_and_clean=*/true);
    if (!model_path.empty()) {
      ABSL_ASSIGN_OR_RETURN(std::string metadata_id,
                            GetFileCacheIdentifier(model_path));
      if (cache_data.program_cache_file.ok() ||
          cache_data.weight_cache_file.ok()) {
        cache_data.cache_key =
            absl::StrCat(model_basename, cache_name, "_", metadata_id);
      }
    } else {
      // If the model path is empty, we should still set a cache key. This
      // cache key should include the file timestamp and size in order
      // to be able to detect changes in the model files.
      LITERT_ASSIGN_OR_RETURN(
          auto scoped_file, executor_settings.GetModelAssets().GetScopedFile());
      bool has_valid_program_cache_fd =
          cache_data.program_cache_file.ok() &&
          std::holds_alternative<std::shared_ptr<litert::lm::ScopedFile>>(
              *cache_data.program_cache_file);
      bool has_valid_weight_cache_fd =
          cache_data.weight_cache_file.ok() &&
          std::holds_alternative<std::shared_ptr<litert::lm::ScopedFile>>(
              *cache_data.weight_cache_file);
      if (scoped_file != nullptr &&
          (has_valid_program_cache_fd || has_valid_weight_cache_fd)) {
        LITERT_ASSIGN_OR_RETURN(std::string metadata_id,
                                GetFileCacheIdentifier(*scoped_file));
        cache_data.cache_key = absl::StrCat("fd_", metadata_id);
      }
    }
  }
  return cache_data;
}

absl::Status InitializeEmbeddingLookups(
    ::litert::Environment& env, ModelResources& resources,
    std::unique_ptr<EmbeddingLookupManager>& embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager>& per_layer_embedding_lookup) {
  absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
  {
    auto end_of_audio_model =
        resources.GetTFLiteModel(ModelType::kTfLiteEndOfAudio);
    if (end_of_audio_model.ok()) {
      end_of_multi_modal_embedding_models.insert(
          {ExecutorAudioData::kEndToken, end_of_audio_model.value()});
    }
  }
  {
    auto end_of_vision_model =
        resources.GetTFLiteModel(ModelType::kTfLiteEndOfVision);
    if (end_of_vision_model.ok()) {
      end_of_multi_modal_embedding_models.insert(
          {ExecutorVisionData::kEndToken, end_of_vision_model.value()});
    }
  }

  auto text_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLiteEmbedder);
  if (text_embedder_model.ok()) {
    ABSL_ASSIGN_OR_RETURN(
        embedding_lookup,
        EmbeddingLookupManager::Create(env, *text_embedder_model,
                                       end_of_multi_modal_embedding_models));
  }

  // Create per layer embedding lookups from the resources.
  auto per_layer_embedder_model =
      resources.GetTFLiteModel(ModelType::kTfLitePerLayerEmbedder);
  if (per_layer_embedder_model.ok()) {
    std::optional<ScopedFile> per_layer_external_weight_file;
    Options::ScopedWeightSectionMap per_layer_external_weight_sections;
    auto section_offset =
        resources.GetWeightsSectionOffset(ModelType::kTfLitePerLayerEmbedder);
    if (section_offset.ok()) {
      per_layer_external_weight_sections["tflite_weights"] = {
          section_offset.value().first,
          section_offset.value().second - section_offset.value().first};
      LITERT_ASSIGN_OR_RETURN(auto scoped_file, resources.GetScopedFile());
      LITERT_ASSIGN_OR_RETURN(auto duplicated_scoped_file,
                              scoped_file.get().Duplicate());
      per_layer_external_weight_file = std::move(duplicated_scoped_file);
    }
    ABSL_ASSIGN_OR_RETURN(per_layer_embedding_lookup,
                          EmbeddingLookupManager::Create(
                              env, *per_layer_embedder_model,
                              /*fully_supports_multi_modal=*/false,
                              /*signature_key=*/std::nullopt,
                              std::move(per_layer_external_weight_file),
                              std::move(per_layer_external_weight_sections)));
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
