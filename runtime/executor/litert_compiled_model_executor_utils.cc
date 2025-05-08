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
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/external_file.pb.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

using ::litert::Expected;
using ::litert::Model;
using ::litert::lm::ModelAssetBundleResources;
using ::litert::lm::proto::ExternalFile;

// The name of the prefill decode model in the task bundle.
constexpr char kPrefilDecodeModelNameInTaskBundle[] = "TF_LITE_PREFILL_DECODE";

// Gemma2 JAX model signatures.
// Input: [batch_size, max_seq_len]
constexpr char kGemma2JAX_InputTokens[] = "token_ids";
// Input: [batch_size, max_seq_len]
constexpr char kGemma2JAX_InputPositions[] = "positions";
// Input: [batch_size, max_seq_len, 1, context_size]
constexpr char kGemma2JAX_InputAttnMask[] = "attn_mask";
constexpr AttentionMaskDataType kGemma2JAX_InputAttnMaskDataType =
    AttentionMaskDataType::BOOLEAN;
// Output: [batch_size, max_seq_len, vocab_size]
constexpr char kGemma2JAX_OutputLogits[] = "logits";

// PyTorch model signatures running on CPU and GPU including Gemma2 & 3 and
// other open source models, which has "mask" as input.
// Input: [batch_size, max_seq_len]
constexpr char kPyTorch_InputTokens[] = "tokens";
// Input: [max_seq_len]
constexpr char kPyTorch_InputPositions[] = "input_pos";
// Input: [batch_size, 1, max_seq_len, context_size]
constexpr char kPyTorch_InputAttnMask[] = "mask";
constexpr AttentionMaskDataType kPyTorch_InputAttnMaskDataType =
    AttentionMaskDataType::FLOAT;
// Output: [batch_size, max_seq_len, vocab_size]
constexpr char kPyTorch_OutputLogits[] = "logits";

// PyTorch model signatures running only on CPU including Gemma2 & 3 and other
// open source models, which does not have "mask" as input.
// Input: [batch_size, max_seq_len]
constexpr char kPyTorchCpuOnly_InputTokens[] = "tokens";
// Input: [max_seq_len]
constexpr char kPyTorchCpuOnly_InputPositions[] = "input_pos";
constexpr AttentionMaskDataType kPyTorchCpuOnly_InputAttnMaskDataType =
    AttentionMaskDataType::FLOAT;
// Output: [batch_size, max_seq_len, vocab_size]
constexpr char kPyTorchCpuOnly_OutputLogits[] = "logits";

// Gemini V1.5 model signatures.
// Input: [batch_size, max_seq_len]
constexpr char kGemini_InputTokens[] = "token_ids";
// Input: [batch_size, max_seq_len]
constexpr char kGemini_InputPositions[] = "positions";
// Input: [batch_size, max_seq_len, 1, context_size]
constexpr char kGemini_InputAttnMask[] = "attn_mask";
constexpr AttentionMaskDataType kGemini_InputAttnMaskDataType =
    AttentionMaskDataType::FLOAT;
// Output: [batch_size, max_seq_len, vocab_size]
constexpr char kGemini_OutputLogits[] = "logits";

bool Contains(const std::vector<absl::string_view>& input_names,
              const char* name) {
  return std::find(input_names.begin(), input_names.end(), name) !=
         input_names.end();
}

bool IsGemma2JAX(const std::vector<absl::string_view>& input_names,
                 const std::vector<absl::string_view>& output_names) {
  return Contains(input_names, kGemma2JAX_InputTokens) &&
         Contains(input_names, kGemma2JAX_InputPositions) &&
         Contains(input_names, kGemma2JAX_InputAttnMask) &&
         Contains(output_names, kGemma2JAX_OutputLogits);
}

bool IsPyTorch(const std::vector<absl::string_view>& input_names,
               const std::vector<absl::string_view>& output_names) {
  return Contains(input_names, kPyTorch_InputTokens) &&
         Contains(input_names, kPyTorch_InputPositions) &&
         Contains(input_names, kPyTorch_InputAttnMask) &&
         Contains(output_names, kPyTorch_OutputLogits);
}

bool IsPyTorchCpuOnly(const std::vector<absl::string_view>& input_names,
                      const std::vector<absl::string_view>& output_names) {
  return Contains(input_names, kPyTorchCpuOnly_InputTokens) &&
         Contains(input_names, kPyTorchCpuOnly_InputPositions) &&
         Contains(output_names, kPyTorchCpuOnly_OutputLogits);
}

bool IsGemini(const std::vector<absl::string_view>& input_names,
              const std::vector<absl::string_view>& output_names) {
  return Contains(input_names, kGemini_InputTokens) &&
         Contains(input_names, kGemini_InputPositions) &&
         Contains(input_names, kGemini_InputAttnMask) &&
         Contains(output_names, kGemini_OutputLogits);
}

}  // namespace

absl::StatusOr<ModelSignatures> GetModelSignaturesFromInputOutputNames(
    const std::vector<absl::string_view>& input_names,
    const std::vector<absl::string_view>& output_names) {
  if (IsGemma2JAX(input_names, output_names)) {
    return ModelSignatures{
        .input_tokens = kGemma2JAX_InputTokens,
        .input_positions = kGemma2JAX_InputPositions,
        .input_attn_mask = kGemma2JAX_InputAttnMask,
        .input_attn_mask_data_type = kGemma2JAX_InputAttnMaskDataType,
        .output_logits = kGemma2JAX_OutputLogits,
    };
  }

  if (IsPyTorch(input_names, output_names)) {
    return ModelSignatures{
        .input_tokens = kPyTorch_InputTokens,
        .input_positions = kPyTorch_InputPositions,
        .input_attn_mask = kPyTorch_InputAttnMask,
        .input_attn_mask_data_type = kPyTorch_InputAttnMaskDataType,
        .output_logits = kPyTorch_OutputLogits,
    };
  }

  if (IsPyTorchCpuOnly(input_names, output_names)) {
    return ModelSignatures{
        .input_tokens = kPyTorch_InputTokens,
        .input_positions = kPyTorch_InputPositions,
        .input_attn_mask_data_type = kPyTorchCpuOnly_InputAttnMaskDataType,
        .output_logits = kPyTorchCpuOnly_OutputLogits,
    };
  }

  if (IsGemini(input_names, output_names)) {
    return ModelSignatures{
        .input_tokens = kGemini_InputTokens,
        .input_positions = kGemini_InputPositions,
        .input_attn_mask = kGemini_InputAttnMask,
        .input_attn_mask_data_type = kGemini_InputAttnMaskDataType,
        .output_logits = kGemini_OutputLogits,
    };
  }

  return absl::FailedPreconditionError("Unsupported model signature.");
}

absl::StatusOr<SortedPrefillSignatureMap> GetPrefillRunnerSetFromModel(
    ::litert::Model& model, const std::string& signature_name_base,
    const std::string& input_tokens_name) {
  SortedPrefillSignatureMap prefill_runner_set;
  auto signatures = model.GetSignatures();
  for (auto& signature : *signatures) {
    if (auto signature_key = signature.Key();
        absl::StartsWith(signature_key, signature_name_base)) {
      auto subgraph = model.Subgraph(signature_key);
      if (!subgraph) {
        return absl::InternalError(subgraph.Error().Message());
      }
      auto input_tokens_tensor = subgraph->Input(input_tokens_name);
      if (!input_tokens_tensor) {
        return absl::InternalError(input_tokens_tensor.Error().Message());
      }
      auto ranked_tensor_type = input_tokens_tensor->RankedTensorType();
      if (!ranked_tensor_type) {
        return absl::InternalError(ranked_tensor_type.Error().Message());
      }

      if (ranked_tensor_type->Layout().Rank() == 2) {
        // [batch_size, max_seq_len]
        prefill_runner_set[ranked_tensor_type->Layout().Dimensions()[1]] =
            std::string(signature_key);
      } else if (ranked_tensor_type->Layout().Rank() == 1) {
        // [max_seq_len]
        prefill_runner_set[ranked_tensor_type->Layout().Dimensions()[0]] =
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
    const SortedPrefillSignatureMap& prefill_runner_set,
    int input_length) {
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

absl::Status InitializeAttentionMask(litert::TensorBuffer& mask,
                                     AttentionMaskDataType mask_data_type,
                                     bool is_f16) {
  auto mask_size = mask.PackedSize();
  RET_CHECK(mask_size) << "Failed to get attention mask buffer size.";
  auto mask_lock_and_addr = litert::TensorBufferScopedLock::Create(mask);
  RET_CHECK(mask_lock_and_addr) << "Failed to lock attention mask buffer.";

  switch (mask_data_type) {
    case AttentionMaskDataType::BOOLEAN: {
      // Boolean mask: Default value = false.
      memset(mask_lock_and_addr->second, 0, *mask_size);
    } break;
    case AttentionMaskDataType::FLOAT: {
      // Float mask: Default value is based on precision.
      // Default value reference:
      // third_party/odml/infra/genai/inference/ml_drift/llm/tasks/apply_attention_mask_test_util.cc
      float* mask_ptr = static_cast<float*>(mask_lock_and_addr->second);
      std::fill(mask_ptr, mask_ptr + *mask_size / sizeof(float),
                is_f16 ? -45824 : -0.7f * std::numeric_limits<float>::max());
    } break;
    default:
      return absl::InvalidArgumentError(
          "Unsupported attention mask data type.");
  }
  return absl::OkStatus();
}

absl::Status FillAttentionMask(litert::TensorBuffer& mask, int start_timestep,
                               int steps,
                               AttentionMaskDataType mask_data_type) {
  auto mask_tensor_type = mask.TensorType();
  RET_CHECK(mask_tensor_type) << "Failed to get attention mask tensor type.";
  RET_CHECK_EQ(mask_tensor_type->Layout().Rank(), 4)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Attention mask must be 4D.";
  int channel_size = mask_tensor_type->Layout().Dimensions()[3];
  auto mask_lock_and_addr = litert::TensorBufferScopedLock::Create(mask);
  RET_CHECK(mask_lock_and_addr) << "Failed to lock attention mask buffer.";

  for (int i = 0; i < steps; ++i) {
    int current_step = start_timestep + i;
    int offset = i * channel_size;
    // For current step = n, we fill (n+1) positions for the mask sequence.
    switch (mask_data_type) {
      case AttentionMaskDataType::BOOLEAN: {
        // Boolean mask: Fill value = true.
        bool* mask_bool_ptr = static_cast<bool*>(mask_lock_and_addr->second);
        std::fill(mask_bool_ptr + offset,
                  mask_bool_ptr + offset + current_step + 1, true);
      } break;
      case AttentionMaskDataType::FLOAT: {
        // Float mask: Fill value = 0.0f.
        float* mask_float_ptr = static_cast<float*>(mask_lock_and_addr->second);
        std::fill(mask_float_ptr + offset,
                  mask_float_ptr + offset + current_step + 1, 0.0f);
      } break;
      default:
        return absl::InvalidArgumentError(
            "Unsupported attention mask data type.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ExecutorModelResources>>
BuildLiteRtCompiledModelResources(const std::string& model_path) {
  auto executor_model_resources = std::make_unique<ExecutorModelResources>();
  Expected<Model> litert_model;
  std::unique_ptr<ModelAssetBundleResources> resources;
  if (absl::EndsWith(model_path, ".task")) {
    // .task format
    auto external_file = std::make_unique<ExternalFile>();
    external_file->set_file_name(model_path);
    ASSIGN_OR_RETURN(resources, ModelAssetBundleResources::Create(
                                    "", std::move(external_file)));
    const std::vector<std::string>& files_list = resources->ListFiles();
    const absl::flat_hash_set<std::string> files_set(files_list.begin(),
                                                     files_list.end());
    RET_CHECK(files_set.contains(kPrefilDecodeModelNameInTaskBundle))
        << kPrefilDecodeModelNameInTaskBundle
        << " model file not found in task bundle.";
    ASSIGN_OR_RETURN(absl::string_view buffer,
                     resources->GetFile(kPrefilDecodeModelNameInTaskBundle));
    litert::BufferRef<uint8_t> buffer_ref(buffer.data(), buffer.size());
    litert_model = Model::CreateFromBuffer(buffer_ref);
    RET_CHECK(litert_model) << "Failed to build "
                            << kPrefilDecodeModelNameInTaskBundle << " model.";
    executor_model_resources->model_asset_bundle_resources =
        std::move(resources);
    executor_model_resources->litert_model = std::move(*litert_model);
  } else {
    // .tflite format
    litert_model = Model::CreateFromFile(model_path.c_str());
    RET_CHECK(litert_model) << "Failed to build "
                            << kPrefilDecodeModelNameInTaskBundle << " model.";
    executor_model_resources->litert_model = std::move(*litert_model);
  }
  return executor_model_resources;
}

// 40% of the time in JoinPath() is from calls with 2 arguments, so we
// specialize that case.
absl::StatusOr<std::string> JoinPath(absl::string_view path1,
                                     absl::string_view path2) {
  if (path1.empty()) return absl::InvalidArgumentError("Empty path1.");
  if (path2.empty()) return absl::InvalidArgumentError("Empty path2.");
  if (path1.back() == '/') {
    if (path2.front() == '/')
      return absl::StrCat(path1, absl::ClippedSubstr(path2, 1));
  } else {
    if (path2.front() != '/') return absl::StrCat(path1, "/", path2);
  }
  return absl::StrCat(path1, path2);
}

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
  absl::string_view::size_type pos = path.find_last_of('/');

  // Handle the case with no '/' in 'path'.
  if (pos == absl::string_view::npos)
    return std::make_pair(path.substr(0, 0), path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));

  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

}  // namespace litert::lm
