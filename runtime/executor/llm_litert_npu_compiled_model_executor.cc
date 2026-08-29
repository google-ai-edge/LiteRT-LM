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

#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_model_types.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/litert/legacy_map_state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_processed_context.h"
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/executor/npu/llm_litert_npu_embedder.h"
#include "runtime/executor/npu/llm_litert_npu_kv_cache.h"
#include "runtime/executor/npu/llm_litert_npu_mask.h"
#include "runtime/executor/npu/llm_litert_npu_rope.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

namespace {
using ::litert::CompiledModel;
using ::litert::Environment;
using ::litert::Expected;
using ::litert::Model;
using ::litert::Options;
using ::litert::TensorBuffer;

class CompiledModelWrapper : public ::litert::CompiledModel {
 public:
  static ::litert::Expected<::litert::CompiledModel> Create(
      ::litert::Environment& env, const LiteRtModel litert_model,
      ::litert::Options& compilation_options) {
    return ::litert::CompiledModel::Create(env, litert_model,
                                           compilation_options);
  }
  static ::litert::Expected<::litert::CompiledModel> Create(
      ::litert::Environment& env, const LiteRtModel litert_model,
      litert::HwAccelerators accelerators) {
    return ::litert::CompiledModel::Create(env, litert_model, accelerators);
  }
};

using LogitsQuantizationParams =
    LlmLiteRtNpuCompiledModelExecutor::LogitsQuantizationParams;

constexpr absl::string_view kv_cache_k_root_name = kKvCacheKRootName;
constexpr absl::string_view kv_cache_v_root_name = kKvCacheVRootName;
constexpr absl::string_view kv_cache_c_root_name = kKvCacheCRootName;
constexpr absl::string_view kv_cache_slice_k_root_name = kKvCacheSliceKRootName;
constexpr absl::string_view kv_cache_slice_v_root_name = kKvCacheSliceVRootName;
constexpr absl::string_view kv_cache_slice_c_root_name = kKvCacheSliceCRootName;

constexpr char cache_k31[] = "kv_cache_k_31";
constexpr char cache_k25[] = "kv_cache_k_25";
constexpr char cache_v25[] = "kv_cache_v_25";
constexpr char cache_k19[] = "kv_cache_k_19";
constexpr char cache_v19[] = "kv_cache_v_19";
constexpr char cache_k23[] = "kv_cache_k_23";
constexpr char cache_v23[] = "kv_cache_v_23";
constexpr char cache_k17[] = "kv_cache_k_17";
constexpr char cache_v17[] = "kv_cache_v_17";
}  // namespace

LlmLiteRtNpuCompiledModelExecutor::~LlmLiteRtNpuCompiledModelExecutor() {
  ABSL_VLOG(1) << "LatencyStats: " << GetLatencyStats();
}

// Allocates input and output buffers for the text decoder compiled model:
// - text_decoder_{prefill,decode,verify}_input_buffers: Stores all non-KV-cache
//   input buffers (embeddings, attention masks, RoPE, sequence positions) that
//   will be shared with auxiliary model outputs (Embedder, Mask, RoPE).
// - input_kv_cache_buffers: Stores persistent full KV cache input tensor
//   buffers across all layers, shared across all execution stages.
// - {prefill,decode,verify}_output_kv_cache_slice_buffers: Stores the newly
//   computed KV cache slices output by the text decoder, to be written into the
//   persistent input_kv_cache_buffers during cache update.
absl::Status LlmLiteRtNpuCompiledModelExecutor::AllocateTextDecoderBuffers(
    litert::Environment& env, const litert::Model* text_decoder_model,
    CompiledModel& text_decoder_compiled_model,
    const ResolvedPrefillSignatures& prefill_signatures,
    absl::string_view decode_signature, absl::string_view verify_signature,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        verify_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, HWQuantParams>& kv_quant_params,
    int64_t kv_cache_init_value) {
  auto prefill_sig =
      text_decoder_model->FindSignature(prefill_signatures.prefill);

  if (prefill_sig.HasValue()) {
    for (auto output_name : prefill_sig->OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_v_root_name)) {
        auto tensor_expected = prefill_sig->OutputTensor(output_name);
        if (tensor_expected.HasValue()) {
          HWQuantParams q_params;
          if (tensor_expected->HasQuantization()) {
            auto pq = tensor_expected->PerTensorQuantization();
            q_params.scale = pq.scale;
            q_params.zero_point = pq.zero_point;
          }
          kv_quant_params[output_name] = q_params;
        }
      }
    }
  }

  // Create input buffers for prefill signature.
  for (auto input_name : prefill_sig->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name) ||
        absl::StartsWith(input_name, kv_cache_c_root_name)) {
      if (!input_kv_cache_buffers.contains(input_name)) {
        LITERT_ASSIGN_OR_RETURN(input_kv_cache_buffers[input_name],
                                text_decoder_compiled_model.CreateInputBuffer(
                                    prefill_signatures.prefill, input_name));
        LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(
            input_kv_cache_buffers[input_name], kv_cache_init_value));
      }
    } else {
      LITERT_ASSIGN_OR_RETURN(text_decoder_prefill_input_buffers[input_name],
                              text_decoder_compiled_model.CreateInputBuffer(
                                  prefill_signatures.prefill, input_name));
      text_decoder_prefill_input_buffers[input_name].Clear();
    }
  }
  // Create input buffers for decode signature. Skip kv cache input buffers as
  // they are already created in the prefill signature.
  auto decode_sig = text_decoder_model->FindSignature(decode_signature);
  for (auto input_name : decode_sig->InputNames()) {
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name) ||
        absl::StartsWith(input_name, kv_cache_c_root_name)) {
      // Create the input kv cache buffer for the decode signature if it is not
      // created in the prefill signature.
      if (!input_kv_cache_buffers.contains(input_name)) {
        LITERT_ASSIGN_OR_RETURN(input_kv_cache_buffers[input_name],
                                text_decoder_compiled_model.CreateInputBuffer(
                                    decode_signature, input_name));
        LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(
            input_kv_cache_buffers[input_name], kv_cache_init_value));
      }
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(text_decoder_decode_input_buffers[input_name],
                            text_decoder_compiled_model.CreateInputBuffer(
                                decode_signature, input_name));
    text_decoder_decode_input_buffers[input_name].Clear();
  }

  // Create output buffers for prefill signature.
  for (auto output_name : prefill_sig->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_c_root_name)) {
      LITERT_ASSIGN_OR_RETURN(
          prefill_output_kv_cache_slice_buffers[output_name],
          text_decoder_compiled_model.CreateOutputBuffer(
              prefill_signatures.prefill, output_name));
    }
  }
  // Create output buffers for decode signature.
  for (auto output_name : decode_sig->OutputNames()) {
    if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_v_root_name) ||
        absl::StartsWith(output_name, kv_cache_slice_c_root_name)) {
      LITERT_ASSIGN_OR_RETURN(decode_output_kv_cache_slice_buffers[output_name],
                              text_decoder_compiled_model.CreateOutputBuffer(
                                  decode_signature, output_name));
    }
  }

  // Create input/output buffers for verify signature if it exists.
  auto verify_sig = text_decoder_model->FindSignature(verify_signature);
  if (verify_sig) {
    for (auto input_name : verify_sig->InputNames()) {
      LITERT_ASSIGN_OR_RETURN(text_decoder_verify_input_buffers[input_name],
                              text_decoder_compiled_model.CreateInputBuffer(
                                  verify_signature, input_name));
      text_decoder_verify_input_buffers[input_name].Clear();
    }
    for (auto output_name : verify_sig->OutputNames()) {
      if (absl::StartsWith(output_name, kv_cache_slice_k_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_v_root_name) ||
          absl::StartsWith(output_name, kv_cache_slice_c_root_name)) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_kv_cache_slice_buffers[output_name],
            text_decoder_compiled_model.CreateOutputBuffer(verify_signature,
                                                           output_name));
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<InferenceContext>
LlmLiteRtNpuCompiledModelExecutor::CreateTextDecoderInferenceContext(
    ::litert::Environment& env,
    ::litert::CompiledModel& text_decoder_compiled_model,
    const ResolvedPrefillSignatures& prefill_signatures,
    absl::string_view decode_signature, absl::string_view verify_signature,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        prefill_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        decode_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        verify_output_kv_cache_slice_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers) {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  {
    for (const auto& [key, value] : text_decoder_prefill_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to prefill inputs.
    LITERT_ASSIGN_OR_RETURN(auto prefill_input_names,
                            text_decoder_compiled_model.GetSignatureInputNames(
                                prefill_signatures.prefill));
    for (const auto& [key, value] : input_kv_cache_buffers) {
      // Check if the kv cache buffer is used in the prefill signature.
      if (absl::c_find(prefill_input_names, std::string(key)) ==
          prefill_input_names.end()) {
        continue;
      }

      // The last layer kv cache in the prefill signature has float32 elements,
      // although it's not used in the model, CompiledModel will complain about
      // the mismatched buffer size. So we need to correct the buffer size here,
      // by creating a new buffer with the correct size.
      LITERT_ASSIGN_OR_RETURN(auto input_tensor_type,
                              text_decoder_compiled_model.GetInputTensorType(
                                  prefill_signatures.prefill, key));
      LITERT_ASSIGN_OR_RETURN(auto input_tensor_size,
                              input_tensor_type.Bytes());
      LITERT_ASSIGN_OR_RETURN(auto input_buffer_size, value.Size());
      if (input_tensor_size != input_buffer_size) {
        LITERT_ASSIGN_OR_RETURN(auto corrected_input_buffer,
                                text_decoder_compiled_model.CreateInputBuffer(
                                    prefill_signatures.prefill, key));
        corrected_input_buffer.Clear();
        LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key],
                                corrected_input_buffer.Duplicate());
      } else {
        LITERT_ASSIGN_OR_RETURN(prefill_input_buffers[key], value.Duplicate());
      }
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  {
    // Duplicate all output kv cache slice buffers to prefill output
    // buffers.
    for (const auto& [key, value] : prefill_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(prefill_output_buffers[key], value.Duplicate());
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  {
    for (const auto& [key, value] : text_decoder_decode_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to decode inputs.
    for (const auto& [key, value] : input_kv_cache_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_input_buffers[key], value.Duplicate());
    }
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  {
    // Duplicate all output kv cache slice buffers to decode output
    // buffers.
    for (const auto& [key, value] : decode_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(decode_output_buffers[key], value.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        auto decode_output_names,
        text_decoder_compiled_model.GetSignatureOutputNames(decode_signature));

    for (const auto& name : decode_output_names) {
      if (name == TextDecoderSignatures::kDecodeLogitsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            decode_output_buffers[TextDecoderSignatures::kDecodeLogitsOutput],
            text_decoder_compiled_model.CreateOutputBuffer(
                decode_signature, TextDecoderSignatures::kDecodeLogitsOutput));
      } else if (name == TextDecoderSignatures::kLastLayerActivationsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            decode_output_buffers
                [TextDecoderSignatures::kLastLayerActivationsOutput],
            text_decoder_compiled_model.CreateOutputBuffer(
                decode_signature,
                TextDecoderSignatures::kLastLayerActivationsOutput));
      }
    }
  }

  auto verify_sig = text_decoder_compiled_model.FindSignature(verify_signature);
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      verify_output_buffers;
  if (verify_sig) {
    for (const auto& [key, value] : text_decoder_verify_input_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
    }
    // Duplicate all kv cache buffers to verify inputs.
    LITERT_ASSIGN_OR_RETURN(
        auto verify_input_names,
        text_decoder_compiled_model.GetSignatureInputNames(verify_signature));
    for (const auto& [key, value] : input_kv_cache_buffers) {
      if (absl::c_find(verify_input_names, std::string(key)) !=
          verify_input_names.end()) {
        LITERT_ASSIGN_OR_RETURN(verify_input_buffers[key], value.Duplicate());
      }
    }

    for (const auto& [key, value] : verify_output_kv_cache_slice_buffers) {
      LITERT_ASSIGN_OR_RETURN(verify_output_buffers[key], value.Duplicate());
    }

    LITERT_ASSIGN_OR_RETURN(
        auto verify_output_names,
        text_decoder_compiled_model.GetSignatureOutputNames(verify_signature));

    for (const auto& name : verify_output_names) {
      if (name == TextDecoderSignatures::kDecodeLogitsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers[TextDecoderSignatures::kDecodeLogitsOutput],
            text_decoder_compiled_model.CreateOutputBuffer(
                verify_signature, TextDecoderSignatures::kDecodeLogitsOutput));
      } else if (name == TextDecoderSignatures::kLastLayerActivationsOutput) {
        LITERT_ASSIGN_OR_RETURN(
            verify_output_buffers
                [TextDecoderSignatures::kLastLayerActivationsOutput],
            text_decoder_compiled_model.CreateOutputBuffer(
                verify_signature,
                TextDecoderSignatures::kLastLayerActivationsOutput));
      }
    }
  }

  return InferenceContext(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::WarmupInference(
    ::litert::CompiledModel& text_decoder_compiled_model,
    std::vector<ContextGroup>& context_groups,
    ::litert::CompiledModel& compiled_model_auxiliary,
    const InferenceContext& rope_inference_context,
    const InferenceContext& mask_inference_context,
    const InferenceContext& cache_update_inference_context) {
  // We need to fill the embedding input buffers with non-zero values because
  // some of the Gemma3 models contain embedding lookup preprocessing that
  // quantize a float embedding tensor into a quantized embedding tensor and use
  // 'DIV' operations in the process. Without this we risk running into: ERROR:
  // third_party/tensorflow/lite/kernels/div.cc:242 data[i] != 0 was not true.
  // ERROR: Node number 21 (DIV) failed to invoke.

  for (auto& group : context_groups) {
    if (group.text_decoder_inference_context.decode_input_buffers.contains(
            TextDecoderSignatures::kInputEmbeddings)) {
      LITERT_RETURN_IF_ERROR(Fill(
          group.text_decoder_inference_context
              .decode_input_buffers[TextDecoderSignatures::kInputEmbeddings],
          1));
    }
    if (group.text_decoder_inference_context.prefill_input_buffers.contains(
            TextDecoderSignatures::kInputEmbeddings)) {
      LITERT_RETURN_IF_ERROR(Fill(
          group.text_decoder_inference_context
              .prefill_input_buffers[TextDecoderSignatures::kInputEmbeddings],
          1));
    }
    auto result = text_decoder_compiled_model.Run(
        group.prefill_signatures.prefill,
        group.text_decoder_inference_context.prefill_input_buffers,
        group.text_decoder_inference_context.prefill_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for Text Decoder (prefill) failed: "
        << result.Error().Message();
    result = text_decoder_compiled_model.Run(
        group.decode_signature,
        group.text_decoder_inference_context.decode_input_buffers,
        group.text_decoder_inference_context.decode_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for Text Decoder (decode) failed: "
        << result.Error().Message();

    if (compiled_model_auxiliary.FindSignature(group.prefill_signatures.rope)) {
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> rope_out;
      for (const auto& [k, v] : rope_inference_context.prefill_output_buffers) {
        if (group.text_decoder_inference_context.prefill_input_buffers.contains(
                k)) {
          LITERT_ASSIGN_OR_RETURN(
              rope_out[k],
              group.text_decoder_inference_context.prefill_input_buffers.at(k)
                  .Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(rope_out[k], v.Duplicate());
        }
      }
      result = compiled_model_auxiliary.Run(
          group.prefill_signatures.rope,
          rope_inference_context.prefill_input_buffers, rope_out);
      RET_CHECK(result)
          << "Inference warmup run for RoPE signature (prefill) failed: "
          << result.Error().Message();
    }

    if (compiled_model_auxiliary.FindSignature(group.prefill_signatures.mask)) {
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> mask_out;
      for (const auto& [k, v] : mask_inference_context.prefill_output_buffers) {
        if (group.text_decoder_inference_context.prefill_input_buffers.contains(
                k)) {
          LITERT_ASSIGN_OR_RETURN(
              mask_out[k],
              group.text_decoder_inference_context.prefill_input_buffers.at(k)
                  .Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(mask_out[k], v.Duplicate());
        }
      }
      result = compiled_model_auxiliary.Run(
          group.prefill_signatures.mask,
          mask_inference_context.prefill_input_buffers, mask_out);
      RET_CHECK(result)
          << "Inference warmup run for mask signature (prefill) failed: "
          << result.Error().Message();
    }

    if (compiled_model_auxiliary.FindSignature(
            group.prefill_signatures.cache_update)) {
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> cu_in;
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> cu_out;
      for (const auto& [k, v] :
           cache_update_inference_context.prefill_input_buffers) {
        if (group.input_kv_cache_buffers.contains(k)) {
          LITERT_ASSIGN_OR_RETURN(
              cu_in[k], group.input_kv_cache_buffers.at(k).Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(cu_in[k], v.Duplicate());
        }
      }
      for (const auto& [k, v] :
           cache_update_inference_context.prefill_output_buffers) {
        if (group.input_kv_cache_buffers.contains(k)) {
          LITERT_ASSIGN_OR_RETURN(
              cu_out[k], group.input_kv_cache_buffers.at(k).Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(cu_out[k], v.Duplicate());
        }
      }
      result = compiled_model_auxiliary.Run(
          group.prefill_signatures.cache_update, cu_in, cu_out);
      RET_CHECK(result)
          << "Inference warmup run for cache update signature (prefill) "
             "failed: "
          << result.Error().Message();
    }

    if (compiled_model_auxiliary.FindSignature(
            group.decode_aux_signatures.rope)) {
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> rope_out;
      for (const auto& [k, v] : rope_inference_context.decode_output_buffers) {
        if (group.text_decoder_inference_context.decode_input_buffers.contains(
                k)) {
          LITERT_ASSIGN_OR_RETURN(
              rope_out[k],
              group.text_decoder_inference_context.decode_input_buffers.at(k)
                  .Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(rope_out[k], v.Duplicate());
        }
      }
      result = compiled_model_auxiliary.Run(
          group.decode_aux_signatures.rope,
          rope_inference_context.decode_input_buffers, rope_out);
      RET_CHECK(result)
          << "Inference warmup run for RoPE signature (decode) failed: "
          << result.Error().Message();
    }

    if (compiled_model_auxiliary.FindSignature(
            group.decode_aux_signatures.mask)) {
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> mask_out;
      for (const auto& [k, v] : mask_inference_context.decode_output_buffers) {
        if (group.text_decoder_inference_context.decode_input_buffers.contains(
                k)) {
          LITERT_ASSIGN_OR_RETURN(
              mask_out[k],
              group.text_decoder_inference_context.decode_input_buffers.at(k)
                  .Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(mask_out[k], v.Duplicate());
        }
      }
      result = compiled_model_auxiliary.Run(
          group.decode_aux_signatures.mask,
          mask_inference_context.decode_input_buffers, mask_out);
      RET_CHECK(result)
          << "Inference warmup run for mask signature (decode) failed: "
          << result.Error().Message();
    }

    if (compiled_model_auxiliary.FindSignature(
            group.decode_aux_signatures.cache_update)) {
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> cu_in;
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> cu_out;
      for (const auto& [k, v] :
           cache_update_inference_context.decode_input_buffers) {
        if (group.input_kv_cache_buffers.contains(k)) {
          LITERT_ASSIGN_OR_RETURN(
              cu_in[k], group.input_kv_cache_buffers.at(k).Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(cu_in[k], v.Duplicate());
        }
      }
      for (const auto& [k, v] :
           cache_update_inference_context.decode_output_buffers) {
        if (group.input_kv_cache_buffers.contains(k)) {
          LITERT_ASSIGN_OR_RETURN(
              cu_out[k], group.input_kv_cache_buffers.at(k).Duplicate());
        } else {
          LITERT_ASSIGN_OR_RETURN(cu_out[k], v.Duplicate());
        }
      }
      result = compiled_model_auxiliary.Run(
          group.decode_aux_signatures.cache_update, cu_in, cu_out);
      RET_CHECK(result)
          << "Inference warmup run for cache update signature (decode) "
          << "failed: " << result.Error().Message();
    }

    // Warmup verify signatures for group if present.
    if (text_decoder_compiled_model.FindSignature(group.verify_signature)) {
      result = text_decoder_compiled_model.Run(
          group.verify_signature,
          group.text_decoder_inference_context.verify_input_buffers,
          group.text_decoder_inference_context.verify_output_buffers);
      RET_CHECK(result) << "Inference warmup run for MTP verify failed: "
                        << result.Error().Message();
    }
  }

  if (compiled_model_auxiliary.FindSignature(RopeSignatures::kVerifyRope)) {
    auto result = compiled_model_auxiliary.Run(
        RopeSignatures::kVerifyRope,
        rope_inference_context.verify_input_buffers,
        rope_inference_context.verify_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for RoPE signature (verify) failed."
        << result.Error().Message();
  }

  if (compiled_model_auxiliary.FindSignature(MaskSignatures::kVerifyMask)) {
    auto result = compiled_model_auxiliary.Run(
        MaskSignatures::kVerifyMask,
        mask_inference_context.verify_input_buffers,
        mask_inference_context.verify_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for mask signature (verify) failed."
        << result.Error().Message();
  }

  if (compiled_model_auxiliary.FindSignature(
          CacheUpdateSignatures::kVerifyCacheUpdate)) {
    auto result = compiled_model_auxiliary.Run(
        CacheUpdateSignatures::kVerifyCacheUpdate,
        cache_update_inference_context.verify_input_buffers,
        cache_update_inference_context.verify_output_buffers);
    RET_CHECK(result)
        << "Inference warmup run for cache update signature (verify) failed."
        << result.Error().Message();
  }

  // Clear the master KV cache buffers on the largest context group after
  // warmup. Because all smaller groups are aliased views sharing the same
  // physical memory, clearing the largest group initializes all physical
  // memory in a single pass.
  if (!context_groups.empty()) {
    LITERT_RETURN_IF_ERROR(ClearKVCacheBuffers(
        context_groups.back()
            .text_decoder_inference_context.decode_input_buffers));
    LITERT_RETURN_IF_ERROR(ClearKVCacheBuffers(
        context_groups.back()
            .text_decoder_inference_context.prefill_input_buffers));
  }
  return absl::OkStatus();
}

absl::StatusOr<DrafterAuxContext> DrafterAuxContext::Create(
    ::litert::Environment& env, const litert::Model& mtp_aux_model,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        drafter_aux_output_buffers,
    MaskUpdateMethod mtp_mask_update_method) {
  LITERT_ASSIGN_OR_RETURN(
      auto mtp_aux_compiled_model,
      CompiledModelWrapper::Create(env, mtp_aux_model.Get(),
                                   litert::HwAccelerators::kCpu));
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      rope_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      rope_output_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mask_input_buffers;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      mask_output_buffers;

  // Drafter aux rope signature.
  LITERT_ASSIGN_OR_RETURN(
      rope_input_buffers[MtpSignatures::kInputPos],
      mtp_aux_compiled_model.CreateInputBuffer(MtpSignatures::kMtpRope,
                                               MtpSignatures::kInputPos));
  rope_input_buffers[MtpSignatures::kInputPos].Clear();

  LITERT_ASSIGN_OR_RETURN(
      auto rope_output_names,
      mtp_aux_compiled_model.GetSignatureOutputNames(MtpSignatures::kMtpRope));
  for (const auto& name : rope_output_names) {
    LITERT_ASSIGN_OR_RETURN(rope_output_buffers[name],
                            drafter_aux_output_buffers.at(name).Duplicate());
  }

  // Drafter aux mask signature.
  LITERT_ASSIGN_OR_RETURN(
      mask_input_buffers[MtpSignatures::kInputTimeStep],
      mtp_aux_compiled_model.CreateInputBuffer(MtpSignatures::kMtpMask,
                                               MtpSignatures::kInputTimeStep));
  mask_input_buffers[MtpSignatures::kInputTimeStep].Clear();

  LITERT_ASSIGN_OR_RETURN(
      mask_input_buffers[MtpSignatures::kInputTokens],
      mtp_aux_compiled_model.CreateInputBuffer(MtpSignatures::kMtpMask,
                                               MtpSignatures::kInputTokens));

  LITERT_ASSIGN_OR_RETURN(
      auto mask_output_names,
      mtp_aux_compiled_model.GetSignatureOutputNames(MtpSignatures::kMtpMask));
  for (const auto& name : mask_output_names) {
    LITERT_ASSIGN_OR_RETURN(mask_output_buffers[name],
                            drafter_aux_output_buffers.at(name).Duplicate());
  }

  LITERT_ASSIGN_OR_RETURN(
      auto drafter_rope,
      NpuRope::CreateForDrafter(&mtp_aux_compiled_model,
                                std::move(rope_input_buffers),
                                std::move(rope_output_buffers)));

  LITERT_ASSIGN_OR_RETURN(
      auto drafter_mask,
      NpuMask::CreateForDrafter(mtp_mask_update_method, &mtp_aux_compiled_model,
                                std::move(mask_input_buffers),
                                std::move(mask_output_buffers)));

  return DrafterAuxContext(std::move(mtp_aux_compiled_model),
                           std::move(drafter_rope), std::move(drafter_mask));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::WarmupDrafterInference(
    const DrafterContext& drafter_context,
    const DrafterAuxContext& drafter_aux_context) {
  auto result = drafter_context.mtp_compiled_model.Run(
      MtpSignatures::kMtpDrafter, drafter_context.mtp_input_buffers,
      drafter_context.mtp_output_buffers);
  RET_CHECK(result) << "Inference warmup run for MTP failed."
                    << result.Error().Message();

  auto mask_result = drafter_aux_context.drafter_mask.RunDrafter();
  RET_CHECK(mask_result.ok())
      << "Inference warmup run for MTP mask failed." << mask_result.message();

  auto rope_result = drafter_aux_context.drafter_rope.RunDrafter();
  RET_CHECK(rope_result.ok())
      << "Inference warmup run for MTP rope failed." << rope_result.message();
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs) {
  return Prefill(inputs, ExecutorPrefillParams());
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {
  std::optional<absl::MutexLock> lock;
  if (IsNpuSyncWorkaroundEnabled()) {
    lock.emplace(&execution_mutex_);
  }
  ran_decode_ = false;
  auto start = absl::Now();
  LITERT_ASSIGN_OR_RETURN(const auto* text_token_ids,
                          inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, text_token_ids->TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_RETURN_IF_ERROR(main_embedder_.UpdateMultiModalEmbeddings(inputs));
  LITERT_ASSIGN_OR_RETURN(auto ids,
                          ReferTensorBufferAsSpan<int32_t>(*text_token_ids));

  LITERT_ASSIGN_OR_RETURN(
      auto work_groups,
      GetOptimizedPrefillWorkGroups(prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    LITERT_RETURN_IF_ERROR(PrefillInternal(
        prefill_signature, ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
    latency_stats_.prefill_num_tokens += prefill_length;
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  LITERT_RETURN_IF_ERROR(main_embedder_.CleanupMultiModalEmbeddings());
  latency_stats_.prefill_e2e_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start);
  return absl::OkStatus();
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtNpuCompiledModelExecutor::DecodeLogits(const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<::litert::TensorBuffer>
LlmLiteRtNpuCompiledModelExecutor::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {

  if (current_step_ >= executor_settings_.GetMaxNumTokens()) {
    return absl::ResourceExhaustedError("Reached maximum number of tokens.");
  }

  if (processed_tokens_.TokenCount() != current_step_) {
    LITERT_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
  }

  if (inputs.GetTextDataPtr().ok()) {
    auto token_ids_buffer = inputs.GetTextTokenIdsPtr();
    if (token_ids_buffer.ok()) {
      auto input_tensor_size = (*token_ids_buffer)->PackedSize();
      if (input_tensor_size && *input_tensor_size != 0) {
        RET_CHECK_EQ(*input_tensor_size, sizeof(int32_t));
        LITERT_ASSIGN_OR_RETURN(
            auto ids, ReferTensorBufferAsSpan<int32_t>(**token_ids_buffer));
        if (ids[0] >= 0) {
          processed_tokens_.InvalidatePendingInputToken();
          std::shared_ptr<TokenData> token =
              std::make_shared<TokenData>(ids[0]);
          LITERT_RETURN_IF_ERROR(
              processed_tokens_.AddPendingInputToken({token}));
        }
      }
    }
  }

  auto [internal_start_step, pending_input_token] =
      processed_tokens_.GetNextUnprocessedToken();
  if (pending_input_token.empty()) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  bool last_run_is_decode = ran_decode_;

  std::shared_ptr<TokenData> token = pending_input_token[0];
  LITERT_RETURN_IF_ERROR(main_embedder_.LookupDecode(token.get()));

  LITERT_RETURN_IF_ERROR(DecodeInternal(internal_start_step, token));
  LITERT_RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());

  const auto& src_buffer =
      ActiveContextGroup()
          .text_decoder_inference_context
          .decode_output_buffers[TextDecoderSignatures::kDecodeLogitsOutput];

  LITERT_ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  LITERT_ASSIGN_OR_RETURN(auto output_logits,
                          CreateTensorBuffer<float>({1, 1, vocab_size}));

  LITERT_RETURN_IF_ERROR(
      DequantizeLogits(src_buffer, output_logits, per_tensor_logits_scale_,
                       per_tensor_logits_zero_point_, false));

  if (ConstrainedDecoder* constrained_decoder =
          decode_params.GetConstrainedDecoder();
      constrained_decoder != nullptr) {
    std::vector<int> current_token_ids = {token->id()};
    if (last_run_is_decode) {
      LITERT_RETURN_IF_ERROR(
          constrained_decoder->UpdateState(absl::MakeSpan(current_token_ids)));
    }
    LITERT_RETURN_IF_ERROR(constrained_decoder->ProcessLogits(output_logits));
  }

  current_step_++;
  ran_decode_ = true;

  return output_logits;
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::DecodeNonSpeculative(
    const ExecutorDecodeParams& decode_params, absl::Time start_time) {
  int max_index = kInvalidTokenId;

  if (decode_params.GetConstrainedDecoder() != nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto masked_logits,
                            DecodeLogits(ExecutorInputs(), decode_params));
    auto start_sample = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        max_index,
        ApplyGreedySampling(masked_logits,
                            npu_config_.enable_neon_for_npu_greedy_sampling));
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_sample);
  } else {
    auto start_get_token = absl::Now();
    auto [internal_start_step, pending_input_token] =
        processed_tokens_.GetNextUnprocessedToken();
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_get_token);

    if (pending_input_token.empty()) {
      return absl::InvalidArgumentError("No id available to be decoded.");
    }

    NPU_EXECUTOR_LOG(INFO)
        << "Step " << internal_start_step
        << ": Running Main Decode Signature (Non-Speculative)";
    LITERT_RETURN_IF_ERROR(
        DecodeInternal(internal_start_step, pending_input_token[0]));

    auto start_mark = absl::Now();
    LITERT_RETURN_IF_ERROR(
        processed_tokens_.MarkPendingInputTokenAsProcessed());
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_mark);

    // Sample the output of the main decode.
    auto start_sample = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        max_index, ApplyGreedySampling(
                       ActiveContextGroup()
                           .text_decoder_inference_context.decode_output_buffers
                               [TextDecoderSignatures::kDecodeLogitsOutput],
                       npu_config_.enable_neon_for_npu_greedy_sampling));
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_sample);
    ++current_step_;
    ran_decode_ = true;
  }

  std::shared_ptr<TokenData> last_output_token =
      std::make_shared<TokenData>(max_index);

  auto start_lookup = absl::Now();
  LITERT_RETURN_IF_ERROR(main_embedder_.LookupDecode(last_output_token.get()));
  latency_stats_.decode_embedder_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_lookup);

  auto start_add = absl::Now();
  LITERT_RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(last_output_token)}));
  latency_stats_.decode_token_queue_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_add);

  latency_stats_.decode_e2e_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_time);
  latency_stats_.decode_num_tokens++;
  return std::vector<std::vector<int>>{{max_index}};
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::PopPendingAcceptedToken(
    absl::Time start_time) {
  auto start_queue = absl::Now();
  int next_token_id = pending_accepted_tokens_.front();
  pending_accepted_tokens_.erase(pending_accepted_tokens_.begin());

  NPU_EXECUTOR_LOG(INFO) << "Decode returning token from queue: "
                         << next_token_id << " (Remaining in queue: "
                         << pending_accepted_tokens_.size() << ")";

  std::shared_ptr<TokenData> next_token =
      std::make_shared<TokenData>(next_token_id);
  auto start_lookup = absl::Now();
  LITERT_RETURN_IF_ERROR(main_embedder_.LookupDecode(next_token.get()));
  latency_stats_.decode_embedder_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_lookup);
  // We must add it as a pending input token so that the NEXT Decode call
  // can find it via GetNextUnprocessedToken if the queue is empty.
  auto mark_status = processed_tokens_.MarkPendingInputTokenAsProcessed();
  if (!mark_status.ok() && !absl::IsNotFound(mark_status)) {
    return mark_status;
  }
  LITERT_RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(next_token)}));
  current_step_++;
  latency_stats_.decode_token_queue_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_queue);

  latency_stats_.decode_e2e_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_time);
  latency_stats_.decode_num_tokens++;
  return std::vector<std::vector<int>>{{next_token_id}};
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtNpuCompiledModelExecutor::Decode(
    const ExecutorDecodeParams& decode_params) {
  auto start = absl::Now();

  if (current_step_ >= executor_settings_.GetMaxNumTokens()) {
    return absl::ResourceExhaustedError("Reached maximum number of tokens.");
  }

  if (processed_tokens_.TokenCount() != current_step_) {
    auto start_queue = absl::Now();
    LITERT_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
    latency_stats_.decode_token_queue_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_queue);
  }

  if (!pending_accepted_tokens_.empty()) {
    return PopPendingAcceptedToken(start);
  }

  // Early return for standard non-speculative decode or when constrained
  // decoding is requested.
  if (speculative_decoding_type_ != SpeculativeDecodingType::kMTP ||
      decode_params.GetConstrainedDecoder() != nullptr) {
    return DecodeNonSpeculative(decode_params, start);
  }

  // Speculative MTP cycle.
  auto start_get_token = absl::Now();
  auto [internal_start_step, pending_input_token] =
      processed_tokens_.GetNextUnprocessedToken();
  latency_stats_.decode_token_queue_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_get_token);

  if (pending_input_token.empty()) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  int mtp_start_step = internal_start_step;
  int mtp_start_token_id = pending_input_token[0]->id();

  auto start_mark = absl::Now();
  LITERT_RETURN_IF_ERROR(processed_tokens_.MarkPendingInputTokenAsProcessed());
  latency_stats_.decode_token_queue_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_mark);

  if (!has_valid_verify_activations_) {
    // Cold start (e.g. right after prefill): we run the main model decode once
    // to produce the initial anchor token and seed activations.
    NPU_EXECUTOR_LOG(INFO) << "Step " << internal_start_step
                           << ": Cold start - Running Main Decode";
    LITERT_RETURN_IF_ERROR(
        DecodeInternal(internal_start_step, pending_input_token[0]));

    // Sample the output of the main decode to get the 'good' token for MTP.
    auto start_sample = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        mtp_start_token_id,
        ApplyGreedySampling(
            ActiveContextGroup()
                .text_decoder_inference_context.decode_output_buffers
                    [TextDecoderSignatures::kDecodeLogitsOutput],
            npu_config_.enable_neon_for_npu_greedy_sampling));
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_sample);

    // The MTP drafter starts from the position of the token we just generated.
    mtp_start_step = internal_start_step + 1;
  } else {
    // Warm cycle: Skip main decode entirely. RunDrafterLoop directly consumes
    // the cached last_verify_activations_ and writes them to the drafter input
    // buffer.
    NPU_EXECUTOR_LOG(INFO)
        << "Step " << internal_start_step
        << ": Skipping Main Decode (Using verify activations)";
  }

  NPU_EXECUTOR_LOG(INFO) << "Step " << mtp_start_step
                         << ": Starting MTP Speculative Cycle";
  NPU_EXECUTOR_LOG(INFO) << "    [Verify] Step -1 at pos " << mtp_start_step - 1
                         << ": Good token ID " << mtp_start_token_id;

  // Ensure the active context group can accommodate the entire verification
  // batch (anchor token + N draft tokens).
  LITERT_ASSIGN_OR_RETURN(
      RankedTensorType verify_output_tensor_type,
      ActiveContextGroup()
          .text_decoder_inference_context
          .verify_output_buffers[MtpSignatures::kInputActivations]
          .TensorType());
  const int drafter_sequence_length =
      verify_output_tensor_type.Layout().Dimensions()[1] - 1;
  LITERT_RETURN_IF_ERROR(
      SwitchContextSizeIfRequired(mtp_start_step + drafter_sequence_length));

  LITERT_ASSIGN_OR_RETURN(std::vector<int> draft_tokens,
                          RunDrafterLoop(mtp_start_step, mtp_start_token_id));
  auto start_verify = absl::Now();
  LITERT_RETURN_IF_ERROR(
      RunVerifierBatch(mtp_start_step, mtp_start_token_id, draft_tokens));
  latency_stats_.decode_llm_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_verify);

  auto start_rs = absl::Now();
  LITERT_ASSIGN_OR_RETURN(
      auto rs_result,
      PerformRejectionSampling(
          draft_tokens,
          ActiveContextGroup()
              .text_decoder_inference_context.verify_output_buffers
                  [TextDecoderSignatures::kVerifyLogitsOutput]));
  latency_stats_.decode_mtp_rejection_sampling_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_rs);

  NPU_EXECUTOR_LOG(INFO) << "  MTP Accepted " << rs_result.num_accepted
                         << " draft tokens. Bonus: "
                         << rs_result.bonus_token_id;

  auto start_commit = absl::Now();
  LITERT_RETURN_IF_ERROR(CommitVerifiedKVCache(mtp_start_step));
  latency_stats_.decode_cache_update_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_commit);

  // Prepare tokens to be returned.
  std::vector<int> all_accepted;
  if (!has_valid_verify_activations_) {
    all_accepted.push_back(mtp_start_token_id);
  }

  for (int i = 0; i < rs_result.num_accepted; ++i) {
    all_accepted.push_back(draft_tokens[i]);
  }
  all_accepted.push_back(rs_result.bonus_token_id);

  // Prepare next activation slice.
  {
    auto start_act_copy = absl::Now();
    const auto& verify_activations_buffer =
        ActiveContextGroup()
            .text_decoder_inference_context.verify_output_buffers
                [TextDecoderSignatures::kLastLayerActivationsOutput];
    LITERT_ASSIGN_OR_RETURN(
        auto full_activations,
        CopyRawBytesFromTensorBuffer(verify_activations_buffer));
    // Divide total bytes by the sequence length (1 current token + N draft
    // tokens) to get the number of bytes per token activation.
    size_t dratfer_seq_len = draft_tokens.size() + 1;
    size_t hidden_size_in_bytes = full_activations.size() / dratfer_seq_len;
    last_verify_activations_.resize(hidden_size_in_bytes);
    memcpy(
        last_verify_activations_.data(),
        full_activations.data() + rs_result.num_accepted * hidden_size_in_bytes,
        hidden_size_in_bytes);
    has_valid_verify_activations_ = true;
    latency_stats_.decode_mtp_activation_copy_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start_act_copy);
  }

  // Return the first token now, queue the rest for future Decode() calls.
  int first_token_id = all_accepted[0];
  for (size_t i = 1; i < all_accepted.size(); ++i) {
    pending_accepted_tokens_.push_back(all_accepted[i]);
  }

  NPU_EXECUTOR_LOG(INFO) << "MTP cycle returning first token: "
                         << first_token_id
                         << " (Queued: " << pending_accepted_tokens_.size()
                         << ")";

  std::shared_ptr<TokenData> first_token =
      std::make_shared<TokenData>(first_token_id);
  auto start_lookup = absl::Now();
  LITERT_RETURN_IF_ERROR(main_embedder_.LookupDecode(first_token.get()));
  latency_stats_.decode_embedder_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_lookup);
  // For MTP, we need to mark them as processed so the next step's
  // GetNextUnprocessedToken works correctly.
  LITERT_RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(first_token)}));
  current_step_++;

  latency_stats_.decode_num_tokens++;
  latency_stats_.decode_e2e_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start);
  return std::vector<std::vector<int>>{{first_token_id}};
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::SwitchContextSizeIfRequired(
    int required_tokens) {
  if (sorted_supported_context_sizes_.empty() || context_groups_.size() <= 1) {
    return absl::OkStatus();
  }
  int current_context_size = ActiveContextGroup().context_size;
  int target_capacity = required_tokens;
  if (target_capacity <= current_context_size) {
    return absl::OkStatus();
  }

  int new_index = -1;
  for (size_t i = 0; i < sorted_supported_context_sizes_.size(); ++i) {
    if (sorted_supported_context_sizes_[i] >= target_capacity) {
      new_index = i;
      break;
    }
  }

  if (new_index == -1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Required tokens (", required_tokens,
                     ") exceeds maximum supported NPU context size (",
                     sorted_supported_context_sizes_.back(), ")."));
  }

  int old_index = active_context_group_index_;
  int old_size = current_context_size;
  int new_size = sorted_supported_context_sizes_[new_index];

  if (current_step_ > 0 && new_size > old_size) {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_cache_.CopyKVCache(
        context_groups_[old_index].input_kv_cache_buffers,
        context_groups_[new_index].input_kv_cache_buffers, current_step_));
    auto end = absl::Now();
    ABSL_LOG(INFO) << "CopyKVCache took "
                   << absl::ToDoubleMilliseconds(end - start) << " ms for "
                   << current_step_ << " elements.";
  }

  active_context_group_index_ = new_index;
  const auto& active_group = context_groups_[new_index];
  LITERT_RETURN_IF_ERROR(main_cache_.UpdateKVCacheBuffers(
      active_group.input_kv_cache_buffers,
      active_group.text_decoder_inference_context.prefill_output_buffers,
      active_group.text_decoder_inference_context.decode_output_buffers,
      active_group.text_decoder_inference_context.verify_output_buffers));
  LITERT_RETURN_IF_ERROR(main_mask_.UpdateOutputBuffers(
      active_group.text_decoder_inference_context.prefill_input_buffers,
      active_group.text_decoder_inference_context.decode_input_buffers,
      active_group.text_decoder_inference_context.verify_input_buffers));
  LITERT_RETURN_IF_ERROR(main_rope_.UpdateOutputBuffers(
      active_group.text_decoder_inference_context.prefill_input_buffers,
      active_group.text_decoder_inference_context.decode_input_buffers,
      active_group.text_decoder_inference_context.verify_input_buffers));
  LITERT_RETURN_IF_ERROR(main_embedder_.UpdateOutputBuffers(
      active_group.text_decoder_inference_context.prefill_input_buffers,
      active_group.text_decoder_inference_context.decode_input_buffers,
      active_group.text_decoder_inference_context.verify_input_buffers));
  if (drafter_context_.has_value()) {
    LITERT_RETURN_IF_ERROR(drafter_context_->UpdateKVCacheBuffers(
        active_group.input_kv_cache_buffers));
  }

  ABSL_LOG(INFO) << "Switched context size from " << old_size << " to "
                 << new_size << ". Selected prefill signature: \""
                 << ActiveContextGroup().prefill_signatures.prefill
                 << "\", decode signature: \""
                 << ActiveContextGroup().decode_signature << "\".";
  return absl::OkStatus();
}

// Prefill internal implementation, for one prefill call to the compiled model
// with a certain length.
absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillInternal(
    absl::string_view prefill_signature, absl::Span<const int> ids) {
  // 1. Validate actual token count against user-configured maximum tokens.
  if (current_step_ + ids.size() > executor_settings_.GetMaxNumTokens()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Prefill length (", ids.size(), ") plus current step (",
                     current_step_, ") exceeds max sequence length (",
                     executor_settings_.GetMaxNumTokens(), ")."));
  }

  // 2. Extract runner size from signature (or fallback to ids.size()).
  int runner_size = ids.size();
  if (absl::StartsWith(prefill_signature, "prefill_")) {
    absl::string_view rest = prefill_signature.substr(8);
    size_t under = rest.find('_');
    absl::string_view num_str =
        (under != absl::string_view::npos) ? rest.substr(0, under) : rest;
    int parsed = 0;
    if (absl::SimpleAtoi(num_str, &parsed) && parsed > 0) {
      runner_size = parsed;
    }
  }

  // 3. Switch context size based on total required physical tokens (including
  // chunk padding).
  LITERT_RETURN_IF_ERROR(
      SwitchContextSizeIfRequired(current_step_ + runner_size));

  // 4. Verify that the padded chunk fits in the active hardware context size.
  if (ActiveContextGroup().context_size > 0 &&
      current_step_ + runner_size > ActiveContextGroup().context_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Padded prefill length (", runner_size, ") plus current step (",
        current_step_, ") exceeds NPU context size (",
        ActiveContextGroup().context_size, ")."));
  }

  const ResolvedPrefillSignatures active_sigs = BuildResolvedPrefillSignatures(
      runner_size, ActiveContextGroup().context_size);

  auto [internal_start_step, pending_input_token] =
      processed_tokens_.GetNextUnprocessedToken();
  auto start_prepare_inputs = absl::Now();
  std::vector<int> tokens_to_embed;
  tokens_to_embed.reserve(ids.size());

  if (processed_tokens_.TokenCount() != current_step_) {
    LITERT_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(current_step_));
  }
  // Check if we have a pending input token, e.g. because we are running a
  // multi-turn conversation or a multi-chunk prefill.
  std::vector<int32_t> prefill_positions;
  prefill_positions.reserve(ids.size());
  if (!pending_input_token.empty()) {
    tokens_to_embed.push_back(pending_input_token[0]->id());
    prefill_positions.push_back(internal_start_step);
    LITERT_RETURN_IF_ERROR(
        processed_tokens_.MarkPendingInputTokenAsProcessed());
  }

  std::vector<int> processed_input_tokens;
  // We will not fill the last token of the current input into the compiled
  // model input buffers just yet. It will be stored in the
  // 'processed_tokens_' and used in the next prefill or decode.
  processed_input_tokens.reserve(ids.size() - 1);
  for (int i = 0; i < ids.size() - 1; ++i, ++current_step_) {
    tokens_to_embed.push_back(ids[i]);
    prefill_positions.push_back(current_step_);
    processed_input_tokens.push_back(ids[i]);
  }
  processed_tokens_.AddProcessedTokens(processed_input_tokens);
  if (!processed_input_tokens.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill tokens: "
                           << FormatFirstN<int>(processed_input_tokens);
  }

  // Set prefill RoPE positions.
  LITERT_RETURN_IF_ERROR(main_rope_.SetPrefillPositions(prefill_positions));
  // Set prefill mask inputs (timestep, tokens, valid mask).
  LITERT_RETURN_IF_ERROR(
      main_mask_.SetPrefillInput(internal_start_step, tokens_to_embed));
  // Set prefill cache inputs (positions, valid mask).
  LITERT_RETURN_IF_ERROR(main_cache_.SetPrefillPositions(prefill_positions));

  latency_stats_.prefill_prepare_input_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_prepare_inputs);

  // Add the last token of the current input as a pending input token, to be
  // used in the next prefill or decode.
  auto last_input_token = std::make_shared<TokenData>(ids.back());

  auto start_embedder = absl::Now();
  LITERT_RETURN_IF_ERROR(main_embedder_.RunPrefill(
      active_sigs.embedder,
      pending_input_token.empty() ? nullptr : pending_input_token[0].get(),
      processed_input_tokens, last_input_token.get()));
  latency_stats_.prefill_embedder_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_embedder);

  // Add the last input token to the pending input token list.
  LITERT_RETURN_IF_ERROR(
      processed_tokens_.AddPendingInputToken({std::move(last_input_token)}));
  ++current_step_;

  // Invoke embedder per layer signature if it exists.
  if (main_embedder_.HasPerLayerEmbeddings()) {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_embedder_.RunPrefillPerLayer(
        active_sigs.embedder_per_layer, tokens_to_embed));
    latency_stats_.prefill_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  return PrefillCommonPipeline(active_sigs);
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillInternalFromEmbeddings(
    absl::string_view prefill_signature,
    absl::Span<const int32_t> sliced_tokens, absl::Span<const float> embeddings,
    absl::Span<const float> ple_embeddings,
    absl::Span<const int32_t> seq_positions) {
  // 1. Validate actual token count against user-configured maximum tokens.
  if (current_step_ + seq_positions.size() >
      executor_settings_.GetMaxNumTokens()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Prefill length (", seq_positions.size(), ") plus current step (",
        current_step_, ") exceeds max sequence length (",
        executor_settings_.GetMaxNumTokens(), ")."));
  }

  // 2. Extract runner size from signature (or fallback to
  // seq_positions.size()).
  int runner_size = seq_positions.size();
  if (absl::StartsWith(prefill_signature, "prefill_")) {
    absl::string_view rest = prefill_signature.substr(8);
    size_t under = rest.find('_');
    absl::string_view num_str =
        (under != absl::string_view::npos) ? rest.substr(0, under) : rest;
    int parsed = 0;
    if (absl::SimpleAtoi(num_str, &parsed) && parsed > 0) {
      runner_size = parsed;
    }
  }

  // 3. Switch context size based on total required physical tokens (including
  // chunk padding).
  LITERT_RETURN_IF_ERROR(
      SwitchContextSizeIfRequired(current_step_ + runner_size));

  // 4. Verify that the padded chunk fits in the active hardware context size.
  if (ActiveContextGroup().context_size > 0 &&
      current_step_ + runner_size > ActiveContextGroup().context_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Padded prefill length (", runner_size, ") plus current step (",
        current_step_, ") exceeds NPU context size (",
        ActiveContextGroup().context_size, ")."));
  }

  const ResolvedPrefillSignatures active_sigs = BuildResolvedPrefillSignatures(
      runner_size, ActiveContextGroup().context_size);

  NPU_EXECUTOR_LOG(INFO)
      << "PrefillInternalFromEmbeddings called: PLE config: type="
      << static_cast<int>(main_embedder_.PleParams().output_type)
      << ", mul_scale=" << main_embedder_.PleParams().mul_scale
      << ", output_scale=" << main_embedder_.PleParams().output_scale
      << ", zero_point=" << main_embedder_.PleParams().final_zero_point;
  if (!sliced_tokens.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill tokens: " << FormatFirstN(sliced_tokens);
  }
  if (!embeddings.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill embeddings: "
                           << FormatFirstN(embeddings);
  }
  if (!ple_embeddings.empty()) {
    NPU_EXECUTOR_LOG(INFO) << "Prefill PLE embeddings: "
                           << FormatFirstN(ple_embeddings);
  }
  // Set prefill input embeddings.
  {
    auto& buffer =
        ActiveContextGroup()
            .text_decoder_inference_context
            .prefill_input_buffers[TextDecoderSignatures::kInputEmbeddings];
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
    NPU_EXECUTOR_LOG(INFO) << "Embeddings buffer element type: "
                           << static_cast<int>(tensor_type.ElementType());
    const auto elem_type = tensor_type.ElementType();
    const size_t elem_size = elem_type == litert::ElementType::Float16
                                 ? sizeof(tflite::half)
                                 : sizeof(float);

    LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
    RET_CHECK_GE(buffer_size, embeddings.size() * elem_size);

    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            buffer, ::litert::TensorBuffer::LockMode::kWrite));

    const std::vector<float> default_emb =
        main_embedder_.GetDefaultEmbeddingVector();

    const auto& dims = tensor_type.Layout().Dimensions();
    if (dims.size() < 3) {
      return absl::InternalError(
          "Prefill input embeddings tensor has unexpected shape.");
    }
    const size_t embedding_dim =
        default_emb.empty() ? dims[2] : default_emb.size();
    size_t starting_token = embeddings.size() / embedding_dim;
    size_t num_tokens_to_fill = buffer_size / (embedding_dim * elem_size);

    if (elem_type == litert::ElementType::Float16) {
      tflite::half* buffer_ptr =
          static_cast<tflite::half*>(lock_and_addr.second);
      for (size_t i = 0; i < embeddings.size(); ++i) {
        buffer_ptr[i] = tflite::half(embeddings[i]);
      }
      tflite::half* padding_ptr = buffer_ptr + starting_token * embedding_dim;
      std::vector<tflite::half> fp16_default_emb(embedding_dim,
                                                 tflite::half(0.0f));
      if (default_emb.size() == embedding_dim) {
        for (size_t i = 0; i < embedding_dim; ++i) {
          fp16_default_emb[i] = tflite::half(default_emb[i]);
        }
      }
      for (size_t i = starting_token; i < num_tokens_to_fill; ++i) {
        std::memcpy(padding_ptr, fp16_default_emb.data(),
                    embedding_dim * sizeof(tflite::half));
        padding_ptr += embedding_dim;
      }
    } else {
      float* buffer_ptr = static_cast<float*>(lock_and_addr.second);
      std::memcpy(buffer_ptr, embeddings.data(),
                  embeddings.size() * sizeof(float));
      float* padding_ptr = buffer_ptr + starting_token * embedding_dim;
      if (default_emb.size() == embedding_dim) {
        for (size_t i = starting_token; i < num_tokens_to_fill; ++i) {
          std::memcpy(padding_ptr, default_emb.data(),
                      embedding_dim * sizeof(float));
          padding_ptr += embedding_dim;
        }
      } else {
        std::memset(padding_ptr, 0,
                    (num_tokens_to_fill - starting_token) * embedding_dim *
                        sizeof(float));
      }
    }
  }

  // Set prefill positions.
  LITERT_RETURN_IF_ERROR(main_rope_.SetPrefillPositions(seq_positions));

  // Set prefill per-layer embeddings if provided.
  if (!ple_embeddings.empty()) {
    LITERT_RETURN_IF_ERROR(
        main_embedder_.WriteAndPadPleEmbeddings(env_, ple_embeddings));
  }

  // Set prefill mask input (timestep, tokens, valid mask).
  LITERT_RETURN_IF_ERROR(
      main_mask_.SetPrefillInput(seq_positions[0], sliced_tokens));

  // Set prefill cache inputs (positions, valid mask).
  LITERT_RETURN_IF_ERROR(main_cache_.SetPrefillPositions(seq_positions));

  return PrefillCommonPipeline(active_sigs);
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::PrefillCommonPipeline(
    const ResolvedPrefillSignatures& sigs) {
  {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_rope_.RunPrefill(sigs.rope));
    latency_stats_.prefill_rope_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_mask_.RunPrefill(sigs.mask));
    latency_stats_.prefill_mask_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    auto start = absl::Now();
    auto res = text_decoder_compiled_model_.Run(
        sigs.prefill,
        ActiveContextGroup()
            .text_decoder_inference_context.prefill_input_buffers,
        ActiveContextGroup()
            .text_decoder_inference_context.prefill_output_buffers);
    RET_CHECK(res) << "Failed to run text decoder model."
                   << res.Error().Message();
    latency_stats_.prefill_llm_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_cache_.RunPrefill(sigs.cache_update));
    latency_stats_.prefill_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeInternal(
    int step, std::shared_ptr<TokenData> token) {
  if (step >= executor_settings_.GetMaxNumTokens()) {
    return absl::ResourceExhaustedError("Reached maximum number of tokens.");
  }
  LITERT_RETURN_IF_ERROR(SwitchContextSizeIfRequired(step + 1));
  int id = token->id();
  auto start_prepare_inputs = absl::Now();

  if (id == kInvalidTokenId && token->embedding().empty()) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  // Always update decode input position and timestep, even if
  // run_rope_and_mask is false. The LLM and Cache Update models still need to
  // know the current step.

  // 1. RoPE position
  LITERT_RETURN_IF_ERROR(main_rope_.SetDecodePosition(step));

  // 2. Mask inputs (timestep, token, valid_mask)
  LITERT_RETURN_IF_ERROR(main_mask_.SetDecodeInput(step, id));

  // 3. Cache update position
  LITERT_RETURN_IF_ERROR(main_cache_.SetDecodePosition(step));

  latency_stats_.decode_prepare_input_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_prepare_inputs);

  auto start_embedder = absl::Now();
  LITERT_RETURN_IF_ERROR(main_embedder_.RunDecode(*token));
  latency_stats_.decode_embedder_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_embedder);

  {
    if (!token->per_layer_embedding().empty()) {
      LITERT_RETURN_IF_ERROR(main_embedder_.WriteDecodePleEmbeddings(
          token->per_layer_embedding()));
    } else if (main_embedder_.HasPerLayerEmbeddings()) {
      auto start = absl::Now();
      LITERT_RETURN_IF_ERROR(main_embedder_.RunDecodePerLayer(token->id()));
      latency_stats_.decode_embedder_per_layer_inference_latency_us +=
          absl::ToInt64Microseconds(absl::Now() - start);
    }
  }

  {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(
        main_rope_.RunDecode(ActiveContextGroup().decode_aux_signatures.rope));
    latency_stats_.decode_rope_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(
        main_mask_.RunDecode(ActiveContextGroup().decode_aux_signatures.mask));
    latency_stats_.decode_mask_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    auto start = absl::Now();
    auto res = text_decoder_compiled_model_.Run(
        ActiveContextGroup().decode_signature,
        ActiveContextGroup()
            .text_decoder_inference_context.decode_input_buffers,
        ActiveContextGroup()
            .text_decoder_inference_context.decode_output_buffers);
    latency_stats_.decode_llm_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    RET_CHECK(res) << "Failed to run text decoder model."
                   << res.Error().Message();
  }

  {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_cache_.RunDecode(
        ActiveContextGroup().decode_aux_signatures.cache_update));
    latency_stats_.decode_cache_update_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::DecodeSingleToken(
    size_t idx, absl::Span<const int32_t> seq_pos_span,
    absl::Span<const int32_t> tokens_span, absl::Span<const float> embeddings,
    size_t embedding_dim, absl::Span<const float> ple_embeddings,
    size_t ple_dim) {
  int step = seq_pos_span[idx];
  int token_id = tokens_span.empty() ? -1 : tokens_span[idx];
  auto token = std::make_shared<TokenData>(token_id);
  if (!embeddings.empty()) {
    token->mutable_embedding() =
        std::vector<float>(embeddings.begin() + idx * embedding_dim,
                           embeddings.begin() + (idx + 1) * embedding_dim);
  }
  if (!ple_embeddings.empty()) {
    token->mutable_per_layer_embedding() =
        std::vector<float>(ple_embeddings.begin() + idx * ple_dim,
                           ple_embeddings.begin() + (idx + 1) * ple_dim);
  }

  LITERT_RETURN_IF_ERROR(DecodeInternal(step, token));
  current_step_ = step + 1;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>>
LlmLiteRtNpuCompiledModelExecutor::RunDrafterLoop(int start_step,
                                                  int current_token_id) {
  if (!drafter_context_.has_value() || !drafter_aux_context_.has_value()) {
    return absl::InternalError("Drafter contexts not initialized.");
  }
  // Get model drafter sequence length.
  LITERT_ASSIGN_OR_RETURN(
      RankedTensorType verify_output_tensor_type,
      ActiveContextGroup()
          .text_decoder_inference_context
          .verify_output_buffers[MtpSignatures::kInputActivations]
          .TensorType());
  const int drafter_sequence_length =
      verify_output_tensor_type.Layout().Dimensions()[1] - 1;
  auto& ctx = drafter_context_.value();
  auto& aux_ctx = drafter_aux_context_.value();

  std::vector<int> draft_tokens;

  // Initial activations from the last layer of the text decoder (or previous
  // cycle).
  std::vector<uint8_t> current_activations;
  if (!has_valid_verify_activations_) {
    LITERT_ASSIGN_OR_RETURN(
        current_activations,
        CopyRawBytesFromTensorBuffer(
            ActiveContextGroup()
                .text_decoder_inference_context.decode_output_buffers
                    [TextDecoderSignatures::kLastLayerActivationsOutput]));
  } else {
    current_activations = last_verify_activations_;
  }

  int input_token_id = current_token_id;

  // Set drafter RoPE position.
  LITERT_RETURN_IF_ERROR(aux_ctx.drafter_rope.SetDecodePosition(start_step));

  // Set drafter mask input.
  LITERT_RETURN_IF_ERROR(
      aux_ctx.drafter_mask.SetDecodeInput(start_step, input_token_id));

  // Set drafter input position.
  LITERT_RETURN_IF_ERROR(ctx.SetInputPos(start_step));

  // // Run Rope/Mask for drafter.
  auto start = absl::Now();
  LITERT_RETURN_IF_ERROR(aux_ctx.drafter_rope.RunDrafter());
  latency_stats_.decode_rope_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start);

  start = absl::Now();
  LITERT_RETURN_IF_ERROR(aux_ctx.drafter_mask.RunDrafter());
  latency_stats_.decode_mask_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start);

  for (int i = 0; i < drafter_sequence_length; ++i) {
    std::vector<float> draft_embedding;
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(
        main_embedder_.LookupDecode(input_token_id, draft_embedding));
    latency_stats_.decode_embedder_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    RET_CHECK(!draft_embedding.empty())
        << "Embedding lookup not available for MTP.";

    {
      // Lock the drafter activations buffer for writing.
      LITERT_ASSIGN_OR_RETURN(
          auto drafter_activations_lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              ctx.mtp_input_buffers[MtpSignatures::kInputActivations],
              ::litert::TensorBuffer::LockMode::kWrite));

      uint8_t* base_ptr =
          static_cast<uint8_t*>(drafter_activations_lock_and_addr.second);

      // Copy embedding FIRST.
      memcpy(base_ptr, draft_embedding.data(),
             draft_embedding.size() * sizeof(float));
      // Copy activations SECOND.
      memcpy(base_ptr + draft_embedding.size() * sizeof(float),
             current_activations.data(), current_activations.size());
    }
    start = absl::Now();
    LITERT_RETURN_IF_ERROR(ctx.mtp_compiled_model.Run(
        MtpSignatures::kMtpDrafter, ctx.mtp_input_buffers,
        ctx.mtp_output_buffers));
    latency_stats_.decode_drafter_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);

    start = absl::Now();
    LITERT_ASSIGN_OR_RETURN(
        int draft_id, ApplyGreedySampling(
                          ctx.mtp_output_buffers[MtpSignatures::kOutputLogits],
                          npu_config_.enable_neon_for_npu_greedy_sampling));
    latency_stats_.decode_sampling_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
    draft_tokens.push_back(draft_id);

    NPU_EXECUTOR_LOG(INFO) << "    [Drafter] Step " << i << " at pos "
                           << start_step << ": Generated Token ID " << draft_id;
    if (i < drafter_sequence_length - 1) {
      LITERT_ASSIGN_OR_RETURN(
          current_activations,
          CopyRawBytesFromTensorBuffer(
              ctx.mtp_output_buffers[MtpSignatures::kOutputActivations]));
      input_token_id = draft_id;
    }
  }

  return draft_tokens;
}

namespace {
// Helper to sample from a slice of logits pointer directly without
// acquiring/releasing locks.
absl::StatusOr<int> SampleLogitsSliceFromLockedPtr(
    ::litert::ElementType element_type, const uint8_t* base_ptr, int batch_idx,
    int vocab_size, size_t element_size, bool enable_neon_sampling) {
  const uint8_t* logits_ptr =
      base_ptr + (batch_idx * vocab_size * element_size);

  auto find_max_index_plain = [&](auto ptr) {
    int max_idx = 0;
    auto max_val = ptr[0];
    for (int i = 1; i < vocab_size; ++i) {
      if (ptr[i] > max_val) {
        max_val = ptr[i];
        max_idx = i;
      }
    }
    return max_idx;
  };

  if (element_type == ::litert::ElementType::Float32) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexFloatNeon(reinterpret_cast<const float*>(logits_ptr),
                                   vocab_size);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (enable_neon_sampling) {
      return FindMaxIndexSse2Float(reinterpret_cast<const float*>(logits_ptr),
                                   vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const float*>(logits_ptr));
  } else if (element_type == ::litert::ElementType::Int16) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexInt16Neon(reinterpret_cast<const int16_t*>(logits_ptr),
                                   vocab_size);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (enable_neon_sampling) {
      return FindMaxIndexSse2Int16(reinterpret_cast<const int16_t*>(logits_ptr),
                                   vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const int16_t*>(logits_ptr));
  } else if (element_type == ::litert::ElementType::Int8) {
#if defined(__ANDROID__) && defined(__ARM_NEON)
    if (enable_neon_sampling) {
      return FindMaxIndexInt8Neon(reinterpret_cast<const int8_t*>(logits_ptr),
                                  vocab_size);
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (enable_neon_sampling) {
      return FindMaxIndexSse2Int8(reinterpret_cast<const int8_t*>(logits_ptr),
                                  vocab_size);
    }
#endif
    return find_max_index_plain(reinterpret_cast<const int8_t*>(logits_ptr));
  }

  return absl::UnimplementedError("Unsupported logit type for batch sampling.");
}
}  // namespace

absl::Status LlmLiteRtNpuCompiledModelExecutor::RunVerifierBatch(
    int start_step, int current_token_id,
    const std::vector<int>& draft_tokens) {
  std::vector<int32_t> verify_ids;
  verify_ids.push_back(current_token_id);
  for (int id : draft_tokens) {
    verify_ids.push_back(id);
  }

  auto start_embedder = absl::Now();
  LITERT_RETURN_IF_ERROR(main_embedder_.RunVerify(verify_ids));
  latency_stats_.decode_embedder_inference_latency_us +=
      absl::ToInt64Microseconds(absl::Now() - start_embedder);

  if (main_embedder_.HasPerLayerEmbeddings()) {
    auto start = absl::Now();
    LITERT_RETURN_IF_ERROR(main_embedder_.RunVerifyPerLayer(verify_ids));
    latency_stats_.decode_embedder_per_layer_inference_latency_us +=
        absl::ToInt64Microseconds(absl::Now() - start);
  }

  {
    LITERT_RETURN_IF_ERROR(
        main_rope_.SetVerifyPositions(start_step, verify_ids.size()));
    NPU_EXECUTOR_LOG(INFO) << "    [Verify] Input Token IDs: ["
                           << absl::StrJoin(verify_ids, ", ") << "]";

    LITERT_RETURN_IF_ERROR(main_mask_.SetVerifyInput(start_step, verify_ids));
  }

  LITERT_RETURN_IF_ERROR(
      main_rope_.RunVerify(ActiveContextGroup().verify_aux_signatures.rope));
  LITERT_RETURN_IF_ERROR(
      main_mask_.RunVerify(ActiveContextGroup().verify_aux_signatures.mask));

  LITERT_RETURN_IF_ERROR(text_decoder_compiled_model_.Run(
      ActiveContextGroup().verify_signature,
      ActiveContextGroup().text_decoder_inference_context.verify_input_buffers,
      ActiveContextGroup()
          .text_decoder_inference_context.verify_output_buffers));

  return absl::OkStatus();
}

absl::StatusOr<LlmLiteRtNpuCompiledModelExecutor::RejectionSamplingResult>
LlmLiteRtNpuCompiledModelExecutor::PerformRejectionSampling(
    const std::vector<int>& draft_tokens,
    const ::litert::TensorBuffer& verifier_logits_buffer) {
  int num_accepted = 0;
  int bonus_token_id = kInvalidTokenId;

  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type,
                          verifier_logits_buffer.TensorType());
  if (tensor_type.Layout().Dimensions().size() < 3) {
    return absl::InvalidArgumentError(
        "Logits tensor must have at least 3 dimensions.");
  }
  const int vocab_size = tensor_type.Layout().Dimensions()[2];
  if (vocab_size == 0) {
    return absl::InvalidArgumentError("Vocab size cannot be 0.");
  }
  size_t element_size = 0;
  switch (tensor_type.ElementType()) {
    case ::litert::ElementType::Float32:
      element_size = sizeof(float);
      break;
    case ::litert::ElementType::Int16:
      element_size = sizeof(int16_t);
      break;
    case ::litert::ElementType::Int8:
      element_size = sizeof(int8_t);
      break;
    default:
      return absl::UnimplementedError(
          "Unsupported logit type for element size.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          verifier_logits_buffer, ::litert::TensorBuffer::LockMode::kRead));
  const uint8_t* base_ptr = static_cast<const uint8_t*>(lock_and_addr.second);

  // Log all sampled tokens from the verifier for transparency.
  std::vector<int> all_verifier_sampled;
  all_verifier_sampled.reserve(draft_tokens.size() + 1);
  for (int i = 0; i < draft_tokens.size() + 1; ++i) {
    LITERT_ASSIGN_OR_RETURN(
        int sampled_token,
        SampleLogitsSliceFromLockedPtr(
            tensor_type.ElementType(), base_ptr, i, vocab_size, element_size,
            npu_config_.enable_neon_for_npu_greedy_sampling));
    all_verifier_sampled.push_back(sampled_token);
  }
  NPU_EXECUTOR_LOG(INFO) << "    [RS] Verifier Sampled Tokens: ["
                         << absl::StrJoin(all_verifier_sampled, ", ") << "]";

  for (int i = 0; i < draft_tokens.size(); ++i) {
    int sampled_verifier_token = all_verifier_sampled[i];

    NPU_EXECUTOR_LOG(INFO) << "    [RS] Step " << i << ": Drafter Token "
                           << draft_tokens[i] << " vs Verifier Sampled Token "
                           << sampled_verifier_token;

    if (sampled_verifier_token == draft_tokens[i]) {
      num_accepted++;
    } else {
      bonus_token_id = sampled_verifier_token;
      break;
    }
  }

  if (num_accepted == draft_tokens.size()) {
    bonus_token_id = all_verifier_sampled[num_accepted];
  }
  latency_stats_.mtp_num_draft_tokens += draft_tokens.size();
  latency_stats_.mtp_num_accepted_tokens += num_accepted;

  return RejectionSamplingResult{num_accepted, bonus_token_id};
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::CommitVerifiedKVCache(
    int start_step) {
  return main_cache_.CommitVerifiedKVCache(
      start_step, ActiveContextGroup().verify_aux_signatures.cache_update);
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::SetCurrentStep(int new_step) {
  const int max_step = processed_tokens_.TokenCount();
  if (new_step != (max_step - 1)) {
    return absl::InvalidArgumentError(
        "NPU executor's SetCurrentStep only supports rolling back one token at "
        "the end of decode.");
  }

  for (int i = new_step; i < current_step_; ++i) {
    if (processed_tokens_.GetTokenAtStep(i).empty()) {
      return absl::InvalidArgumentError(
          "SetCurrentStep does not currently support rolling back vision or "
          "audio tokens.");
    }
  }

  current_step_ = new_step;
  return absl::OkStatus();
};

absl::StatusOr<const ProcessedTokens*>
LlmLiteRtNpuCompiledModelExecutor::GetProcessedTokens() const {
  return &processed_tokens_;
}

absl::StatusOr<int> LlmLiteRtNpuCompiledModelExecutor::GetVocabSize() {
  LITERT_ASSIGN_OR_RETURN(
      auto logits_tensor_type,
      ActiveContextGroup()
          .text_decoder_inference_context
          .decode_output_buffers[TextDecoderSignatures::kDecodeLogitsOutput]
          .TensorType());
  const auto rank = logits_tensor_type.Layout().Dimensions().size();
  RET_CHECK(rank == 2 || rank == 3) << "Logits must be a 2D or 3D tensor.";
  return logits_tensor_type.Layout().Dimensions()[rank - 1];
}

const LatencyStats& LlmLiteRtNpuCompiledModelExecutor::GetLatencyStats() const {
  return latency_stats_;
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::Reset() {
  NPU_EXECUTOR_LOG(INFO) << "Custom NPU execution latency stats:\n"
                         << latency_stats_;
  current_step_ = 0;
  ran_decode_ = false;
  active_context_group_index_ = 0;
  LITERT_RETURN_IF_ERROR(processed_tokens_.RollBackToStep(0));
  latency_stats_ = {};
  last_verify_activations_.clear();
  pending_accepted_tokens_.clear();

  for (auto& group : context_groups_) {
    LITERT_RETURN_IF_ERROR(ClearKVCache(
        group.text_decoder_inference_context.decode_input_buffers));
    LITERT_RETURN_IF_ERROR(ClearKVCache(
        group.text_decoder_inference_context.prefill_input_buffers));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtNpuCompiledModelExecutor::CreateNewContext(
    std::optional<uint32_t> lora_id, RuntimeConfig runtime_config) const {
  std::unique_ptr<ProcessedContext> processed_context =
      std::make_unique<LlmProcessedContext>(
          lora_id,
          std::make_unique<LegacyMapState>(
              absl::flat_hash_map<std::string, ::litert::TensorBuffer>()));

  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::make_unique<RuntimeState>());
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtNpuCompiledModelExecutor::CloneContext() const {
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> kv_cache_buffers;
  for (const auto& [name, buffer] :
       ActiveContextGroup()
           .text_decoder_inference_context.prefill_input_buffers) {
    if (absl::StartsWith(name, kv_cache_k_root_name) ||
        absl::StartsWith(name, kv_cache_v_root_name) ||
        absl::StartsWith(name, kv_cache_c_root_name)) {
      LITERT_ASSIGN_OR_RETURN(auto buffer_copy, CopyTensorBuffer(env_, buffer));
      kv_cache_buffers[name] = std::move(buffer_copy);
    }
  }

  std::unique_ptr<ProcessedContext> processed_context =
      std::make_unique<LlmProcessedContext>(
          /*lora_id=*/std::nullopt,
          std::make_unique<LegacyMapState>(std::move(kv_cache_buffers)),
          processed_tokens_);

  RuntimeConfig runtime_config;
  runtime_config.sampler_params = sampler_params_;

  RuntimeState runtime_state;
  runtime_state.current_step = current_step_;

  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::make_unique<RuntimeState>(std::move(runtime_state)));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::RestoreContext(
    std::unique_ptr<LlmContext> context_data) {
  current_step_ = context_data->runtime_state().current_step;
  processed_tokens_ = context_data->processed_context().processed_tokens();
  if (context_data->runtime_config().sampler_params.has_value()) {
    sampler_params_ = *context_data->runtime_config().sampler_params;
  }

  if (current_step_ > 0) {
    auto& processed_ctx =
        static_cast<LlmProcessedContext&>(context_data->processed_context());
    auto* map_state =
        dynamic_cast<LegacyMapState*>(processed_ctx.state().get());
    RET_CHECK(map_state != nullptr)
        << "Expected LegacyMapState in RestoreContext";
    const auto& saved_kv_buffers = map_state->buffers();

    // 1. Determine the target active context group index.
    int target_group_idx = 0;
    for (size_t i = 0; i < sorted_supported_context_sizes_.size(); ++i) {
      if (sorted_supported_context_sizes_[i] >= current_step_) {
        target_group_idx = i;
        break;
      }
    }
    active_context_group_index_ = target_group_idx;
    auto& active_group = context_groups_[target_group_idx];

    // 2. Restore KV cache buffers into the target active context group.
    for (const auto& [name, saved_buffer] : saved_kv_buffers) {
      if (active_group.input_kv_cache_buffers.contains(name)) {
        auto& target_buffer = active_group.input_kv_cache_buffers[name];
        LITERT_RETURN_IF_ERROR(NpuKVCache::CopySingleKVCacheBuffer(
            saved_buffer, target_buffer, current_step_, kv_cache_init_value_));
      }
    }
  } else {
    for (auto& group : context_groups_) {
      LITERT_RETURN_IF_ERROR(ClearKVCache(
          group.text_decoder_inference_context.decode_input_buffers));
      LITERT_RETURN_IF_ERROR(ClearKVCache(
          group.text_decoder_inference_context.prefill_input_buffers));
    }
    active_context_group_index_ = 0;
  }

  // 3. Re-bind all modular sub-components to the restored active context group.
  auto& active_group = ActiveContextGroup();
  LITERT_RETURN_IF_ERROR(main_cache_.UpdateKVCacheBuffers(
      active_group.input_kv_cache_buffers,
      active_group.text_decoder_inference_context.prefill_output_buffers,
      active_group.text_decoder_inference_context.decode_output_buffers,
      active_group.text_decoder_inference_context.verify_output_buffers));
  LITERT_RETURN_IF_ERROR(main_mask_.UpdateOutputBuffers(
      active_group.text_decoder_inference_context.prefill_input_buffers,
      active_group.text_decoder_inference_context.decode_input_buffers,
      active_group.text_decoder_inference_context.verify_input_buffers));
  LITERT_RETURN_IF_ERROR(main_rope_.UpdateOutputBuffers(
      active_group.text_decoder_inference_context.prefill_input_buffers,
      active_group.text_decoder_inference_context.decode_input_buffers,
      active_group.text_decoder_inference_context.verify_input_buffers));
  LITERT_RETURN_IF_ERROR(main_embedder_.UpdateOutputBuffers(
      active_group.text_decoder_inference_context.prefill_input_buffers,
      active_group.text_decoder_inference_context.decode_input_buffers,
      active_group.text_decoder_inference_context.verify_input_buffers));
  if (drafter_context_.has_value()) {
    LITERT_RETURN_IF_ERROR(drafter_context_->UpdateKVCacheBuffers(
        active_group.input_kv_cache_buffers));
  }

  return absl::OkStatus();
}

absl::StatusOr<int>
LlmLiteRtNpuCompiledModelExecutor::DetermineMaxSequenceLength(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    const litert::Model& text_decoder_model) {
  int ans = 0;

  // (1) Check for the presence of the max_num_tokens in the LlmMetadata
  if (auto metadata_status = resources.GetLlmMetadata(); metadata_status.ok()) {
    const proto::LlmMetadata* metadata = *metadata_status;
    if (metadata && metadata->max_num_tokens() > 0) {
      ans = metadata->max_num_tokens();
    }
  }

  // (2) If not present fall back to iterate through all KV cache input buffers
  // of the text_decoder_model and get the maximum number.
  if (ans <= 0) {
    // We need to go through all KV cache layers because in sliding window
    // attention models various layers will have a smaller (ringbuffer) cache,
    // and instead we need to find the true global KV cache.
    // Once the "Executor metadata" design is implemented the information can
    // instead be taken from there.
    LITERT_ASSIGN_OR_RETURN(const int prefill_size,
                            DetectPrefillSize(text_decoder_model));
    LITERT_ASSIGN_OR_RETURN(const std::vector<int> context_sizes,
                            DetectSupportedContextSizes(text_decoder_model));
    const int ctx_size = context_sizes.empty() ? 0 : context_sizes.back();
    const ResolvedPrefillSignatures prefill_sigs =
        BuildResolvedPrefillSignatures(prefill_size, ctx_size);
    LITERT_ASSIGN_OR_RETURN(
        SimpleSignature prefill_signature,
        text_decoder_model.FindSignature(prefill_sigs.prefill));
    for (auto input_name : prefill_signature.InputNames()) {
      if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
          absl::StartsWith(input_name, kv_cache_v_root_name) ||
          absl::StartsWith(input_name, kv_cache_c_root_name)) {
        LITERT_ASSIGN_OR_RETURN(const litert::SimpleTensor& tensor,
                                prefill_signature.InputTensor(input_name));
        LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType type,
                                tensor.RankedTensorType());
        for (auto dim : type.Layout().Dimensions()) {
          ans = std::max(ans, dim);
        }
      }
    }
  }

  // (3) Check the passed in executor setting max num tokens field.
  int settings_max_num_tokens = executor_settings.GetMaxNumTokens();
  if (settings_max_num_tokens > 0) {
    if (ans > 0) {
      if (settings_max_num_tokens > ans) {
        ABSL_LOG(WARNING) << "Passed in max_num_tokens ("
                          << settings_max_num_tokens
                          << ") is larger than what the model supports (" << ans
                          << "). Using model limit.";
      } else {
        ans = settings_max_num_tokens;
      }
    } else {
      ans = settings_max_num_tokens;
    }
  }

  if (ans <= 0) {
    return absl::InternalError("Failed to determine max sequence length.");
  }

  return ans;
}

// static
absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
LlmLiteRtNpuCompiledModelExecutor::Create(
    const LlmExecutorSettings& executor_settings, ModelResources& resources,
    Environment& env) {
  bool enable_npu_debug_logging = false;
  auto npu_config_status = executor_settings.GetBackendConfig<NpuConfig>();
  if (npu_config_status.ok()) {
    enable_npu_debug_logging = npu_config_status->enable_npu_debug_logging;
  }

  LITERT_ASSIGN_OR_RETURN(
      const litert::Model* text_decoder_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  LITERT_ASSIGN_OR_RETURN(
      int max_sequence_length,
      DetermineMaxSequenceLength(executor_settings, resources,
                                 *text_decoder_model));

  // `DetermineMaxSequenceLength` resolves the effective limit by taking the
  // minimum of the user-requested limit (if > 0) and the model-supported limit.
  // If they differ (e.g. user limit is 0 or larger than what the model
  // supports), we override the settings. If the user requested a smaller limit,
  // it is respected and not overridden.
  LlmExecutorSettings mutable_settings = executor_settings;
  if (mutable_settings.GetMaxNumTokens() != max_sequence_length) {
    ABSL_LOG(WARNING) << "Overriding executor settings max_num_tokens ("
                      << mutable_settings.GetMaxNumTokens()
                      << ") with NPU model max sequence length: ("
                      << max_sequence_length << ")";
    mutable_settings.SetMaxNumTokens(max_sequence_length);
  }

  // Detect supported context sizes.
  LITERT_ASSIGN_OR_RETURN(std::vector<int> sorted_supported_context_sizes,
                          DetectSupportedContextSizes(*text_decoder_model));
  if (sorted_supported_context_sizes.empty()) {
    sorted_supported_context_sizes = {0};
  }

  // Initialize logits quantization parameters using the primary 'decode'
  // signature.
  LogitsQuantizationParams quantization_params = {.scale = 1.0f,
                                                  .zero_point = 0};
  const std::string default_decode_sig =
      sorted_supported_context_sizes.size() > 1
          ? absl::StrCat("decode_cache_",
                         sorted_supported_context_sizes.front())
          : std::string(kDecodeSignature);
  LITERT_ASSIGN_OR_RETURN(
      auto decode_signature,
      text_decoder_model->FindSignature(default_decode_sig));
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor,
                          decode_signature.OutputTensor(
                              TextDecoderSignatures::kDecodeLogitsOutput));
  if (logits_tensor.HasQuantization()) {
    auto q_params = logits_tensor.PerTensorQuantization();
    quantization_params.scale = q_params.scale;
    quantization_params.zero_point = static_cast<int32_t>(q_params.zero_point);
    ABSL_LOG_IF(INFO, enable_npu_debug_logging)
        << "Logits quantization params from '" << default_decode_sig
        << "' signature: scale=" << quantization_params.scale
        << " zero_point=" << quantization_params.zero_point;
  } else {
    ABSL_LOG_IF(WARNING, enable_npu_debug_logging)
        << "No quantization for logits in '" << default_decode_sig
        << "' signature (using default scale= " << quantization_params.scale
        << ", zero_point= " << quantization_params.zero_point << ").";
  }

  // Detect the prefill length the model was compiled with (e.g. 128 or 256) and
  // resolve all prefill-family signature names from it.
  LITERT_ASSIGN_OR_RETURN(const int prefill_size,
                          DetectPrefillSize(*text_decoder_model));
  const int max_context_size = sorted_supported_context_sizes.back();
  const ResolvedPrefillSignatures max_prefill_signatures =
      BuildResolvedPrefillSignatures(prefill_size, max_context_size);
  ABSL_LOG(INFO) << "Detected NPU prefill size: " << prefill_size
                 << " (signature \"" << max_prefill_signatures.prefill
                 << "\").";

  // For the lack of a better way to identify the model variants, we use the
  // presence of per-layer embeddings as the signal for Gemma3n.
  LITERT_ASSIGN_OR_RETURN(
      const bool has_per_layer_embeddings,
      HasPerLayerEmbedder(*text_decoder_model, max_prefill_signatures.prefill));

  int64_t kv_cache_init_value = GetKvCacheInitValue(resources);
  // If the model is fully AOT compiled for NPU, NPU accelerator is used
  // automatically.
  LITERT_ASSIGN_OR_RETURN(auto options,
                          CreateLiteRtNpuOptions(mutable_settings));
  LITERT_ASSIGN_OR_RETURN(
      CompiledModel text_decoder_compiled_model,
      CompiledModel::Create(env, text_decoder_model->Get(), options));

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, HWQuantParams> kv_quant_params;

  std::vector<ContextGroup> context_groups;
  context_groups.resize(sorted_supported_context_sizes.size());

  // =========================================================================
  // Phase 1: Allocate Largest Context Group (Owns All Master Hardware Buffers)
  // =========================================================================
  const int largest_group_idx = sorted_supported_context_sizes.size() - 1;
  {
    const ResolvedPrefillSignatures max_prefill_sigs =
        BuildResolvedPrefillSignatures(prefill_size, max_context_size);
    const std::string max_decode_sig =
        sorted_supported_context_sizes.size() > 1
            ? absl::StrCat("decode_cache_", max_context_size)
            : std::string(kDecodeSignature);
    const std::string max_verify_sig =
        sorted_supported_context_sizes.size() > 1
            ? absl::StrCat("verify_cache_", max_context_size)
            : std::string(TextDecoderSignatures::kVerify);

    absl::flat_hash_map<absl::string_view, TensorBuffer>
        master_kv_cache_buffers;
    absl::flat_hash_map<absl::string_view, TensorBuffer> group_prefill_in;
    absl::flat_hash_map<absl::string_view, TensorBuffer> group_decode_in;
    absl::flat_hash_map<absl::string_view, TensorBuffer> group_verify_in;
    absl::flat_hash_map<absl::string_view, TensorBuffer>
        group_prefill_out_slices;
    absl::flat_hash_map<absl::string_view, TensorBuffer>
        group_decode_out_slices;
    absl::flat_hash_map<absl::string_view, TensorBuffer>
        group_verify_out_slices;

    LITERT_RETURN_IF_ERROR(AllocateTextDecoderBuffers(
        env, text_decoder_model, text_decoder_compiled_model, max_prefill_sigs,
        max_decode_sig, max_verify_sig, group_prefill_in, group_decode_in,
        group_verify_in, master_kv_cache_buffers, group_prefill_out_slices,
        group_decode_out_slices, group_verify_out_slices, kv_quant_params,
        kv_cache_init_value));

    // Dynamic HostMemory workaround: ensure all KV buffers in the master group
    // are on NPU memory (e.g. layer 19 in Gemma3n or other layers unconnected
    // in prefill)
    for (auto& [name, buf] : master_kv_cache_buffers) {
      if (absl::StartsWith(name, kv_cache_k_root_name) ||
          absl::StartsWith(name, kv_cache_v_root_name) ||
          absl::StartsWith(name, kv_cache_c_root_name)) {
        auto buf_type = buf.BufferType();
        if (buf_type.HasValue() &&
            buf_type.Value() == ::litert::TensorBufferType::kHostMemory) {
          LITERT_ASSIGN_OR_RETURN(auto new_buf,
                                  text_decoder_compiled_model.CreateInputBuffer(
                                      max_decode_sig, name));
          LITERT_RETURN_IF_ERROR(
              FillKVCacheBuffer(new_buf, kv_cache_init_value));
          buf = std::move(new_buf);
        }
      }
    }

    LITERT_ASSIGN_OR_RETURN(
        auto max_inference_context,
        CreateTextDecoderInferenceContext(
            env, text_decoder_compiled_model, max_prefill_sigs, max_decode_sig,
            max_verify_sig, master_kv_cache_buffers, group_prefill_out_slices,
            group_decode_out_slices, group_verify_out_slices, group_prefill_in,
            group_decode_in, group_verify_in));

    ContextGroup group;
    group.context_size = max_context_size;
    group.prefill_signatures = max_prefill_sigs;
    group.decode_signature = max_decode_sig;
    group.verify_signature = max_verify_sig;
    group.input_kv_cache_buffers = std::move(master_kv_cache_buffers);
    group.text_decoder_inference_context = std::move(max_inference_context);
    context_groups[largest_group_idx] = std::move(group);
  }

  // =========================================================================
  // Phase 2: Create Aliased Context Groups for Smaller Context Sizes
  // =========================================================================
  const auto& master_kv_buffers =
      context_groups[largest_group_idx].input_kv_cache_buffers;

  for (size_t i = 0; i + 1 < sorted_supported_context_sizes.size(); ++i) {
    int ctx_size = sorted_supported_context_sizes[i];
    const ResolvedPrefillSignatures group_prefill_sigs =
        BuildResolvedPrefillSignatures(prefill_size, ctx_size);
    const std::string group_decode_sig =
        sorted_supported_context_sizes.size() > 1
            ? absl::StrCat("decode_cache_", ctx_size)
            : std::string(kDecodeSignature);
    const std::string group_verify_sig =
        sorted_supported_context_sizes.size() > 1
            ? absl::StrCat("verify_cache_", ctx_size)
            : std::string(TextDecoderSignatures::kVerify);

    LITERT_ASSIGN_OR_RETURN(
        auto decode_signature,
        text_decoder_model->FindSignature(group_decode_sig));

    absl::flat_hash_map<absl::string_view, TensorBuffer> aliased_kv_buffers;
    // Alias all KV cache tensors from the master buffer
    for (const auto& [name, master_buf] : master_kv_buffers) {
      auto input_tensor = decode_signature.InputTensor(name);
      if (!input_tensor.HasValue()) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(auto target_type,
                              input_tensor->RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto alias_buf,
                              CreateAliasBuffer(env, master_buf, target_type));
      aliased_kv_buffers[name] = std::move(alias_buf);
    }

    absl::flat_hash_map<absl::string_view, TensorBuffer> group_prefill_in;
    absl::flat_hash_map<absl::string_view, TensorBuffer> group_decode_in;
    absl::flat_hash_map<absl::string_view, TensorBuffer> group_verify_in;
    absl::flat_hash_map<absl::string_view, TensorBuffer>
        group_prefill_out_slices;
    absl::flat_hash_map<absl::string_view, TensorBuffer>
        group_decode_out_slices;
    absl::flat_hash_map<absl::string_view, TensorBuffer>
        group_verify_out_slices;

    LITERT_RETURN_IF_ERROR(AllocateTextDecoderBuffers(
        env, text_decoder_model, text_decoder_compiled_model,
        group_prefill_sigs, group_decode_sig, group_verify_sig,
        group_prefill_in, group_decode_in, group_verify_in, aliased_kv_buffers,
        group_prefill_out_slices, group_decode_out_slices,
        group_verify_out_slices, kv_quant_params, kv_cache_init_value));

    LITERT_ASSIGN_OR_RETURN(
        auto group_inference_context,
        CreateTextDecoderInferenceContext(
            env, text_decoder_compiled_model, group_prefill_sigs,
            group_decode_sig, group_verify_sig, aliased_kv_buffers,
            group_prefill_out_slices, group_decode_out_slices,
            group_verify_out_slices, group_prefill_in, group_decode_in,
            group_verify_in));

    ContextGroup group;
    group.context_size = ctx_size;
    group.prefill_signatures = group_prefill_sigs;
    group.decode_signature = group_decode_sig;
    group.verify_signature = group_verify_sig;
    group.input_kv_cache_buffers = std::move(aliased_kv_buffers);
    group.text_decoder_inference_context = std::move(group_inference_context);
    context_groups[i] = std::move(group);
  }

  if (!has_per_layer_embeddings) {
    // Gemma3 specific fix:
    //
    // TODO(b/416702118): Buffers kv_cache_{k,v}_25 have float element type for
    // the prefill signature but int16_t for the decode signature. Therefore,
    // unlike for the other KV cache tensors, we can not re-use the same tensor
    // during prefill and decode (because trying to register a tensor of element
    // type float for the decode signature that expects it in int16_t will
    // fail). Luckily these buffers are not used, so we can simply create new
    // ones to satisfy the compiled model run API.  We can remove this
    // workaround once we have a model that removes these buffers.
    //
    // We only need this for models with a single cache signature, so we do it
    // for the first context group only.
    auto& text_decoder_inference_context =
        context_groups[0].text_decoder_inference_context;
    if (text_decoder_inference_context.prefill_input_buffers.contains(
            cache_k31)) {
      // For models with 32 layers. Do nothing.
    } else if (text_decoder_inference_context.prefill_input_buffers.contains(
                   cache_k25)) {
      LITERT_ASSIGN_OR_RETURN(auto buffer_k,
                              text_decoder_compiled_model.CreateInputBuffer(
                                  kDecodeSignature, cache_k25));
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer_k, kv_cache_init_value));
      text_decoder_inference_context.decode_input_buffers[cache_k25] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(auto buffer_v,
                              text_decoder_compiled_model.CreateInputBuffer(
                                  kDecodeSignature, cache_v25));
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer_v, kv_cache_init_value));
      text_decoder_inference_context.decode_input_buffers[cache_v25] =
          std::move(buffer_v);
    } else if (text_decoder_inference_context.prefill_input_buffers.contains(
                   cache_k23)) {
      // Fast VLM model specific fix:
      LITERT_ASSIGN_OR_RETURN(auto buffer_k,
                              text_decoder_compiled_model.CreateInputBuffer(
                                  kDecodeSignature, cache_k23));
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer_k, kv_cache_init_value));
      text_decoder_inference_context.decode_input_buffers[cache_k23] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(auto buffer_v,
                              text_decoder_compiled_model.CreateInputBuffer(
                                  kDecodeSignature, cache_v23));
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer_v, kv_cache_init_value));
      text_decoder_inference_context.decode_input_buffers[cache_v23] =
          std::move(buffer_v);
    } else if (text_decoder_inference_context.prefill_input_buffers.contains(
                   cache_k17)) {
      // Tiny Gemma 270M specific fix:
      LITERT_ASSIGN_OR_RETURN(auto buffer_k,
                              text_decoder_compiled_model.CreateInputBuffer(
                                  kDecodeSignature, cache_k17));
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer_k, kv_cache_init_value));
      text_decoder_inference_context.decode_input_buffers[cache_k17] =
          std::move(buffer_k);
      LITERT_ASSIGN_OR_RETURN(auto buffer_v,
                              text_decoder_compiled_model.CreateInputBuffer(
                                  kDecodeSignature, cache_v17));
      LITERT_RETURN_IF_ERROR(FillKVCacheBuffer(buffer_v, kv_cache_init_value));
      text_decoder_inference_context.decode_input_buffers[cache_v17] =
          std::move(buffer_v);
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_lrt_model,
                          resources.GetTFLiteModel(ModelType::kTfLiteAux));

  LITERT_ASSIGN_OR_RETURN(auto npu_auxiliary_context,
                          CreateNpuAuxiliaryContext(
                              env, *npu_auxiliary_lrt_model, mutable_settings));

  for (auto& group : context_groups) {
    group.decode_aux_signatures = BuildResolvedDecodeAuxiliarySignatures(
        npu_auxiliary_context.npu_auxiliary_compiled_model, group.context_size);
    group.verify_aux_signatures = BuildResolvedVerifyAuxiliarySignatures(
        npu_auxiliary_context.npu_auxiliary_compiled_model, group.context_size);
  }

  MaskUpdateMethod mask_update_method = MaskUpdateMethod::kModel;
  KVCacheUpdateMethod cache_update_method = KVCacheUpdateMethod::kModel;
  if (npu_config_status.ok()) {
    if (npu_config_status->use_hw_masking_for_npu) {
      mask_update_method = MaskUpdateMethod::kWH;
    }
    if (npu_config_status->use_hw_cache_update_for_npu) {
      cache_update_method = KVCacheUpdateMethod::kWH;
    }
  }

  // We create the auxiliary components and start out with the smallest context
  // size.
  const auto& first_prefill_sigs = context_groups[0].prefill_signatures;
  const auto& first_decode_aux_sigs = context_groups[0].decode_aux_signatures;
  const auto& first_verify_aux_sigs = context_groups[0].verify_aux_signatures;
  auto& first_prefill_in =
      context_groups[0].text_decoder_inference_context.prefill_input_buffers;
  auto& first_decode_in =
      context_groups[0].text_decoder_inference_context.decode_input_buffers;
  auto& first_verify_in =
      context_groups[0].text_decoder_inference_context.verify_input_buffers;

  LITERT_ASSIGN_OR_RETURN(
      auto main_mask,
      NpuMask::Create(mask_update_method,
                      &npu_auxiliary_context.npu_auxiliary_compiled_model,
                      first_prefill_sigs.mask, first_decode_aux_sigs.mask,
                      first_verify_aux_sigs.mask, first_prefill_in,
                      first_decode_in, first_verify_in));

  LITERT_ASSIGN_OR_RETURN(
      auto main_rope,
      NpuRope::Create(&npu_auxiliary_context.npu_auxiliary_compiled_model,
                      first_prefill_sigs.rope, first_decode_aux_sigs.rope,
                      first_verify_aux_sigs.rope, first_prefill_in,
                      first_decode_in, first_verify_in));

  const bool has_sliding_window_attention =
      DetectIsSwa(context_groups[0].input_kv_cache_buffers);

  LITERT_ASSIGN_OR_RETURN(
      auto main_cache,
      NpuKVCache::Create(
          cache_update_method,
          &npu_auxiliary_context.npu_auxiliary_compiled_model,
          first_prefill_sigs.cache_update, first_decode_aux_sigs.cache_update,
          first_verify_aux_sigs.cache_update,
          context_groups[0].input_kv_cache_buffers,
          context_groups[0]
              .text_decoder_inference_context.prefill_output_buffers,
          context_groups[0]
              .text_decoder_inference_context.decode_output_buffers,
          context_groups[0]
              .text_decoder_inference_context.verify_output_buffers,
          kv_quant_params, has_sliding_window_attention, kv_cache_init_value));

  // Initialize NpuEmbedder (encapsulating all PLE parsing and embedding lookup
  // manager).
  LITERT_ASSIGN_OR_RETURN(
      auto main_embedder,
      NpuEmbedder::Create(env, resources, mutable_settings, first_prefill_sigs,
                          first_prefill_in, first_decode_in, first_verify_in,
                          has_per_layer_embeddings));

  // For now we only support one prefill length in the model.
  SortedPrefillSignatureMap prefill_runner_set;
  prefill_runner_set[first_prefill_sigs.size] = first_prefill_sigs.prefill;

  SpeculativeDecodingType speculative_decoding_type =
      SpeculativeDecodingType::kNone;
  std::optional<DrafterContext> drafter_context = std::nullopt;
  std::optional<DrafterAuxContext> drafter_aux_context = std::nullopt;

  if (mutable_settings.GetAdvancedSettings().has_value() &&
      mutable_settings.GetAdvancedSettings()->enable_speculative_decoding) {
    auto mtp_drafter_model =
        resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter);
    auto mtp_aux_model = resources.GetTFLiteModel(ModelType::kTfLiteMtpAux);

    if (mtp_drafter_model.ok() && mtp_aux_model.ok()) {
      if (!has_per_layer_embeddings) {
        return absl::InvalidArgumentError(
            "Speculative decoding is not supported for model without per layer "
            "embedding.");
      }
      LITERT_ASSIGN_OR_RETURN(
          drafter_context,
          DrafterContext::Create(
              env, **mtp_drafter_model,
              context_groups[0].input_kv_cache_buffers,
              context_groups[0]
                  .text_decoder_inference_context.decode_output_buffers
                      [TextDecoderSignatures::kLastLayerActivationsOutput]));
      MaskUpdateMethod mtp_mask_update_method = MaskUpdateMethod::kModel;
      if (npu_config_status.ok() && npu_config_status->use_hw_masking_for_npu) {
        mtp_mask_update_method = MaskUpdateMethod::kWH;
      }
      LITERT_ASSIGN_OR_RETURN(
          drafter_aux_context,
          DrafterAuxContext::Create(env, **mtp_aux_model,
                                    drafter_context->mtp_input_buffers,
                                    mtp_mask_update_method));
      speculative_decoding_type = SpeculativeDecodingType::kMTP;

      LITERT_RETURN_IF_ERROR(WarmupDrafterInference(
          drafter_context.value(), drafter_aux_context.value()));
    }
  }

  LITERT_RETURN_IF_ERROR(WarmupInference(
      text_decoder_compiled_model, context_groups,
      npu_auxiliary_context.npu_auxiliary_compiled_model, main_rope.Context(),
      main_mask.Context(), main_cache.Context()));

  return absl::WrapUnique(new LlmLiteRtNpuCompiledModelExecutor(
      mutable_settings, env, std::move(npu_auxiliary_context),
      std::move(text_decoder_compiled_model), std::move(context_groups),
      std::move(sorted_supported_context_sizes), std::move(prefill_runner_set),
      first_prefill_sigs, quantization_params, kv_cache_init_value,
      speculative_decoding_type, std::move(drafter_context),
      std::move(drafter_aux_context), std::move(main_embedder),
      std::move(main_rope), std::move(main_mask), std::move(main_cache)));
}

absl::Status LlmLiteRtNpuCompiledModelExecutor::ClearKVCache(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& buffers)
    const {
  return ClearKVCacheBuffers(buffers, kv_cache_init_value_);
}

}  // namespace litert::lm
