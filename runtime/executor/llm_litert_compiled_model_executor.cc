#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/litert/c/litert_common.h"  // from @litert
#include "litert/litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/litert/cc/litert_environment.h"  // from @litert
#include "litert/litert/cc/litert_expected.h"  // from @litert
#include "litert/litert/cc/litert_model.h"  // from @litert
#include "litert/litert/cc/litert_options.h"  // from @litert
#include "litert/litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/litert/runtime/accelerators/gpu/accelerator_options.h"  // from @litert
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

using ::absl::Span;
using ::litert::Expected;
using ::litert::Model;
using ::litert::TensorBuffer;
using ::litert::ml_drift::GpuOptions;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr char kPrefillSignatureRunner[] = "prefill";
constexpr char kDecodeSignatureRunner[] = "decode";

absl::Status GetCacheRootNames(std::vector<absl::string_view> input_names,
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
    }
  }
  return absl::FailedPreconditionError("No KV cache inputs found.");
}

bool IsCalculationPrecisionF16() { return true; }

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutor::Prefill(
    const Inputs& inputs, const litert::lm::PrefillQueryParams& query_params) {
  LITERT_ASSIGN_OR_RETURN_ABSL(auto tensor_type,
                               inputs.text_input.token_ids.TensorType());
  // Only accept batch size 1 for now.
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto ids, ReferTensorBufferAsSpan<int32_t>(inputs.text_input.token_ids));

  ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                         prefill_signature_map_, ids.size()));
  for (const auto& [prefill_signature, prefill_length] : work_groups) {
    RETURN_IF_ERROR(PrefillInternal(prefill_signature,
                                    ids.subspan(/*pos=*/0, prefill_length)));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::PrefillInternal(
    absl::string_view prefill_signature, Span<const int> ids) {
  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_buffer =
        prefill_input_buffers_[signatures_.input_tokens];
    LITERT_ASSIGN_OR_RETURN_ABSL(auto prefill_input_size,
                                 prefill_input_buffer.PackedSize());
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto prefill_input_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(prefill_input_buffer));
    auto* prefill_input_ptr =
        static_cast<int32_t*>(prefill_input_lock_and_addr.second);

    auto& prefill_input_pos =
        prefill_input_buffers_[signatures_.input_positions];
    LITERT_ASSIGN_OR_RETURN_ABSL(auto prefill_input_pos_size,
                                 prefill_input_pos.PackedSize());
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto prefill_input_pos_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(prefill_input_pos));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);
    bool has_input_attn_mask = signatures_.input_attn_mask.has_value();

    memset(prefill_input_ptr, 0, prefill_input_size);
    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    if (has_input_attn_mask) {
      RET_CHECK(signatures_.input_attn_mask_data_type.has_value())
          << "Attention mask data type is not provided.";
      RETURN_IF_ERROR(InitializeAttentionMask(
          prefill_input_buffers_[signatures_.input_attn_mask.value()],
          signatures_.input_attn_mask_data_type.value(),
          IsCalculationPrecisionF16()));
    }
    // We will not fill the last token of the current input into the interpreter
    // now. It will be stored in next_input_token_id_ and used in the next
    // prefill or decode.
    int start_step = current_step_;
    for (int i = 0, input_idx = 0; i < ids.size() - 1;
         input_idx++, current_step_++) {
      if (next_input_token_id_ != -1) {
        // Use next_input_token_id_ if it is valid.
        // Currently we use -1 to indicate that next_input_token_id_ is invalid.
        prefill_input_ptr[input_idx] = next_input_token_id_;
        // next_input_token_id_ should only be used once at the beginning of the
        // loop.
        next_input_token_id_ = -1;
      } else {
        prefill_input_ptr[input_idx] = ids[i];
        // Only increase i if we used the token inside ids.
        i++;
      }
      prefill_input_pos_ptr[input_idx] = current_step_;
    }
    if (has_input_attn_mask) {
      RETURN_IF_ERROR(FillAttentionMask(
          prefill_input_buffers_[signatures_.input_attn_mask.value()],
          start_step,
          /*steps=*/current_step_ - start_step,
          signatures_.input_attn_mask_data_type.value()));
    }
  }
  next_input_token_id_ = ids[ids.size() - 1];

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers;
  for (const auto& [input_name, input_buffer] : prefill_input_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    prefill_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    prefill_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_output_buffers;
  for (const auto& [output_name, output_buffer] : prefill_input_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    prefill_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    prefill_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }

  auto res = compiled_model_.Run(prefill_signature, prefill_input_buffers,
                                 prefill_output_buffers);
  RET_CHECK(res) << "Failed to run compiled model." << res.Error().Message();
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    ::litert::TensorBuffer& output_tokens) {
  if (!decoded_logits_) {
    // When external output handler not provided or sampling not initialized,
    // use CPU sampling. Note that this will introduce extra overhead for
    // downloading the logits from GPU to CPU.
    // Provide empty input to use the next_input_token_id_.
    ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
    LITERT_ASSIGN_OR_RETURN_ABSL(decoded_logits_,
                                 CreateTensorBuffer<float>({1, 1, vocab_size}));
  }
  RETURN_IF_ERROR(Decode(Inputs(), decoded_logits_));
  LITERT_ASSIGN_OR_RETURN_ABSL(auto logits,
                               ReferTensorBufferAsSpan<float>(decoded_logits_));
  ASSIGN_OR_RETURN(std::vector<int> output_ids, SampleLogits(logits));
  next_input_token_id_ = output_ids[0];
  return ToAbslStatus(output_tokens.Write(absl::MakeConstSpan(output_ids)));
}

absl::Status LlmLiteRtCompiledModelExecutor::Decode(
    const Inputs& inputs, ::litert::TensorBuffer& output_logits) {
  int id = next_input_token_id_;

  auto input_tensor_size = inputs.text_input.token_ids.PackedSize();
  if (input_tensor_size && *input_tensor_size != 0) {
    // Input token ids provided, so use it regardless of whether next input
    // token id is set. Only accept batch size 1 and a single token for now.
    RET_CHECK_EQ(*input_tensor_size, 1 * sizeof(int32_t));
    LITERT_ASSIGN_OR_RETURN_ABSL(auto ids, ReferTensorBufferAsSpan<int32_t>(
                                               inputs.text_input.token_ids));
    id = ids[0];
  }
  if (id == -1) {
    return absl::InvalidArgumentError("No id available to be decoded.");
  }

  // Invalidate the previous next_input_token_id_, regardless of whether it is
  // used.
  next_input_token_id_ = -1;

  {
    // Fill the input buffers with scoped locks.
    auto& decode_input_buffer = decode_input_buffers_[signatures_.input_tokens];
    auto decode_input_lock_and_addr =
        ::litert::TensorBufferScopedLock::Create(decode_input_buffer);
    RET_CHECK(decode_input_lock_and_addr)
        << "Failed to lock decode input buffer.";
    int32_t* decode_input_ptr =
        static_cast<int32_t*>(decode_input_lock_and_addr->second);
    auto& decode_input_pos_buffer =
        decode_input_buffers_[signatures_.input_positions];
    auto decode_input_pos_lock_and_addr =
        ::litert::TensorBufferScopedLock::Create(decode_input_pos_buffer);
    RET_CHECK(decode_input_pos_lock_and_addr)
        << "Failed to lock decode input position buffer.";
    auto* decode_input_pos_ptr =
        static_cast<int32_t*>(decode_input_pos_lock_and_addr->second);
    bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
    decode_input_ptr[0] = id;
    if (has_input_attn_mask) {
      RET_CHECK(signatures_.input_attn_mask_data_type.has_value())
          << "Attention mask data type is not provided.";
      RETURN_IF_ERROR(InitializeAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask.value()],
          signatures_.input_attn_mask_data_type.value(),
          IsCalculationPrecisionF16()));
      RETURN_IF_ERROR(FillAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask.value()],
          current_step_, /*steps=*/1,
          signatures_.input_attn_mask_data_type.value()));
    }
    decode_input_pos_ptr[0] = current_step_;
  }

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    decode_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  for (const auto& [input_name, input_buffer] : *input_kv_cache_buffers_) {
    auto duplicated_input_buffer = input_buffer.Duplicate();
    RET_CHECK(duplicated_input_buffer) << "Failed to duplicate input buffer.";
    decode_input_buffers[input_name] = std::move(*duplicated_input_buffer);
  }
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    auto duplicated_output_buffer = output_name == signatures_.output_logits
                                        ? output_logits.Duplicate()
                                        : output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }
  for (const auto& [output_name, output_buffer] : *output_kv_cache_buffers_) {
    auto duplicated_output_buffer = output_buffer.Duplicate();
    RET_CHECK(duplicated_output_buffer) << "Failed to duplicate output buffer.";
    decode_output_buffers[output_name] = std::move(*duplicated_output_buffer);
  }

  auto res = compiled_model_.Run(kDecodeSignatureRunner, decode_input_buffers,
                                 decode_output_buffers);
  RET_CHECK(res) << "Failed to run compiled model: " << res.Error().Message();
  std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);

  ++current_step_;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmLiteRtCompiledModelExecutor::SampleLogits(
    Span<const float> logits) {
  if (cpu_sampler_ == nullptr) {
    ASSIGN_OR_RETURN(cpu_sampler_,
                     TopPSampler::Create(
                         /*k=*/1, /*p=*/0.0f, /*temperature=*/1.0f,
                         decoded_logits_.TensorType()->Layout().Dimensions()[0],
                         /*seed=*/0));
  }
  ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  LITERT_ASSIGN_OR_RETURN_ABSL(auto logits_tensor,
                               CreateTensorBuffer<float>({1, vocab_size}));
  logits_tensor.Write(absl::MakeConstSpan(logits.data(), logits.size()));

  std::vector<int> ids_vector(output_batch_size_);
  // Construct a tensor buffer with the shape of [output_batch_size_] and
  // populate it with the ids_vector. This tensor buffer will be used as the
  // output of the sampler.
  auto ids_tensor = litert::lm::CopyToTensorBuffer<int>(
      absl::MakeConstSpan(ids_vector), {output_batch_size_});
  RETURN_IF_ERROR(cpu_sampler_->SampleToIdAndScoreBuffer(
      logits_tensor, *ids_tensor, /*scores_tensor=*/nullptr));
  auto ids = litert::lm::CopyFromTensorBuffer<int>(*ids_tensor);
  return *ids;
}

absl::Status LlmLiteRtCompiledModelExecutor::Reset() {
  current_step_ = 0;
  next_input_token_id_ = -1;
  processed_tokens_.clear();
  cpu_sampler_.reset();
  return absl::OkStatus();
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutor::GetVocabSize() {
  if (!decode_output_buffers_.contains(signatures_.output_logits)) {
    return absl::NotFoundError("Output logits info not found.");
  }

  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto logits_tensor_type,
      decode_output_buffers_[signatures_.output_logits].TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  return logits_tensor_type.Layout().Dimensions()[2];
}

// static
// Creates a LlmLiteRtCompiledModelExecutor from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutor>>
LlmLiteRtCompiledModelExecutor::Create(const LlmExecutorConfig& executor_config,
                                       ::litert::Model& litert_model) {
  // For the LlmLiteRtCompiledModelExecutor, ML_DRIFT backend is used by
  // default.
  // TODO(b/405424188): - Add support for NPU backends.
  auto compilation_options = ::litert::Options::Create();
  switch (executor_config.GetBackend()) {
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      Expected<GpuOptions> gpu_compilation_options = GpuOptions::Create();
      gpu_compilation_options->EnableConstantTensorSharing(true);
      gpu_compilation_options->EnableInfiniteFloatCapping(true);
      gpu_compilation_options->EnableAllowSrcQuantizedFcConvOps(true);
      gpu_compilation_options->SetDelegatePrecision(
          LiteRtDelegatePrecision::kLiteRtDelegatePrecisionFp16);
      compilation_options->AddOpaqueOptions(
          std::move(*gpu_compilation_options));
      compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
      break;
    }
    case Backend::CPU:
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", executor_config.GetBackend()));
  }

  auto lrt_env = ::litert::Environment::Create({});
  if (!lrt_env) {
    return absl::InternalError(absl::StrCat(
        "Failed to create litert environment: ", lrt_env.Error().Message()));
  }

  if (!litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
  }
  auto compiled_model = ::litert::CompiledModel::Create(
      *lrt_env, litert_model, std::move(*compilation_options));
  if (!compiled_model) {
    return absl::InternalError(absl::StrCat("Failed to create compiled model: ",
                                            compiled_model.Error().Message()));
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_kv_cache_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_kv_cache_buffers;

  absl::string_view prefill_signature_key = "";
  for (int i = 0; i < litert_model.GetNumSignatures(); ++i) {
    absl::string_view key = litert_model.GetSignature(i)->Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }
  auto prefill_signature = litert_model.FindSignature(prefill_signature_key);
  RET_CHECK(prefill_signature) << "Prefill signature not found.";
  std::string kv_cache_k_root_name;
  std::string kv_cache_v_root_name;
  RETURN_IF_ERROR(GetCacheRootNames(prefill_signature->InputNames(),
                                    kv_cache_k_root_name,
                                    kv_cache_v_root_name));

  for (auto input_name : prefill_signature->InputNames()) {
    auto input_buffer =
        compiled_model->CreateInputBuffer(prefill_signature_key, input_name);
    if (!input_buffer) {
      return absl::InternalError(
          absl::StrCat("Failed to create prefill input buffer for ", "'",
                       input_name, "' : ", input_buffer.Error().Message()));
    }
    if (absl::StartsWith(input_name, kv_cache_k_root_name) ||
        absl::StartsWith(input_name, kv_cache_v_root_name)) {
      input_kv_cache_buffers[input_name] = std::move(*input_buffer);
    } else {
      prefill_input_buffers[input_name] = std::move(*input_buffer);
    }
  }
  for (auto output_name : prefill_signature->OutputNames()) {
    auto output_buffer =
        compiled_model->CreateOutputBuffer(prefill_signature_key, output_name);
    if (!output_buffer) {
      return absl::InternalError(
          absl::StrCat("Failed to create prefill output buffer for ", "'",
                       output_name, "' : ", output_buffer.Error().Message()));
    }
    if (absl::StartsWith(output_name, kv_cache_k_root_name) ||
        absl::StartsWith(output_name, kv_cache_v_root_name)) {
      output_kv_cache_buffers[output_name] = std::move(*output_buffer);
    } else {
      prefill_output_buffers[output_name] = std::move(*output_buffer);
    }
  }

  auto decode_signature = litert_model.FindSignature(kDecodeSignatureRunner);
  for (auto input_name : decode_signature->InputNames()) {
    if (!absl::StartsWith(input_name, kv_cache_k_root_name) &&
        !absl::StartsWith(input_name, kv_cache_v_root_name)) {
      auto input_buffer =
          compiled_model->CreateInputBuffer(kDecodeSignatureRunner, input_name);
      if (!input_buffer) {
        return absl::InternalError(
            absl::StrCat("Failed to create decode input buffer for ", "'",
                         input_name, "' : ", input_buffer.Error().Message()));
      }
      decode_input_buffers[input_name] = std::move(*input_buffer);
    }
  }
  for (auto output_name : decode_signature->OutputNames()) {
    if (!absl::StartsWith(output_name, kv_cache_k_root_name) &&
        !absl::StartsWith(output_name, kv_cache_v_root_name)) {
      auto output_buffer = compiled_model->CreateOutputBuffer(
          kDecodeSignatureRunner, output_name);
      if (!output_buffer) {
        return absl::InternalError(
            absl::StrCat("Failed to create decode output buffer for ", "'",
                         output_name, "' : ", output_buffer.Error().Message()));
      }
      decode_output_buffers[output_name] = std::move(*output_buffer);
    }
  }

  ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature->InputNames(),
                                             decode_signature->OutputNames()));

  auto output_logits_buffer =
      decode_output_buffers[signatures.output_logits].Duplicate();
  RET_CHECK(output_logits_buffer)
      << "Failed to duplicate output logits buffer.";
  RET_CHECK(output_logits_buffer->TensorType()->Layout().Dimensions().size() ==
            3)
      << "Output logits must be (batch, seq, vocab)";
  RET_CHECK(output_logits_buffer->TensorType()->Layout().Dimensions()[0] == 1)
      << "Only support batch size 1 for now.";
  int batch_size = output_logits_buffer->TensorType()->Layout().Dimensions()[0];

  ASSIGN_OR_RETURN(auto prefill_runner_set,
                   GetPrefillRunnerSetFromModel(
                       litert_model, kPrefillSignatureRunner,
                       /*input_tokens_name=*/signatures.input_tokens));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  return absl::WrapUnique(new LlmLiteRtCompiledModelExecutor(
      std::move(*lrt_env), std::move(litert_model), std::move(*compiled_model),
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(input_kv_cache_buffers), std::move(output_kv_cache_buffers),
      std::move(prefill_runner_set), signatures, batch_size));
}

}  // namespace litert::lm
