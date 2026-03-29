#include "runtime/executor/llm_litert_mtp_drafter.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

constexpr bool kEnableMtpDrafterLogs = false;

#define MTP_DRAFTER_LOG() \
  ABSL_LOG_IF(INFO, kEnableMtpDrafterLogs) << "MTP Drafter - "

constexpr absl::string_view kVerifySignatureRunner = "verify";

absl::StatusOr<std::unique_ptr<Sampler>> CreateGreedySampler(Environment& env,
                                                             int output_heads,
                                                             int sequence_size,
                                                             int vocab_size,
                                                             Backend backend) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  return CreateSampler(backend, output_heads, std::move(sampler_params),
                       &env, sequence_size, vocab_size);
}

absl::Status ConcatenateEmbeddingsAndActivations(
    const std::vector<float>& embedding_vector,
    TensorBuffer& projected_activations, TensorBuffer& output_activations) {
  size_t chunk_size_bytes = embedding_vector.size() * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(
      auto output_activations_lock_and_addr,
      TensorBufferScopedLock::Create(output_activations,
                                     TensorBuffer::LockMode::kWrite));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second),
         embedding_vector.data(), chunk_size_bytes);

  LITERT_ASSIGN_OR_RETURN(
      auto projected_activations_lock_and_addr,
      TensorBufferScopedLock::Create(projected_activations,
                                     TensorBuffer::LockMode::kRead));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second) +
             chunk_size_bytes,
         static_cast<const char*>(projected_activations_lock_and_addr.second),
         chunk_size_bytes);
  return absl::OkStatus();
}

absl::Status ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
    const std::vector<float>& embedding_vector,
    TensorBuffer& verifier_projected_activations,
    int last_verified_token_id_idx, TensorBuffer& output_activations) {
  size_t chunk_size_bytes = embedding_vector.size() * sizeof(float);

  LITERT_ASSIGN_OR_RETURN(
      auto output_activations_lock_and_addr,
      TensorBufferScopedLock::Create(output_activations,
                                     TensorBuffer::LockMode::kWrite));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second),
         embedding_vector.data(), chunk_size_bytes);

  LITERT_ASSIGN_OR_RETURN(
      auto verifier_projected_activations_lock_and_addr,
      TensorBufferScopedLock::Create(verifier_projected_activations,
                                     TensorBuffer::LockMode::kRead));
  // Grab the last valid activation from verifier output buffer with shape
  // [batch, draft_steps + 1, hidden_size].
  // Offset is as such the last verified token id's index multiplied by the
  // hiden_size in bytes.
  size_t offset = last_verified_token_id_idx * chunk_size_bytes;
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second) +
             chunk_size_bytes,
         static_cast<const char*>(
             verifier_projected_activations_lock_and_addr.second) +
             offset,
         chunk_size_bytes);
  return absl::OkStatus();
}

absl::StatusOr<int> GetVocabSizeFromLogitsTensor(TensorBuffer& logits_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  // logits tensor shape is [batch, seq, vocab].
  return logits_tensor_type.Layout().Dimensions()[2];
}

}  // namespace

LlmLiteRtMtpDrafter::~LlmLiteRtMtpDrafter() {
  ABSL_LOG(INFO) << "Num drafted tokens: " << num_drafted_tokens_;
  ABSL_LOG(INFO) << "Num verified tokens: " << num_verified_tokens_;
  if (num_drafted_tokens_ > 0) {
    ABSL_LOG(INFO) << "Success rate: "
                   << static_cast<double>(num_verified_tokens_) /
                          num_drafted_tokens_;
  }
}

absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>>
LlmLiteRtMtpDrafter::Create(Environment& env, ModelResources& resources,
                            const LlmExecutorSettings& executor_settings,
                            CompiledModel base_model,
                            EmbeddingLookupManager& embedding_manager,
                            EmbeddingLookupManager& ple_manager) {
  std::optional<std::string> backend_constraint =
      resources.GetTFLiteModelBackendConstraint(ModelType::kTfLiteMtpDrafter);
  if (backend_constraint.has_value()) {
    if (absl::EqualsIgnoreCase(backend_constraint.value(), "cpu")) {
      if (executor_settings.GetBackend() != Backend::CPU) {
        return absl::InvalidArgumentError(
            "Backend constraint for MTP drafter is CPU, but executor backend "
            "is not CPU.");
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Backend constraint ", backend_constraint.value(),
                       " for MTP drafter is not supported."));
    }
  }

  ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, ActivationDataType::FLOAT32,
                               /*signatures=*/std::nullopt,
                               /*cache_suffix=*/".mtp_drafter"));
  ASSIGN_OR_RETURN(auto model,
                   resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter));
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, model->Get(), compilation_options));

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_output_buffers;
  std::vector<std::string> kv_cache_input_names;
  {
    LITERT_ASSIGN_OR_RETURN(SimpleSignature signature,
                            compiled_model.GetSignature(/*signature_index=*/0));
    for (absl::string_view input_name : signature.InputNames()) {
      if (absl::StrContains(input_name, "kv")) {
        kv_cache_input_names.emplace_back(input_name);
        continue;
      }

      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          compiled_model.CreateInputBuffer(signature.Key(), input_name));
      mtp_drafter_input_buffers[input_name] = std::move(input_buffer);
    }

    for (absl::string_view output_name : signature.OutputNames()) {
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          compiled_model.CreateOutputBuffer(signature.Key(), output_name));
      mtp_drafter_output_buffers[output_name] = std::move(output_buffer);
    }
  }

  LITERT_ASSIGN_OR_RETURN(SimpleSignature verify_signature,
                          base_model.FindSignature(kVerifySignatureRunner));
  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_output_buffers;
  int num_draft_steps;
  {
    for (absl::string_view input_name : verify_signature.InputNames()) {
      if (absl::StrContains(input_name, "kv")) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          base_model.CreateInputBuffer(verify_signature.Key(), input_name));
      verifier_input_buffers[input_name] = std::move(input_buffer);
    }
    for (absl::string_view output_name : verify_signature.OutputNames()) {
      if (absl::StrContains(output_name, "kv")) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          base_model.CreateOutputBuffer(verify_signature.Key(), output_name));
      verifier_output_buffers[output_name] = std::move(output_buffer);
    }

    LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                            verify_signature.InputTensorType("input_pos"));
    // Expecred shape: [T = G + 1] where G is the number of draft steps
    const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
    num_draft_steps = input_pos_dims[0] - 1;
  }

  LITERT_ASSIGN_OR_RETURN(
      int vocab_size,
      GetVocabSizeFromLogitsTensor(verifier_output_buffers["logits"]));
  ASSIGN_OR_RETURN(
      auto drafter_sampler,
      CreateGreedySampler(env, /*output_heads=*/1, /*sequence_size=*/1,
                          vocab_size, executor_settings.GetBackend()));
  ASSIGN_OR_RETURN(
      auto verifier_sampler,
      CreateGreedySampler(env, /*output_heads=*/1,
                          /*sequence_size=*/num_draft_steps + 1, vocab_size,
                          executor_settings.GetBackend()));

  return absl::WrapUnique(new LlmLiteRtMtpDrafter(
      std::move(compiled_model), std::move(base_model),
      std::move(verify_signature), embedding_manager, ple_manager,
      std::move(drafter_sampler), std::move(verifier_sampler),
      std::move(kv_cache_input_names), std::move(mtp_drafter_input_buffers),
      std::move(mtp_drafter_output_buffers), std::move(verifier_input_buffers),
      std::move(verifier_output_buffers), num_draft_steps));
}

absl::StatusOr<std::vector<std::vector<int>>> LlmLiteRtMtpDrafter::Draft(
    int position, int token_id, std::optional<TensorBuffer> activations,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        output_kv_cache_buffers) {
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_input_buffers;
  for (const auto& [input_name, input_buffer] : drafter_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    mtp_drafter_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (const auto& kv_cache_input_name : kv_cache_input_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto kv_cache_buffer_dup,
        output_kv_cache_buffers[kv_cache_input_name].Duplicate());
    mtp_drafter_input_buffers[kv_cache_input_name] =
        std::move(kv_cache_buffer_dup);
  }
  LITERT_RETURN_IF_ERROR(mtp_drafter_input_buffers["input_pos"].Write<int32_t>(
      absl::MakeSpan(&position, 1)));
  RETURN_IF_ERROR(InitializeAttentionMask(mtp_drafter_input_buffers["mask"],
                                          /*use_fp16_precision=*/false));
  RETURN_IF_ERROR(FillAttentionMask(mtp_drafter_input_buffers["mask"],
                                    /*start_step=*/position,
                                    /*steps=*/1));
  if (mtp_drafter_input_buffers.contains("param_tensor")) {
    RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        mtp_drafter_input_buffers["param_tensor"], position,
        /*update_length=*/1));
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_output_buffers;
  for (const auto& [output_name, output_buffer] : drafter_output_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    LITERT_RETURN_IF_ERROR(output_buffer_dup.ClearEvent());
    mtp_drafter_output_buffers[output_name] = std::move(output_buffer_dup);
  }

  std::vector<int> drafted_tokens;
  drafted_tokens.reserve(num_draft_steps_);
  LITERT_ASSIGN_OR_RETURN(TensorBuffer id_tensor,
                          CreateTensorBuffer<int32_t>({1}));
  int last_drafted_token_id = token_id;
  std::vector<float> embedding_vector;
  TensorBuffer* activations_ptr =
      activations.has_value() ? &activations.value() : nullptr;
  for (int i = 0; i < num_draft_steps_; ++i) {
    // Concat and lookup embeddings with previous activations.
    // Embedding lookup has shape [B = 1, T = 1, D = 1536]
    // Drafter output activation has shape [B = 1, T = 1, D = 1536]
    // Concatenated embedding + activation has shape [B = 1, T = 1, D = 3072]
    TensorBuffer* drafter_activations_buffer =
        &mtp_drafter_input_buffers["activations"];
    RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id,
                                                    embedding_vector));
    if (activations_ptr) {
      RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(
          embedding_vector, *activations_ptr, *drafter_activations_buffer));
    } else {
      RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
          embedding_vector, verifier_output_buffers_["activations"],
          last_verified_token_id_idx_, *drafter_activations_buffer));
    }

    LITERT_RETURN_IF_ERROR(mtp_drafter_model_.Run(mtp_drafter_input_buffers,
                                                  mtp_drafter_output_buffers));

    RETURN_IF_ERROR(drafter_sampler_->SampleToIdAndScoreBuffer(
        mtp_drafter_output_buffers["logits"], id_tensor,
        /*scores_tensor=*/nullptr));
    LITERT_ASSIGN_OR_RETURN(auto id_vector,
                            CopyFromTensorBuffer<int32_t>(id_tensor));
    RET_CHECK_EQ(id_vector.size(), 1);
    drafted_tokens.push_back(id_vector[0]);

    last_drafted_token_id = id_vector[0];
    activations_ptr = &mtp_drafter_output_buffers["projected_activations"];
  }

  // verify the drafter tokens (i.e., run decode with size num_draft_steps_ + 1)
  {
    LITERT_ASSIGN_OR_RETURN(
        auto verifier_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(verifier_input_buffers_["input_pos"],
                                       TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(verifier_input_pos_lock_and_addr.second);
    for (int i = 0; i < num_draft_steps_ + 1; ++i) {
      *prefill_input_pos_ptr++ = position + i;
    }
  }

  RETURN_IF_ERROR(InitializeAttentionMask(verifier_input_buffers_["mask"],
                                          /*use_fp16_precision=*/false));
  RETURN_IF_ERROR(FillAttentionMask(verifier_input_buffers_["mask"],
                                    /*start_step=*/position,
                                    /*steps=*/num_draft_steps_ + 1));

  std::vector<int> drafted_tokens_with_input_token;
  drafted_tokens_with_input_token.reserve(num_draft_steps_ + 1);
  drafted_tokens_with_input_token.push_back(token_id);
  drafted_tokens_with_input_token.insert(drafted_tokens_with_input_token.end(),
                                         drafted_tokens.begin(),
                                         drafted_tokens.end());
  RETURN_IF_ERROR(embedding_manager_.LookupPrefill(
      drafted_tokens_with_input_token, &verifier_input_buffers_["embeddings"],
      /*offset=*/0));
  RETURN_IF_ERROR(ple_manager_.LookupPrefill(
      drafted_tokens_with_input_token,
      &verifier_input_buffers_["per_layer_embeddings"],
      /*offset=*/0));

  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_input_buffers;
  for (const auto& [input_name, input_buffer] : verifier_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    verifier_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  for (auto& [input_name, input_buffer] : input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    verifier_input_buffers[input_name] = std::move(input_buffer_dup);
  }
  if (verifier_input_buffers.contains("param_tensor")) {
    RETURN_IF_ERROR(
        FillSingleBufferCacheParamTensor(verifier_input_buffers["param_tensor"],
                                         position, num_draft_steps_ + 1));
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_output_buffers;
  for (const auto& [output_name, output_buffer] : verifier_output_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    LITERT_RETURN_IF_ERROR(output_buffer_dup.ClearEvent());
    verifier_output_buffers[output_name] = std::move(output_buffer_dup);
  }
  for (auto& [output_name, output_buffer] : output_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    LITERT_RETURN_IF_ERROR(output_buffer_dup.ClearEvent());
    verifier_output_buffers[output_name] = std::move(output_buffer_dup);
  }

  LITERT_RETURN_IF_ERROR(base_model_.Run(
      kVerifySignatureRunner, verifier_input_buffers, verifier_output_buffers));

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer verifier_id_tensor,
      CreateTensorBuffer<int32_t>({1, num_draft_steps_ + 1}));
  RETURN_IF_ERROR(verifier_sampler_->SampleToIdAndScoreBuffer(
      verifier_output_buffers["logits"], verifier_id_tensor,
      /*scores_tensor=*/nullptr));
  LITERT_ASSIGN_OR_RETURN(auto verifier_id_vector,
                          CopyFromTensorBuffer<int32_t>(verifier_id_tensor));
  RET_CHECK_EQ(verifier_id_vector.size(), num_draft_steps_ + 1);

  int num_correct_tokens = 0;
  int bonus_token = -1;
  for (int i = 0; i < num_draft_steps_; ++i) {
    last_verified_token_id_idx_ = i;
    if (verifier_id_vector[i] != drafted_tokens[i]) {
      bonus_token = verifier_id_vector[i];
      break;
    }
    num_correct_tokens++;
  }
  if (bonus_token == -1) {
    last_verified_token_id_idx_ = num_draft_steps_;
    bonus_token = verifier_id_vector[num_draft_steps_];
  }

  MTP_DRAFTER_LOG() << "drafted_tokens: "
                    << absl::StrJoin(drafted_tokens, ", ");
  MTP_DRAFTER_LOG() << "bonus_token: " << bonus_token;
  MTP_DRAFTER_LOG() << "num_correct_tokens: " << num_correct_tokens;

  // The first token comes from the decode output and is always correct.
  drafted_tokens.resize(num_correct_tokens);
  drafted_tokens.push_back(bonus_token);
  num_drafted_tokens_ += num_draft_steps_;
  num_verified_tokens_ += num_correct_tokens;

  MTP_DRAFTER_LOG() << "drafter output: "
                    << absl::StrJoin(drafted_tokens, ", ");
  MTP_DRAFTER_LOG() << "--------------------------------------------------";

  return std::vector<std::vector<int>>{std::move(drafted_tokens)};
}

}  // namespace litert::lm
