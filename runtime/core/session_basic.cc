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

#include "runtime/core/session_basic.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/pipeline.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/components/lora_manager.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

using TaskController = Engine::Session::TaskController;

}

absl::flat_hash_set<LlmExecutor*>* SessionBasic::occupied_executors_ =
    new absl::flat_hash_set<LlmExecutor*>();
ABSL_CONST_INIT absl::Mutex SessionBasic::occupied_executors_mu_(
    absl::kConstInit);

// static
absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    LlmExecutor* executor, Tokenizer* tokenizer,
    VisionExecutor* vision_executor, AudioExecutor* audio_executor,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* worker_thread_pool) {
  // Check if the session already exists.
  absl::MutexLock lock(occupied_executors_mu_);  // NOLINT
  if (occupied_executors_->contains(executor)) {
    return absl::FailedPreconditionError(
        "A session already exists. Only one session is supported at a time. "
        "Please delete the existing session before creating a new one.");
  }
  auto sampler_backend = session_config.GetSamplerBackend();
  std::unique_ptr<Sampler> sampler;
  // If use CPU sampling, we create it here; For GPU sampling, we let executor
  // create it internally.
  if (sampler_backend == Backend::CPU) {
    ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(sampler_backend, session_config.GetNumOutputCandidates(),
                      session_config.GetSamplerParams()));
  } else if (sampler_backend != Backend::GPU &&
             sampler_backend != Backend::NPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported sampler backend: ", sampler_backend));
  }

  if (benchmark_info.has_value()) {
    ABSL_LOG(INFO) << "Benchmark is enabled.";
  }
  StopTokenDetector stop_token_detector(
      session_config.GetNumOutputCandidates());
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    RETURN_IF_ERROR(
        stop_token_detector.AddStopTokenSequence(stop_token_sequence));
  }

  occupied_executors_->insert(executor);
  return absl::WrapUnique(new SessionBasic(
      executor, tokenizer, vision_executor, audio_executor, std::move(sampler),
      session_config, benchmark_info, worker_thread_pool, stop_token_detector));
}

SessionBasic::~SessionBasic() {
  auto status = executor_.Reset();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to reset executor: " << status;
  }
  if (audio_executor_ != nullptr) {
    status = audio_executor_->Reset();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to reset audio executor: " << status;
    }
  }
  absl::MutexLock lock(occupied_executors_mu_);  // NOLINT
  occupied_executors_->erase(&executor_);
}

namespace {

// Concrete TrainableParameterHandle that holds LoRA TensorBuffers alive.
class LoraTrainableParameterHandle : public TrainableParameterHandle {
 public:
  // Takes ownership of the duplicated TensorBuffer map.
  static absl::StatusOr<std::unique_ptr<LoraTrainableParameterHandle>> Create(
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> buffers) {
    auto handle = std::make_unique<LoraTrainableParameterHandle>();

    for (auto& [name, buffer] : buffers) {
      // Lock briefly to discover the float* address.
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          ::litert::TensorBufferScopedLock::Create(
              buffer, ::litert::TensorBuffer::LockMode::kWrite));
      float* data = static_cast<float*>(lock_and_addr.second);

      LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
      LITERT_ASSIGN_OR_RETURN(auto num_elements,
                              tensor_type.Layout().NumElements());

      bool is_bias_or_ln = !IsLoRAInputName(name);

      TrainableParameter param;
      param.name = std::string(name);
      param.data = data;
      param.num_elements = num_elements;
      param.is_bias_or_layernorm = is_bias_or_ln;
      handle->parameters_.push_back(std::move(param));

      // Keep the lock alive to maintain the float* validity.
      handle->locks_.push_back(std::move(lock_and_addr.first));
      // Keep the TensorBuffer alive.
      handle->buffers_.push_back(std::move(buffer));
    }

    return handle;
  }

  const std::vector<TrainableParameter>& GetParameters() const override {
    return parameters_;
  }

 private:
  friend class std::unique_ptr<LoraTrainableParameterHandle>;
  std::vector<TrainableParameter> parameters_;
  std::vector<::litert::TensorBuffer> buffers_;
  std::vector<::litert::TensorBufferScopedLock> locks_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<TrainableParameterHandle>>
SessionBasic::GetTrainableParameters() {
  LoraManager* lora_manager = executor_.GetLoraManager();
  if (lora_manager == nullptr) {
    return absl::FailedPreconditionError(
        "No LoRA manager available. The executor does not support LoRA.");
  }
  ASSIGN_OR_RETURN(auto buffers, lora_manager->GetLoRABuffers());
  if (buffers.empty()) {
    return absl::NotFoundError("No trainable LoRA parameters found.");
  }
  auto handle = LoraTrainableParameterHandle::Create(std::move(buffers));
  if (!handle.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to create trainable parameter handle: ",
                     handle.status().ToString()));
  }
  return std::unique_ptr<TrainableParameterHandle>(*std::move(handle));
}

// TODO - b/436674053: Modularize the preprocessing logic into a separate
// preprocessor class, and have unit test for it.
absl::StatusOr<ExecutorInputs> SessionBasic::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents) {
  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;
  for (const auto& preprocessed_content : preprocessed_contents) {
    if (const auto* input_text =
            std::get_if<InputText>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* token_ids,
                       input_text->GetPreprocessedTextTensor());
      if (token_ids == nullptr) {
        return absl::InvalidArgumentError(
            "Token IDs is null in preprocessed_contents.");
      }
      LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                              ReferTensorBufferAsSpan<int>(*token_ids));
      combined_token_ids.insert(combined_token_ids.end(),
                                ids_buffer_span.begin(), ids_buffer_span.end());
    } else if (const auto* input_image =
                   std::get_if<InputImage>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* image_tensor,
                       input_image->GetPreprocessedImageTensor());
      if (image_tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Image tensor is null in preprocessed_contents.");
      }
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto single_image_data,
                       vision_executor_->Encode(*image_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto embeddings_ptr,
                       single_image_data.GetEmbeddingsPtr());
      const auto& dimensions = TensorBufferDims(*embeddings_ptr);
      // The last two dimensions are [..., image_token_num, model_dimension].
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_audio =
                   std::get_if<InputAudio>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                       input_audio->GetPreprocessedAudioTensor());
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      ASSIGN_OR_RETURN(auto single_audio_data,
                       audio_executor_->Encode(*spectrogram_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      all_audio_data.push_back(std::move(single_audio_data));
      combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                ExecutorAudioData::kSpecialToken);
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported input data type in preprocessed_contents.");
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ASSIGN_OR_RETURN(combined_image_data,
                     CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ASSIGN_OR_RETURN(combined_audio_data,
                     CombineExecutorAudioData(all_audio_data));
  }

  ASSIGN_OR_RETURN(auto token_ids_buffer,
                   tokenizer_.TokenIdsToTensorBuffer(combined_token_ids));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::Status SessionBasic::PrefillInternal(
    const std::vector<InputData>& preprocessed_contents,
    bool wait_for_completion) {
  ASSIGN_OR_RETURN(ExecutorInputs inputs,
                   ProcessAndCombineContents(preprocessed_contents));

  ASSIGN_OR_RETURN(
      last_prefill_token_id_,
      Prefill(executor_, inputs, wait_for_completion, benchmark_info_));
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(const std::vector<InputData>& contents) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ABSL_LOG(INFO) << "RunPrefill: ";
  for (const auto& content : contents) {
    ABSL_LOG(INFO) << content;
  }

  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  std::vector<InputData> preprocessed_contents;
  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(contents, session_config_, tokenizer_,
                                        benchmark_info_));
  } else {
    ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                     ApplyPromptTemplates(contents, session_config_, tokenizer_,
                                          is_first_turn_));
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(templated_contents, session_config_,
                                        tokenizer_, benchmark_info_));
  }

  return PrefillInternal(preprocessed_contents,
                         /*wait_for_completion=*/true);
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionBasic::RunPrefillAsync(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ABSL_LOG(INFO) << "RunPrefillAsync: ";
  for (const auto& content : contents) {
    ABSL_LOG(INFO) << content;
  }

  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  std::vector<InputData> preprocessed_contents;
  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(contents, session_config_, tokenizer_,
                                        benchmark_info_));
  } else {
    ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                     ApplyPromptTemplates(contents, session_config_, tokenizer_,
                                          is_first_turn_));
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(templated_contents, session_config_,
                                        tokenizer_, benchmark_info_));
  }
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, preprocessed_contents = std::move(preprocessed_contents),
       callback = std::move(callback)]() mutable {
        absl::Status status = this->PrefillInternal(
            preprocessed_contents, /*wait_for_completion=*/false);
        ABSL_LOG(INFO) << "RunPrefillAsync status: " << status;
        if (cancelled_.load()) {
          callback(
              absl::CancelledError("Session is cancelled during prefill."));
          return;
        }
        if (!status.ok()) {
          callback(status);
        } else {
          callback(Responses(TaskState::kDone));
        }
      }));
  return nullptr;
}

absl::StatusOr<Responses> SessionBasic::DecodeInternal(
    const DecodeConfig& decode_config) {
  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(auto responses,
                     Decode(executor_, tokenizer_, stop_token_detector_,
                            session_config_.GetNumOutputCandidates(),
                            decode_config.GetConstraint(), benchmark_info_,
                            &cancelled_, session_config_.GetMaxOutputTokens()));
    return responses;
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    LITERT_ASSIGN_OR_RETURN(
        auto decoded_ids_buffer,
        CopyToTensorBuffer<int>(decoded_ids,
                                {session_config_.GetNumOutputCandidates(), 1}));
    ASSIGN_OR_RETURN(auto responses,
                     DecodeCustomSampling(
                         executor_, tokenizer_, stop_token_detector_,
                         session_config_.GetNumOutputCandidates(), *sampler_,
                         std::move(decoded_ids_buffer),
                         decode_config.GetConstraint(), benchmark_info_,
                         &cancelled_, session_config_.GetMaxOutputTokens()));
    return responses;
  }
}

absl::Status SessionBasic::DecodeInternalStreaming(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  if (sampler_ == nullptr) {
    RETURN_IF_ERROR(DecodeStreaming(
        executor_, tokenizer_, stop_token_detector_,
        session_config_.GetNumOutputCandidates(), decode_config.GetConstraint(),
        benchmark_info_, std::move(callback), &cancelled_,
        session_config_.GetMaxOutputTokens()));
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    LITERT_ASSIGN_OR_RETURN(
        auto decoded_ids_buffer,
        CopyToTensorBuffer<int>(decoded_ids,
                                {session_config_.GetNumOutputCandidates(), 1}));

    RETURN_IF_ERROR(DecodeCustomSamplingStreaming(
        executor_, tokenizer_, stop_token_detector_,
        session_config_.GetNumOutputCandidates(), *sampler_,
        std::move(decoded_ids_buffer), decode_config.GetConstraint(),
        benchmark_info_, std::move(callback), &cancelled_,
        session_config_.GetMaxOutputTokens()));
  }
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::RunDecode() {
  return RunDecode(DecodeConfig::CreateDefault());
}

absl::StatusOr<Responses> SessionBasic::RunDecode(
    const DecodeConfig& decode_config) {
  ABSL_LOG(INFO) << "RunDecodeSync";
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  return DecodeInternal(decode_config);
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionBasic::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return RunDecodeAsync(std::move(callback), DecodeConfig::CreateDefault());
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionBasic::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  ABSL_LOG(INFO) << "RunDecodeAsync";
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, callback = std::move(callback), decode_config]() mutable {
        this->DecodeInternalStreaming(std::move(callback), decode_config)
            .IgnoreError();
      }));
  return nullptr;
}

absl::StatusOr<Responses> SessionBasic::GenerateContent(
    const std::vector<InputData>& contents) {
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode(DecodeConfig::CreateDefault());
}

absl::StatusOr<Responses> SessionBasic::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    bool store_token_lengths) {
  absl::StatusOr<Responses> collected_responses;
  auto scoring_sync_callback =
      [&collected_responses](absl::StatusOr<Responses> responses) {
        collected_responses = std::move(responses);
      };

  ASSIGN_OR_RETURN(auto task_controller,
                   RunTextScoringAsync(target_text,
                                       std::move(scoring_sync_callback),
                                       store_token_lengths));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>
SessionBasic::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    bool store_token_lengths) {
  if (target_text.size() != 1) {
    return absl::InvalidArgumentError("Target text size should be 1.");
  }

  // TODO(b/435040163): Handle the temperature. Should it be calculated from
  // the sampler or the sampler parameters? For now, hardcode it to 1.0f for
  // testing.
  auto temperature = 1.0f;
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, callback = std::move(callback), target_text, store_token_lengths,
       temperature]() mutable {
        std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                     last_prefill_token_id_);
        auto decoded_ids_buffer = CopyToTensorBuffer<int>(
            decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
        if (!decoded_ids_buffer.HasValue()) {
          callback(absl::InternalError(decoded_ids_buffer.Error().Message()));
          return;
        }
        callback(ScoreCustomSampling(executor_, tokenizer_, target_text,
                                     temperature,
                                     std::move(decoded_ids_buffer.Value()),
                                     store_token_lengths));
      }));
  return nullptr;
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return GenerateContentStream(contents, std::move(callback),
                               DecodeConfig::CreateDefault());
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }

  ASSIGN_OR_RETURN(
      auto task_controller,
      RunPrefillAsync(
          contents,
          [this, callback = std::move(callback), decode_config = decode_config](
              absl::StatusOr<Responses> responses) mutable {
            if (!responses.ok()) {
              callback(responses.status());
            } else {
              if (cancelled_.load()) {
                callback(absl::CancelledError(
                    "Session is cancelled during prefill."));
                return;
              }
              auto status = RunDecodeAsync(std::move(callback), decode_config);
            }
          }));
  return absl::OkStatus();
}

absl::StatusOr<BenchmarkInfo> SessionBasic::GetBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

absl::StatusOr<BenchmarkInfo*> SessionBasic::GetMutableBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return &benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

}  // namespace litert::lm
