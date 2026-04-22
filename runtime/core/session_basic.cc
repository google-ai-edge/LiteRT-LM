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
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
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
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
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

  bool enable_speculative_decoding = false;
  {
    ASSIGN_OR_RETURN(auto executor_settings, executor->GetExecutorSettings());
    auto advanced_settings = executor_settings.GetAdvancedSettings();
    if (advanced_settings.has_value()) {
      enable_speculative_decoding =
          advanced_settings->enable_speculative_decoding;
    }
  }

  auto sampler_backend = session_config.GetSamplerBackend();
  std::unique_ptr<Sampler> sampler;
  // If use CPU sampling, we create it here; For GPU sampling and when
  // speculative decoding is enabled, we let executor create it internally.
  if (sampler_backend == Backend::CPU) {
    if (!enable_speculative_decoding) {
      ASSIGN_OR_RETURN(sampler,
                       CreateSampler(sampler_backend,
                                     session_config.GetNumOutputCandidates(),
                                     session_config.GetSamplerParams()));
    }
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
  WaitUntilDone().IgnoreError();
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
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ExecutorVisionData single_image_data;
      if (input_image->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto tensor_buffer,
                         input_image->GetPreprocessedImageTensor());
        ASSIGN_OR_RETURN(single_image_data,
                         vision_executor_->Encode(*tensor_buffer));
      } else if (input_image->IsTensorBufferMap()) {
        ASSIGN_OR_RETURN(auto tensor_buffer_map,
                         input_image->GetPreprocessedImageTensorMap());
        ASSIGN_OR_RETURN(single_image_data,
                         vision_executor_->Encode(*tensor_buffer_map));
      } else {
        return absl::FailedPreconditionError(
            "The image is not preprocessed and does not have a tensor.");
      }
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto embeddings_ptr,
                       single_image_data.GetEmbeddingsPtr());
      ASSIGN_OR_RETURN(const auto& dimensions,
                       TensorBufferDims(*embeddings_ptr));
      // The last two dimensions are [..., image_token_num, model_dimension].
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_image_end =
                   std::get_if<InputImageEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorVisionData::kEndToken);
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

absl::Status SessionBasic::SaveCheckpoint(absl::string_view label) {
  ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
  checkpoint_map_[label] = current_step;
  return absl::OkStatus();
}

absl::Status SessionBasic::RewindToCheckpoint(absl::string_view label) {
  if (auto it = checkpoint_map_.find(label); it != checkpoint_map_.end()) {
    ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
    if (it->second > current_step) {
      // This shouldn't ever happen because we remove all checkpoints after the
      // current step when we rewind to a checkpoint.
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot rewind to a future step: ", it->second));
    }

    // Set the current step of the executor to the checkpoint step.
    RETURN_IF_ERROR(executor_.SetCurrentStep(it->second));

    // Remove all checkpoints after the current step.
    absl::erase_if(checkpoint_map_,
                   [current_step = it->second](const auto& pair) {
                     return pair.second > current_step;
                   });
    return absl::OkStatus();
  }
  return absl::NotFoundError(absl::StrCat("Checkpoint not found: ", label));
}

absl::StatusOr<int> SessionBasic::GetCurrentStep() const {
  return executor_.GetCurrentStep();
}

absl::Status SessionBasic::PrefillInternal(
    const std::vector<InputData>& preprocessed_contents,
    bool wait_for_completion) {
  ASSIGN_OR_RETURN(ExecutorInputs inputs,
                   ProcessAndCombineContents(preprocessed_contents));
  ASSIGN_OR_RETURN(
      last_prefill_token_id_,
      Prefill(executor_, inputs, wait_for_completion, benchmark_info_));
  session_state_ = SessionState::kPrefilled;
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(const std::vector<InputData>& contents) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ABSL_VLOG(1) << "RunPrefill: ";
  for (const auto& content : contents) {
    ABSL_VLOG(1) << content;
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
    bool is_first_turn = session_state_ == SessionState::kFresh;
    ContentType content_type;
    if (session_config_.GetApplyPromptTemplateInSession()) {
      content_type = (is_first_turn || session_state_ == SessionState::kDecoded)
                         ? ContentType::kFirst
                         : ContentType::kMiddle;
    } else {
      content_type = ContentType::kNA;
    }
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, content_type, session_config_,
                             tokenizer_, is_first_turn));
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
  ABSL_VLOG(1) << "RunPrefillAsync: ";
  for (const auto& content : contents) {
    ABSL_VLOG(1) << content;
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
    bool is_first_turn = session_state_ == SessionState::kFresh;
    ContentType content_type;
    if (session_config_.GetApplyPromptTemplateInSession()) {
      content_type = (is_first_turn || session_state_ == SessionState::kDecoded)
                         ? ContentType::kFirst
                         : ContentType::kMiddle;
    } else {
      content_type = ContentType::kNA;
    }
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, content_type, session_config_,
                             tokenizer_, is_first_turn));
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
  if (session_state_ != SessionState::kPrefilled) {
    return absl::InternalError("Session is not prefilled yet.");
  }

  // We need to do a last prefill before initializing the decode, to make sure
  // the prompt is correctly set up for decode.
  if (session_config_.GetApplyPromptTemplateInSession()) {
    std::vector<InputData> contents;
    contents.emplace_back(InputText(""));
    ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, ContentType::kLast, session_config_,
                             tokenizer_, /*is_first_turn=*/false));
    if (!templated_contents.empty()) {
      ASSIGN_OR_RETURN(std::vector<InputData> preprocessed_contents,
                       PreprocessContents(templated_contents, session_config_,
                                          tokenizer_, benchmark_info_));
      RETURN_IF_ERROR(PrefillInternal(preprocessed_contents,
                                      /*wait_for_completion=*/true));
    }
  }
  session_state_ = SessionState::kDecoded;

  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(
        auto responses,
        Decode(executor_, tokenizer_, stop_token_detector_,
               session_config_.GetNumOutputCandidates(),
               decode_config.GetConstraint(), benchmark_info_, &cancelled_,
               decode_config.GetMaxOutputTokens().value_or(
                   session_config_.GetMaxOutputTokens())));
    return responses;
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    LITERT_ASSIGN_OR_RETURN(
        auto decoded_ids_buffer,
        CopyToTensorBuffer<int>(decoded_ids,
                                {session_config_.GetNumOutputCandidates(), 1}));
    ASSIGN_OR_RETURN(
        auto responses,
        DecodeCustomSampling(executor_, tokenizer_, stop_token_detector_,
                             session_config_.GetNumOutputCandidates(),
                             *sampler_, std::move(decoded_ids_buffer),
                             decode_config.GetConstraint(), benchmark_info_,
                             &cancelled_,
                             decode_config.GetMaxOutputTokens().value_or(
                                 session_config_.GetMaxOutputTokens())));
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
        decode_config.GetMaxOutputTokens().value_or(
            session_config_.GetMaxOutputTokens())));
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
        decode_config.GetMaxOutputTokens().value_or(
            session_config_.GetMaxOutputTokens())));
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

  ASSIGN_OR_RETURN(
      auto task_controller,
      RunTextScoringAsync(target_text, std::move(scoring_sync_callback),
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
        callback(ScoreCustomSampling(
            executor_, tokenizer_, target_text, temperature,
            std::move(decoded_ids_buffer.Value()), store_token_lengths));
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
