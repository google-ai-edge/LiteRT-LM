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
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/pipeline.h"
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
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

template <typename T>
absl::StatusOr<T> CombineExecutorDataImpl(std::vector<T>& executor_data) {
  if (executor_data.empty()) {
    return absl::InvalidArgumentError("Executor data is empty.");
  }
  if (executor_data.size() == 1) {
    // If there is only one image, we can just move it to the combined image
    // data.
    return std::move(executor_data[0]);
  }
  // If there are multiple executor data, we need to first combine them into a
  // TensorBuffer, then create a single ExecutorVisionData from the
  // TensorBuffer.
  int num_executor_data = executor_data.size();
  ASSIGN_OR_RETURN(const auto* first_tensor,
                   executor_data[0].GetEmbeddingsPtr());
  LITERT_ASSIGN_OR_RETURN_ABSL(auto first_tensor_type,
                               first_tensor->TensorType());
  auto first_tensor_dims = TensorBufferDims(*first_tensor);
  int total_token_num = 0;
  int total_packed_size = 0;
  std::vector<int> combined_token_num;
  for (const auto& executor_data : executor_data) {
    ASSIGN_OR_RETURN(const auto* embeddings_ptr,
                     executor_data.GetEmbeddingsPtr());
    auto dims = TensorBufferDims(*embeddings_ptr);
    if (dims.size() != 3 && dims.size() != 4) {
      return absl::InvalidArgumentError(
          "The embedding tensor type must have 3 or 4 dimensions.");
    }
    combined_token_num.push_back(dims[dims.size() - 2]);
    total_token_num += dims[dims.size() - 2];
    LITERT_ASSIGN_OR_RETURN_ABSL(size_t packed_size,
                                 embeddings_ptr->PackedSize());
    total_packed_size += packed_size;
  }
  Layout combined_layout;
  if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 3) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], 1, total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 4) {
    combined_layout =
        Layout(Dimensions({first_tensor_dims[0], first_tensor_dims[1],
                           total_token_num, first_tensor_dims[3]}));
  }
  ::litert::RankedTensorType combined_tensor_type(
      first_tensor_type.ElementType(), std::move(combined_layout));

  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto combined_tensor_buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeHostMemory,
                                  combined_tensor_type, total_packed_size));
  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto combined_embeddings_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(combined_tensor_buffer,
                                               TensorBuffer::LockMode::kWrite));
  char* combined_tensor_buffer_ptr =
      static_cast<char*>(combined_embeddings_lock_and_addr.second);
  for (int i = 0; i < num_executor_data; ++i) {
    ASSIGN_OR_RETURN(auto embeddings_ptr,
                     executor_data[i].GetMutableEmbeddingsPtr());
    LITERT_ASSIGN_OR_RETURN_ABSL(auto embeddings_size,
                                 embeddings_ptr->PackedSize());
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto embeddings_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            *embeddings_ptr, TensorBuffer::LockMode::kRead));
    memcpy(combined_tensor_buffer_ptr, embeddings_lock_and_addr.second,
           embeddings_size);
    combined_tensor_buffer_ptr += embeddings_size;
  }
  if constexpr (std::is_same_v<T, ExecutorVisionData>) {
    return ExecutorVisionData(std::move(combined_tensor_buffer),
                              /*per_layer_embeddings=*/std::nullopt);
  } else if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    int num_audio_tokens = 0;
    for (const auto& executor_data : executor_data) {
      num_audio_tokens += executor_data.GetValidTokens();
    }
    return ExecutorAudioData(std::move(combined_tensor_buffer),
                             /*per_layer_embeddings=*/std::nullopt,
                             num_audio_tokens);
  } else {
    return absl::InvalidArgumentError("Executor data type is not supported.");
  }
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    LlmExecutor* executor, Tokenizer* tokenizer,
    VisionExecutor* vision_executor, AudioExecutor* audio_executor,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* worker_thread_pool) {
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
  return absl::WrapUnique(new SessionBasic(
      executor, tokenizer, vision_executor, audio_executor, std::move(sampler),
      session_config, benchmark_info, worker_thread_pool, stop_token_detector));
}

SessionBasic::~SessionBasic() {
  auto status = executor_.Reset();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to reset executor: " << status;
  }
}

absl::StatusOr<ExecutorVisionData> SessionBasic::CombineExecutorData(
    std::vector<ExecutorVisionData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

absl::StatusOr<ExecutorAudioData> SessionBasic::CombineExecutorData(
    std::vector<ExecutorAudioData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

absl::StatusOr<std::vector<InputData>> SessionBasic::ApplyPromptTemplates(
    const std::vector<InputData>& contents) {
  auto bos_token_id = session_config_.GetStartTokenId();
  std::string bos_string = "";
  // Lookup the BOS string from the tokenizer.
  // If the BOS token id is not valid, the bos string will remain empty.
  if (bos_token_id >= 0) {
    ASSIGN_OR_RETURN(bos_string, tokenizer_.TokenIdsToText({bos_token_id}));
  }

  std::vector<InputData> templated_contents;
  for (int i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];
    const bool is_first_chunk = i == 0;
    const bool is_last_chunk = i == contents.size() - 1;
    absl::string_view raw_text = "";
    if (const auto* input_text = std::get_if<InputText>(&content);
        input_text != nullptr && !input_text->IsTensorBuffer()) {
      ASSIGN_OR_RETURN(raw_text, input_text->GetRawTextString());
    }

    // Check if the input starts with the BOS string. If it does, return an
    // error. This is to prevent the user from including the BOS string in the
    // input. This is also needed for the current implementation as tokenizer
    // will treat the BOS string differently from other strings. If the BOS
    // string is empty, it means the BOS token id is not valid. In this case, we
    // will not check for the BOS string in the input.
    if (!bos_string.empty() && absl::StartsWith(raw_text, bos_string)) {
      return absl::InvalidArgumentError(
          "Input contains bos control token. Control token should not be "
          "included in the input.");
    }

    std::string session_prefix = "";
    if (is_first_chunk) {
      session_prefix = is_first_turn_ ? bos_string : "\n";
      if (is_first_turn_) is_first_turn_ = false;
    }
    std::string turn_prefix = absl::StrCat(
        session_prefix, session_config_.GetPromptTemplates().user().prefix());
    std::string turn_suffix =
        absl::StrCat(session_config_.GetPromptTemplates().user().suffix(),
                     session_config_.GetPromptTemplates().model().prefix());

    if (raw_text.empty()) {
      // Non-text chunk. Add templates as separate InputText objects.
      if (is_first_chunk && !turn_prefix.empty()) {
        templated_contents.push_back(InputText(std::move(turn_prefix)));
      }
      // TODO - b/445254659: Remove all actual copies.
      ASSIGN_OR_RETURN(auto content_copy, CreateInputDataCopy(content));
      templated_contents.emplace_back(std::move(content_copy));
      if (is_last_chunk && !turn_suffix.empty()) {
        templated_contents.push_back(InputText(std::move(turn_suffix)));
      }
    } else {
      // Raw text chunk. Combine templates with the raw text.
      std::string templated_text;
      if (is_first_chunk) {
        templated_text = absl::StrCat(turn_prefix, raw_text);
      } else {
        templated_text = std::string(raw_text);
      }
      if (is_last_chunk) {
        absl::StrAppend(&templated_text, turn_suffix);
      }
      templated_contents.push_back(InputText(std::move(templated_text)));
    }
  }
  return templated_contents;
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
      LITERT_ASSIGN_OR_RETURN_ABSL(auto ids_buffer_span,
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
      ASSIGN_OR_RETURN(auto single_image_data,
                       vision_executor_->Encode(*image_tensor));
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
      ASSIGN_OR_RETURN(auto single_audio_data,
                       audio_executor_->Encode(*spectrogram_tensor));
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      all_audio_data.push_back(std::move(single_audio_data));
      combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                ExecutorAudioData::kSpecialToken);
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ASSIGN_OR_RETURN(combined_image_data, CombineExecutorData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ASSIGN_OR_RETURN(combined_audio_data, CombineExecutorData(all_audio_data));
  }

  ASSIGN_OR_RETURN(auto token_ids_buffer,
                   tokenizer_.TokenIdsToTensorBuffer(combined_token_ids));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::StatusOr<InputText> SessionBasic::StringToProcessedInputText(
    absl::string_view text) {
  auto bos_token_id = session_config_.GetStartTokenId();
  std::string bos_string = "";
  if (bos_token_id >= 0) {
    ASSIGN_OR_RETURN(bos_string, tokenizer_.TokenIdsToText({bos_token_id}));
  }
  bool bos_token_found = false;
  if (!bos_string.empty() && absl::StartsWith(text, bos_string)) {
    text = text.substr(bos_string.size());
    bos_token_found = true;
  }

  int benchmark_prefill_token_count = 0;
  if (benchmark_info_.has_value()) {
    benchmark_prefill_token_count =
        benchmark_info_->GetBenchmarkParams().num_prefill_tokens();
  }
  ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer_.TextToTokenIds(text));
  if (benchmark_prefill_token_count > 0) {
    // If benchmark is enabled, we will use the benchmark prefill token
    // count to set the prefill token count.
    ids.resize(benchmark_prefill_token_count);
  } else if (bos_token_found) {
    ids.insert(ids.begin(), session_config_.GetStartTokenId());
  }
  ASSIGN_OR_RETURN(auto ids_buffer, tokenizer_.TokenIdsToTensorBuffer(ids));
  return InputText(std::move(ids_buffer));
}

absl::StatusOr<std::vector<InputData>> SessionBasic::PreprocessContents(
    const std::vector<InputData>& contents) {
  std::vector<InputData> preprocessed_contents;
  for (int i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];
    if (const auto* input_text = std::get_if<InputText>(&content)) {
      if (input_text->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto input_text_copy, input_text->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_text_copy));
      } else {
        ASSIGN_OR_RETURN(auto templated_text, input_text->GetRawTextString());
        ASSIGN_OR_RETURN(auto processed_input_text,
                         StringToProcessedInputText(templated_text));
        preprocessed_contents.emplace_back(std::move(processed_input_text));
      }
    } else if (const auto* input_image = std::get_if<InputImage>(&content)) {
      if (input_image->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto input_image_copy, input_image->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_image_copy));
      } else {
        return absl::InternalError(
            "Image must be preprocessed before being used in SessionBasic.");
      }
    } else if (const auto* input_audio = std::get_if<InputAudio>(&content)) {
      if (input_audio->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto input_audio_copy, input_audio->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_audio_copy));
      } else {
        return absl::InternalError(
            "Audio must be preprocessed before being used in SessionBasic.");
      }
    }
  }
  return preprocessed_contents;
}

absl::Status SessionBasic::PrefillInternal(
    const std::vector<InputData>& preprocessed_contents,
    bool wait_for_completion) {
  ASSIGN_OR_RETURN(ExecutorInputs inputs,
                   ProcessAndCombineContents(preprocessed_contents));

  // This should be added to the beginning of the next prefill call as will no?
  // Also, this is not thread safe. More discussion with @ztenghui is needed.
  ASSIGN_OR_RETURN(
      last_prefill_token_id_,
      Prefill(executor_, inputs, wait_for_completion, benchmark_info_));
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(const std::vector<InputData>& contents) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  if (benchmark_info_.has_value()) {
    RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnStart());
  }
  std::vector<InputData> preprocessed_contents;
  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    ASSIGN_OR_RETURN(preprocessed_contents, PreprocessContents(contents));
  } else {
    ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                     ApplyPromptTemplates(contents));
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(templated_contents));
  }
  absl::Status status;
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, preprocessed_contents = std::move(preprocessed_contents),
       &status]() {
        status = this->PrefillInternal(preprocessed_contents,
                                       /*wait_for_completion=*/true);
      }));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return status;
}

absl::Status SessionBasic::RunPrefillAsync(
    const std::vector<InputData>& contents,
    std::unique_ptr<InferenceCallbacks> callbacks) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  if (benchmark_info_.has_value()) {
    RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnStart());
  }
  std::vector<InputData> preprocessed_contents;
  if (benchmark_info_.has_value() &&
      benchmark_info_->GetBenchmarkParams().num_prefill_tokens() > 0) {
    ASSIGN_OR_RETURN(preprocessed_contents, PreprocessContents(contents));
  } else {
    ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                     ApplyPromptTemplates(contents));
    ASSIGN_OR_RETURN(preprocessed_contents,
                     PreprocessContents(templated_contents));
  }
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, preprocessed_contents = std::move(preprocessed_contents),
       callbacks = std::move(callbacks)]() {
        absl::Status status = this->PrefillInternal(
            preprocessed_contents, /*wait_for_completion=*/false);
        ABSL_LOG(INFO) << "RunPrefillAsync status: " << status;
        if (status.ok()) {
          callbacks->OnDone();
        } else {
          callbacks->OnError(status);
        }
      }));
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::DecodeInternal(
    const DecodeConfig& decode_config) {
  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(
        auto responses,
        Decode(executor_, tokenizer_, stop_token_detector_,
               session_config_.GetNumOutputCandidates(),
               decode_config.GetConstraint(), benchmark_info_, &cancelled_));
    return responses;
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    auto decoded_ids_buffer = CopyToTensorBuffer<int>(
        decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
    ASSIGN_OR_RETURN(auto responses,
                     DecodeCustomSampling(
                         executor_, tokenizer_, stop_token_detector_,
                         session_config_.GetNumOutputCandidates(), *sampler_,
                         *decoded_ids_buffer, decode_config.GetConstraint(),
                         benchmark_info_, &cancelled_));
    return responses;
  }
}

absl::Status SessionBasic::DecodeInternalStreaming(
    std::unique_ptr<InferenceCallbacks> callbacks,
    const DecodeConfig& decode_config) {
  if (sampler_ == nullptr) {
    RETURN_IF_ERROR(DecodeStreaming(
        executor_, tokenizer_, stop_token_detector_,
        session_config_.GetNumOutputCandidates(), decode_config.GetConstraint(),
        benchmark_info_, std::move(callbacks), &cancelled_));
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    auto decoded_ids_buffer = CopyToTensorBuffer<int>(
        decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
    RETURN_IF_ERROR(DecodeCustomSamplingStreaming(
        executor_, tokenizer_, stop_token_detector_,
        session_config_.GetNumOutputCandidates(), *sampler_,
        *decoded_ids_buffer, decode_config.GetConstraint(), benchmark_info_,
        std::move(callbacks), &cancelled_));
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
  absl::StatusOr<Responses> responses;
  RETURN_IF_ERROR(
      worker_thread_pool_.Schedule([this, &responses, decode_config]() {
        responses = this->DecodeInternal(decode_config);
      }));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return responses;
}

absl::Status SessionBasic::RunDecodeAsync(
    std::unique_ptr<InferenceCallbacks> callbacks) {
  return RunDecodeAsync(std::move(callbacks), DecodeConfig::CreateDefault());
}

absl::Status SessionBasic::RunDecodeAsync(
    std::unique_ptr<InferenceCallbacks> callbacks,
    const DecodeConfig& decode_config) {
  ABSL_LOG(INFO) << "RunDecodeAsync";
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  return worker_thread_pool_.Schedule(
      [this, callbacks = std::move(callbacks), decode_config]() mutable {
        this->DecodeInternalStreaming(std::move(callbacks), decode_config)
            .IgnoreError();
      });
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
    const std::vector<absl::string_view>& target_text) {
  if (target_text.empty()) {
    return absl::InvalidArgumentError("Target text is empty.");
  }
  std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                               last_prefill_token_id_);
  // Remove the last token from the decoded ids, since the last prefill token
  // is used as the query during decoding of the model.
  auto decoded_ids_buffer = CopyToTensorBuffer<int>(
      decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
  // TODO(b/435040163): Handle the temperature. Should it be calculated from
  // the sampler or the sampler parameters? For now, hardcode it to 1.0f for
  // testing.
  auto temperature = 1.0f;
  absl::StatusOr<Responses> score;
  // Scheduled on the worker thread pool to ensure serialized execution with
  // other engine operations as the function waits for completion.
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, &score, &target_text, &decoded_ids_buffer, &temperature]() {
        score = ScoreCustomSampling(executor_, tokenizer_, target_text,
                                    temperature, *decoded_ids_buffer);
      }));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return score;
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents,
    std::unique_ptr<InferenceCallbacks> callbacks) {
  return GenerateContentStream(contents, std::move(callbacks),
                               DecodeConfig::CreateDefault());
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents,
    std::unique_ptr<InferenceCallbacks> callbacks,
    const DecodeConfig& decode_config) {
  if (cancelled_.load()) {
    // Reset the cancelled flag before processing the next turn.
    cancelled_ = false;
  }
  // An callbacks to handle the result of the async prefill operation.
  // It triggers the decode step if prefill is successful, or propagates the
  // error.
  class PrefillCallbacks : public InferenceCallbacks {
   public:
    PrefillCallbacks(SessionBasic* session,
                     std::unique_ptr<InferenceCallbacks> next_callbacks,
                     std::atomic<bool>* cancelled,
                     const DecodeConfig& decode_config)
        : session_(session),
          next_callbacks_(std::move(next_callbacks)),
          cancelled_(cancelled),
          decode_config_(decode_config) {}

    void OnNext(const Responses& responses) override {
      ABSL_LOG(WARNING) << "OnNext should not be called during prefill!";
    }

    void OnError(const absl::Status& status) override {
      next_callbacks_->OnError(status);
    }

    void OnDone() override {
      if (cancelled_ != nullptr && cancelled_->load()) {
        if (*cancelled_) {
          next_callbacks_->OnError(
              absl::CancelledError("CancelProcess is called during prefill. "
                                   "Skip the following decode call."));
          return;
        }
      }
      absl::Status status =
          session_->RunDecodeAsync(std::move(next_callbacks_), decode_config_);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to start decode async: " << status;
      }
    }

   private:
    SessionBasic* session_;
    mutable std::unique_ptr<InferenceCallbacks> next_callbacks_;
    std::atomic<bool>* cancelled_;  // Not owned.
    DecodeConfig decode_config_;
  };

  auto prefill_callbacks = std::make_unique<PrefillCallbacks>(
      this, std::move(callbacks), &cancelled_, decode_config);
  RETURN_IF_ERROR(RunPrefillAsync(contents, std::move(prefill_callbacks)));
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

}  // namespace litert::lm
