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

#include "runtime/core/tasks.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/composite_constraint.h"
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_constraint.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_constraint.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_constraint.h"
#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/scoring_cpu_util.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "support/tokenizer/buffered_streaming_detokenizer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm::Tasks {
namespace {

// Converts a span of fp16 values to a vector of fp32 values.
// TODO: b/499304966 - move this to a common util file and add tests.
void ConvertFp16ToFp32(absl::Span<const tflite::half> fp16_values,
                       std::vector<float>& out) {
  out.resize(fp16_values.size());
  for (int i = 0; i < fp16_values.size(); ++i) {
    out[i] = static_cast<float>(fp16_values[i]);
  }
}

// TODO(b/423364170): all LLM Executors should respect the max number of tokens
// returned by the model. We should remove this default value once all Executors
// are compliant with the max number of tokens.
constexpr int kDefaultMaxNumTokens = 4096;
int TryGetMaxNumTokens(const LlmExecutor& executor) {
  auto settings = executor.GetExecutorSettings();
  if (!settings.ok()) {
    // If the executor settings are not available, we will use the default
    // value.
    ABSL_LOG(WARNING) << "Failed to get executor settings: "
                      << settings.status();
    return kDefaultMaxNumTokens;
  }
  return settings->GetMaxNumTokens();
}

// Check whether the decoding loop should stop.
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                int max_output_tokens) {
  // Stopping conditions.
  if (hit_stop_tokens && benchmark_decode_token_count == 0) {
    // Only early stop if no decode step
    // is requested by benchmark.
    return true;
  } else if (benchmark_decode_token_count > 0 &&
             num_decoded_steps >= benchmark_decode_token_count) {
    // Stop when the number of decode steps is equal to the
    // benchmark_decode_token_count (when specified).
    return true;
  } else if (current_step >= max_num_tokens) {
    // Reaching maximum number of kv-cache size.
    return true;
  } else if (num_decoded_steps >= max_output_tokens) {
    // Reaching maximum number of output tokens.
    return true;
  }
  return false;
}

// A helper class to filter the token stream based on stop tokens.
//
// This filter is needed because detokenization is typically streaming and
// incremental. Once a token is fed to the detokenizer, it may release
// corresponding text immediately. If we were to feed a token that is part of a
// stop sequence (or a partial stop sequence), the detokenizer might release
// text that should have been suppressed. Since we cannot "un-release" text
// during streaming, we must buffer tokens that potentially match a stop
// sequence until they are confirmed to be either a complete stop sequence (in
// which case they are discarded) or not a stop sequence (in which case they are
// safe to feed to the detokenizer).
class StopTokenStreamFilter {
 public:
  StopTokenStreamFilter(StopTokenDetector* absl_nonnull detector,
                        int num_candidates)
      : detector_(*detector),
        stop_token_buffer_(num_candidates),
        candidate_stopped_(num_candidates, false) {}

  absl::StatusOr<std::vector<std::vector<int>>> ProcessStep(
      const std::vector<std::vector<int>>& step_tokens) {
    int num_candidates = stop_token_buffer_.size();
    std::vector<std::vector<int>> tokens_to_feed(num_candidates);

    ABSL_RETURN_IF_ERROR(detector_.ProcessTokens(step_tokens));

    for (int i = 0; i < num_candidates; ++i) {
      if (candidate_stopped_[i]) {
        tokens_to_feed[i] = {};
        continue;
      }

      // Append new tokens to buffer first.
      stop_token_buffer_[i].insert(stop_token_buffer_[i].end(),
                                   step_tokens[i].begin(),
                                   step_tokens[i].end());

      if (detector_.GetStopTokensFound()[i]) {
        // JUST matched stop token.
        int L = detector_.GetStepsBeforeStopTokens()[i];
        int num_to_feed = stop_token_buffer_[i].size() - L;
        if (num_to_feed > 0) {
          tokens_to_feed[i].assign(stop_token_buffer_[i].begin(),
                                   stop_token_buffer_[i].begin() + num_to_feed);
        } else {
          tokens_to_feed[i] = {};
        }
        stop_token_buffer_[i].clear();  // Discard stop token.
        candidate_stopped_[i] = true;
      } else {
        int max_length = detector_.MaxPartialStopTokenLength(i);
        if (max_length > 0) {
          // Partial match. Keep last `max_length` tokens in buffer.
          int num_to_feed = stop_token_buffer_[i].size() - max_length;
          if (num_to_feed > 0) {
            tokens_to_feed[i].assign(
                stop_token_buffer_[i].begin(),
                stop_token_buffer_[i].begin() + num_to_feed);
            stop_token_buffer_[i].erase(
                stop_token_buffer_[i].begin(),
                stop_token_buffer_[i].begin() + num_to_feed);
          } else {
            tokens_to_feed[i] = {};
          }
        } else {
          // No match. Feed everything.
          tokens_to_feed[i] = std::move(stop_token_buffer_[i]);
          stop_token_buffer_[i].clear();
        }
      }
    }
    return tokens_to_feed;
  }

 private:
  // Detector used to match tokens against stop sequences.
  StopTokenDetector& detector_;
  // Per-candidate buffer holding tokens that may be part of a stop sequence.
  std::vector<std::vector<int>> stop_token_buffer_;
  // Per-candidate flag tracking whether a stop sequence has been encountered.
  std::vector<bool> candidate_stopped_;
};

// A wrapper class to run one step of the decode process, handling both internal
// and external sampling.
class DecodeOneStep {
 private:
  DecodeOneStep(LlmExecutor* absl_nonnull executor,
                Tokenizer* absl_nonnull tokenizer, int num_output_candidates,
                const StopTokenDetector& stop_token_detector,
                std::optional<BenchmarkInfo>& benchmark_info,
                std::optional<Sampler*> sampler,
                std::unique_ptr<CompositeConstraint> composite_constraint,
                std::unique_ptr<ConstrainedDecoder> constrained_decoder,
                std::optional<litert::TensorBuffer> scores_tensor,
                const std::atomic<bool>* cancelled)
      : executor_(*executor),
        tokenizer_(*tokenizer),
        detokenizer_(&tokenizer_, num_output_candidates),
        num_output_candidates_(num_output_candidates),
        sampler_(sampler),
        composite_constraint_(std::move(composite_constraint)),
        constrained_decoder_(std::move(constrained_decoder)),
        benchmark_info_(benchmark_info),
        stop_token_detector_(stop_token_detector),
        stop_token_filter_(&stop_token_detector_, num_output_candidates),
        result_text_(num_output_candidates, ""),
        result_token_ids_(num_output_candidates),
        cancelled_(cancelled) {
    if (scores_tensor.has_value()) {
      scores_tensor_ = std::move(*scores_tensor);
    }
  }

 public:
  static absl::StatusOr<std::unique_ptr<DecodeOneStep>> Create(
      LlmExecutor* absl_nonnull executor, Tokenizer* absl_nonnull tokenizer,
      int num_output_candidates, const StopTokenDetector& stop_token_detector,
      std::optional<BenchmarkInfo>& benchmark_info,
      std::optional<Sampler*> sampler,
      RepetitionPenaltyConfig repetition_penalty_config,
      NoRepeatNgramConfig no_repeat_ngram_config,
      SuppressTokensConfig suppress_tokens_config, Constraint* constraint,
      const std::atomic<bool>* cancelled =
          nullptr  // Add cancelled signal for one decode step (eg.
                   // for diffusion-llm)
  ) {
    ABSL_ASSIGN_OR_RETURN(auto composite_constraint,
                          CompositeConstraint::Create());
    if (repetition_penalty_config.enabled()) {
      ABSL_RETURN_IF_ERROR(composite_constraint->AddConstraint(
          std::make_unique<RepetitionPenaltyConstraint>(
              tokenizer->GetVocabSize(),
              std::move(repetition_penalty_config))));
    }
    if (no_repeat_ngram_config.enabled()) {
      ABSL_RETURN_IF_ERROR(composite_constraint->AddConstraint(
          std::make_unique<NoRepeatNgramConstraint>(
              tokenizer->GetVocabSize(), std::move(no_repeat_ngram_config))));
    }
    if (suppress_tokens_config.enabled()) {
      ABSL_RETURN_IF_ERROR(composite_constraint->AddConstraint(
          std::make_unique<SuppressTokensConstraint>(
              tokenizer->GetVocabSize(), std::move(suppress_tokens_config))));
    }
    if (constraint != nullptr) {
      ABSL_RETURN_IF_ERROR(
          composite_constraint->AddUnownedConstraint(constraint));
    }
    std::unique_ptr<ConstrainedDecoder> constrained_decoder;
    if (!composite_constraint->empty()) {
      constrained_decoder = std::make_unique<ConstrainedDecoder>(
          composite_constraint.get(), num_output_candidates);
    }
    std::optional<litert::TensorBuffer> scores_tensor;
    if (sampler.has_value()) {  // External sampling setup
      LITERT_ASSIGN_OR_RETURN(
          scores_tensor, CreateTensorBuffer<float>({num_output_candidates}));
    }
    return std::unique_ptr<DecodeOneStep>(new DecodeOneStep(
        executor, tokenizer, num_output_candidates, stop_token_detector,
        benchmark_info, sampler, std::move(composite_constraint),
        std::move(constrained_decoder), std::move(scores_tensor), cancelled));
  }

  // Runs one step of the decode process and returns if all stops for all
  // candidates have been found.
  // For external sampling, `decoded_ids` must be provided and will be updated.
  // For internal sampling, `decoded_ids` is ignored.
  absl::StatusOr<bool> Run(
      std::optional<litert::TensorBuffer> decoded_ids = std::nullopt) {
    ABSL_ASSIGN_OR_RETURN(auto token_ids,
                          DecodeAndSample(std::move(decoded_ids)));

    size_t sequence_length = token_ids[0].size();
    for (size_t i = 1; i < token_ids.size(); ++i) {
      RET_CHECK_EQ(token_ids[i].size(), sequence_length)
          << "The current implementation of ProcessTokens() requires that "
             "latest_tokens must contain sequences of the same length.";
    }

    for (int i = 0; i < num_output_candidates_; ++i) {
      result_text_[i].clear();
      result_token_ids_[i].clear();
    }

    for (size_t step = 0; step < sequence_length; ++step) {
      std::vector<std::vector<int>> step_tokens;
      step_tokens.reserve(num_output_candidates_);
      for (int batch = 0; batch < num_output_candidates_; ++batch) {
        step_tokens.push_back({token_ids[batch][step]});
      }

      ABSL_ASSIGN_OR_RETURN(auto tokens_to_feed,
                            stop_token_filter_.ProcessStep(step_tokens));

      ABSL_ASSIGN_OR_RETURN(auto released_outputs,
                            detokenizer_.ProcessStep(tokens_to_feed));

      for (int i = 0; i < num_output_candidates_; ++i) {
        if (!released_outputs[i].text.empty()) {
          result_text_[i] += released_outputs[i].text;
          result_token_ids_[i].insert(result_token_ids_[i].end(),
                                      released_outputs[i].token_ids.begin(),
                                      released_outputs[i].token_ids.end());
        }
      }

      if (sampler_.has_value()) {
        LITERT_ASSIGN_OR_RETURN(scores_span_,
                                ReferTensorBufferAsSpan<float>(scores_tensor_));
      }

      is_first_step_ = false;
      ABSL_ASSIGN_OR_RETURN(bool all_done, stop_token_detector_.AllDone());
      if (all_done) {
        if (step != sequence_length - 1) {
          // we are done before all the tokens are processed, so we need to
          // rollback the processed tokens in executor.
          int diff = sequence_length - step;
          ABSL_ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
          ABSL_RETURN_IF_ERROR(executor_.SetCurrentStep(current_step - diff));
        }
        return true;
      }
    }
    return false;
  }

  // Flushes any remaining withheld text and token IDs from the detokenizer.
  // Overwrites `result_text_` and `result_token_ids_` with only the newly
  // released outputs (incremental delta) from this flush operation.
  absl::StatusOr<std::vector<std::string>> Flush() {
    ABSL_ASSIGN_OR_RETURN(auto released_outputs, detokenizer_.Flush());
    std::vector<std::string> released_texts(num_output_candidates_);
    for (int i = 0; i < num_output_candidates_; ++i) {
      result_text_[i] = released_outputs[i].text;
      result_token_ids_[i] = released_outputs[i].token_ids;
      released_texts[i] = released_outputs[i].text;
    }
    return released_texts;
  }

  absl::Span<float> GetScores() { return scores_span_; }

  // Returns the released text delta for the most recent step or Flush().
  const std::vector<std::string>& GetResultText() const { return result_text_; }

  // Returns the released token IDs delta for the most recent step or Flush().
  const std::vector<std::vector<int>>& GetResultTokenIds() const {
    return result_token_ids_;
  }

  // This function is only supported for external sampling.
  // It computes the log likelihoods for the sampled ids corresponding to the
  // ids of a batch and returns it as a vector of floats.
  // step_input_ids: The ids corresponding to the input text for the batch.
  // decoded_ids: The decoded id tensor buffer in which the sampled ids are
  //              written so that the model uses reference text future step.
  // Returns: A vector of log likelihoods for the sampled ids.
  // TODO: b/499304966 - Add tests for the float16 path.
  absl::StatusOr<std::vector<float>> RunScoreStep(
      const float temperature, const std::vector<int>& step_input_ids,
      litert::TensorBuffer decoded_ids) {
    LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                            decoded_ids.Duplicate());
    const ExecutorInputs inputs(
        ExecutorTextData(std::move(duplicate_decoded_ids)),
        /*vision_data=*/std::nullopt,
        /*audio_data=*/std::nullopt);
    // Decoding section.
    if (benchmark_info_.has_value()) {
      ABSL_RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
    }
    ABSL_ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
    if (benchmark_info_.has_value()) {
      ABSL_RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
    }
    decoded_ids.Write<int>(step_input_ids);
    LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type,
                            output_logits.TensorType());
    auto logits_dims = logits_tensor_type.Layout().Dimensions();
    // Logits dims are {batch, seq, vocab}. For scoring, we expect batch size to
    // be the same as the input batch size, sequence length to be 1, and vocab
    // size to be the same as the tokenizer size.
    RET_CHECK_EQ(logits_dims.size(), 3)
        << "Output logits must have shape [batch, seq, vocab].";
    const int batch_size = step_input_ids.size();
    RET_CHECK_EQ(logits_dims[0], batch_size)
        << "Logits batch size does not match the input batch size.";
    RET_CHECK_EQ(logits_dims[1], 1) << "Scoring expects a single decode step.";

    absl::Span<float> logits_data;
    std::vector<float> logits_data_buffer;
    if (logits_tensor_type.ElementType() == litert::ElementType::Float32) {
      auto logits_data_or = ReferTensorBufferAsSpan<float>(output_logits);
      if (!logits_data_or) {
        LITERT_ASSIGN_OR_RETURN(logits_data_buffer,
                                CopyFromTensorBuffer<float>(output_logits));
        logits_data = absl::MakeSpan(logits_data_buffer);
      } else {
        logits_data = *logits_data_or;
      }
    } else if (logits_tensor_type.ElementType() ==
               litert::ElementType::Float16) {
      LITERT_ASSIGN_OR_RETURN(
          auto logits_data_f16,
          CopyFromTensorBuffer<tflite::half>(output_logits));
      ConvertFp16ToFp32(absl::MakeConstSpan(logits_data_f16),
                        logits_data_buffer);
      logits_data = absl::MakeSpan(logits_data_buffer);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported logits element type for scoring: ",
                       logits_tensor_type.ElementType()));
    }
    RET_CHECK_EQ(logits_data.size(), batch_size * logits_dims[2])
        << "Logits buffer size does not match logits tensor shape.";
    return ComputeLogLikelihood(logits_data, step_input_ids, temperature);
  }

 private:
  // Runs the core decoding and sampling step, for either internal or external
  // sampling. Returns a pointer to the tensor buffer containing the next token
  // IDs.
  absl::StatusOr<std::vector<std::vector<int>>> DecodeAndSample(
      std::optional<litert::TensorBuffer> decoded_ids) {
    if (sampler_) {  // External sampling path
      if (!decoded_ids) {
        return absl::InternalError(
            "decoded_ids must be provided for external sampling.");
      }
      LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                              decoded_ids->Duplicate());
      ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)),
                            std::nullopt, std::nullopt);
      // Update the constraint state only with decode ids.
      // If this is the first step, last_token_ids comes from prefill, therefore
      // should be ignored.
      if (!is_first_step_ && constrained_decoder_ != nullptr) {
        LITERT_ASSIGN_OR_RETURN(auto last_token_ids, decoded_ids->Duplicate());
        ABSL_RETURN_IF_ERROR(constrained_decoder_->UpdateState(last_token_ids));
      }
      // Decoding section.
      if (benchmark_info_.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      ABSL_ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
      if (benchmark_info_.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      // If constraints are active, process the logits based on the current
      // state.
      if (constrained_decoder_ != nullptr) {
        ABSL_RETURN_IF_ERROR(
            constrained_decoder_->ProcessLogits(output_logits));
      }

      // Sampling section.
      if (benchmark_info_.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }
      ABSL_RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(
          output_logits, decoded_ids.value(), &scores_tensor_));
      if (benchmark_info_.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }

      ABSL_ASSIGN_OR_RETURN(auto token_ids, tokenizer_.TensorBufferToTokenIds(
                                                decoded_ids.value()));
      return token_ids;
    } else {  // Internal sampling path
      // Benchmark executor_decode_and_sample section.
      if (benchmark_info_.has_value()) {
        ABSL_RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      std::vector<std::vector<int>> output_tokens;
      auto decode_params = ExecutorDecodeParams();
      // Convey the cancellation token for the decode process.
      decode_params.SetCancelled(cancelled_);
      if (constrained_decoder_ != nullptr) {
        decode_params.SetConstrainedDecoder(constrained_decoder_.get());
      }
      ABSL_ASSIGN_OR_RETURN(output_tokens, executor_.Decode(decode_params));
      if (benchmark_info_.has_value()) {
        ABSL_RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      return output_tokens;
    }
  }

  LlmExecutor& executor_;
  Tokenizer& tokenizer_;
  litert::support::BufferedStreamingDetokenizer detokenizer_;
  const int num_output_candidates_;
  std::optional<Sampler*> sampler_;
  std::unique_ptr<CompositeConstraint> composite_constraint_;
  std::unique_ptr<ConstrainedDecoder> constrained_decoder_;
  std::optional<BenchmarkInfo> benchmark_info_;
  StopTokenDetector stop_token_detector_;
  StopTokenStreamFilter stop_token_filter_;

  // For external sampling.
  // Holds the scores for the output candidates. Dim: {num_output_candidates}
  litert::TensorBuffer scores_tensor_;
  absl::Span<float> scores_span_;

  // Stores the incremental delta (newly released text and token IDs) produced
  // during the most recent `Run()` step or `Flush()` call.
  std::vector<std::string> result_text_;
  std::vector<std::vector<int>> result_token_ids_;

  bool is_first_step_ = true;
  const std::atomic<bool>* cancelled_ = nullptr;
};

}  // namespace

absl::StatusOr<Responses> Prefill(
    LlmExecutor& executor, ExecutorInputs& inputs, bool wait_for_completion,
    std::optional<BenchmarkInfo>& benchmark_info) {
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  ABSL_ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
  RET_CHECK(text_data != nullptr) << "text_data must not be null.";
  LITERT_ASSIGN_OR_RETURN(auto token_id_tensor_type,
                          text_data->GetTokenIds().TensorType());
  auto num_tokens = token_id_tensor_type.Layout().Dimensions().back();
  if (num_tokens >= max_num_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. Exceeding the maximum number of tokens "
        "allowed: ",
        num_tokens, " >= ", max_num_tokens));
  }
  size_t num_token_ids;
  {
    LITERT_ASSIGN_OR_RETURN(
        auto ids_buffer_span,
        ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
    if (ids_buffer_span.empty()) {
      return absl::InternalError("Input token ids are empty.");
    }
    num_token_ids = ids_buffer_span.size();
  }
  ExecutorPrefillParams params;
  // Wait for prefill to complete if benchmark mode is enabled.
  params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimePrefillTurnStart());
  }
  ABSL_RETURN_IF_ERROR(executor.Prefill(inputs, params));
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(num_token_ids));
    auto profile_summary = executor.GetProfileSummary();
    if (profile_summary.ok()) {
      if (profile_summary->empty()) {
        ABSL_LOG(WARNING) << "Prefill profile summary is empty!";
      } else {
        benchmark_info->SetProfileSummary(*profile_summary);
      }
    } else if (profile_summary.status().code() ==
               absl::StatusCode::kFailedPrecondition) {
      ABSL_VLOG(1) << "Profiling is not enabled.";
    } else {
      ABSL_LOG(WARNING) << "Failed to get prefill profile summary: "
                        << profile_summary.status();
    }
  }
  return Responses(TaskState::kDone);
}

absl::StatusOr<Responses> Decode(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    std::optional<BenchmarkInfo>& benchmark_info,
    std::optional<Sampler*> sampler,
    RepetitionPenaltyConfig repetition_penalty_config,
    NoRepeatNgramConfig no_repeat_ngram_config,
    SuppressTokensConfig suppress_tokens_config, Constraint* constraint,
    std::optional<litert::TensorBuffer> decoded_ids,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback,
    std::atomic<bool>* cancelled, int max_output_tokens,
    std::optional<int> thinking_token_budget,
    const std::vector<int>& thinking_end_token_ids,
    const std::vector<int>& thinking_start_token_ids) {
  const bool is_streaming = callback != nullptr;
  const bool is_custom_sampling = sampler.has_value();

  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    // Initialize sampler early if the executor supports it.
    auto* compiled_model_executor =
        dynamic_cast<LlmLiteRtCompiledModelExecutorBase*>(&executor);
    if (compiled_model_executor != nullptr) {
      compiled_model_executor->InitializeSampler().IgnoreError();
    }
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }

  // The final decoded texts for each candidate.
  std::vector<std::string> final_texts(num_output_candidates);
  // The final token IDs for each candidate.
  std::vector<std::vector<int>> final_token_ids(num_output_candidates);
  // The final scores for each candidate.
  std::vector<float> final_scores(num_output_candidates);
  // The accumulated scores for each candidate (for custom sampling).
  std::vector<float> accumulated_scores(num_output_candidates);
  // The number of decoded tokens for each candidate (for custom sampling).
  std::vector<int> num_decoded_tokens(num_output_candidates);

  ABSL_ASSIGN_OR_RETURN(int executor_step_before_decode,
                        executor.GetCurrentStep());
  const int max_num_tokens = TryGetMaxNumTokens(executor);

  std::unique_ptr<Constraint> thinking_budget_constraint;
  int vocab_size = tokenizer.GetTokens().size();

  // Apply the thinking budget constraint whenever thinking is enabled (budget
  // != 0). Even if the budget is -1 (unlimited), the constraint is required to
  // suspend any active user_constraint until the thinking phase ends.
  if (thinking_token_budget.has_value() && *thinking_token_budget != 0) {
    if (thinking_end_token_ids.empty()) {
      ABSL_LOG(WARNING)
          << "Thinking budget is set but thinking_end_token_ids is empty. "
             "Ignoring thinking budget constraint.";
    } else {
      thinking_budget_constraint = std::make_unique<ThinkingBudgetConstraint>(
          constraint, *thinking_token_budget, thinking_start_token_ids,
          thinking_end_token_ids, vocab_size);
      constraint = thinking_budget_constraint.get();
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      auto run_one_step,
      DecodeOneStep::Create(
          &executor, &tokenizer, num_output_candidates, stop_token_detector,
          benchmark_info, sampler, std::move(repetition_penalty_config),
          std::move(no_repeat_ngram_config), std::move(suppress_tokens_config),
          constraint, cancelled));
  while (true) {
    if (cancelled != nullptr && cancelled->load()) {
      if (benchmark_info.has_value()) {
        ABSL_ASSIGN_OR_RETURN(int current_step, executor.GetCurrentStep());
        int num_decode_steps = current_step - executor_step_before_decode;
        // If the process is cancelled, we need to end this benchmark phase.
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(
            num_decode_steps * num_output_candidates));
        auto profile_summary = executor.GetProfileSummary();
        if (profile_summary.ok() && !profile_summary->empty()) {
          benchmark_info->SetProfileSummary(*profile_summary);
        }
      }
      if (is_custom_sampling) {
        // For external sampling, the sampled tokens are provided by the
        // sampler. We must run one prefill to add the last token as pending
        // token in the LLM Executor when cancellation happens.
        LITERT_ASSIGN_OR_RETURN(auto duplicated_decoded_ids,
                                decoded_ids->Duplicate());
        ExecutorInputs inputs;
        inputs.SetTextData(ExecutorTextData(std::move(duplicated_decoded_ids)));
        std::optional<BenchmarkInfo> unused_benchmark_info;
        ABSL_ASSIGN_OR_RETURN(auto current_step, executor.GetCurrentStep());
        ABSL_RETURN_IF_ERROR(executor.SetCurrentStep(current_step - 1));
        auto status = Prefill(executor, inputs, /*wait_for_completion=*/true,
                              unused_benchmark_info);
        if (!status.ok()) {
          return status.status();
        }
      }
      return absl::CancelledError("Process cancelled.");
    }
    std::optional<litert::TensorBuffer> decoded_ids_to_use = std::nullopt;
    if (decoded_ids.has_value()) {
      LITERT_ASSIGN_OR_RETURN(decoded_ids_to_use, decoded_ids->Duplicate());
    }
    absl::StatusOr<bool> all_done =
        run_one_step->Run(std::move(decoded_ids_to_use));
    if (!all_done.ok()) {
      return all_done.status();
    }
    std::vector<std::string> step_texts;
    std::vector<std::vector<int>> step_token_ids;
    std::vector<float> step_scores;
    if (is_streaming) {
      step_texts.resize(num_output_candidates);
      step_token_ids.resize(num_output_candidates);
      step_scores.resize(num_output_candidates);
    }
    bool any_updates = false;
    for (int j = 0; j < num_output_candidates; ++j) {
      std::string output_text = run_one_step->GetResultText()[j];
      if (output_text.empty()) {
        // No output text for this candidate - could be due to
        // 1. early stopping.
        // 2. partial BPE sequence.
        // 3. matching partial stop tokens.
        continue;
      }
      any_updates = true;
      if (is_streaming) {
        step_texts[j] = output_text;
        step_token_ids[j] = run_one_step->GetResultTokenIds()[j];
        if (is_custom_sampling) {
          step_scores[j] = run_one_step->GetScores()[j];
        }
      } else {
        final_texts[j] += output_text;
        final_token_ids[j].insert(final_token_ids[j].end(),
                                  run_one_step->GetResultTokenIds()[j].begin(),
                                  run_one_step->GetResultTokenIds()[j].end());
        if (is_custom_sampling) {
          accumulated_scores[j] += run_one_step->GetScores()[j];
          num_decoded_tokens[j]++;
        }
      }
    }

    if (is_streaming && any_updates) {
      callback(Responses(TaskState::kProcessing, std::move(step_texts),
                         std::move(step_scores), /*token_lengths=*/{},
                         std::move(step_token_ids)));
    }

    ABSL_ASSIGN_OR_RETURN(int current_step, executor.GetCurrentStep());
    int num_decode_steps = current_step - executor_step_before_decode;
    if (ShouldStop(*all_done, benchmark_decode_token_count, num_decode_steps,
                   current_step, max_num_tokens, max_output_tokens)) {
      break;
    }
  }

  {
    ABSL_ASSIGN_OR_RETURN(std::vector<std::string> flushed_texts,
                          run_one_step->Flush());

    if (is_streaming) {
      bool any_updates = false;
      std::vector<std::string> step_texts(num_output_candidates);
      std::vector<std::vector<int>> step_token_ids(num_output_candidates);
      for (int j = 0; j < num_output_candidates; ++j) {
        if (!flushed_texts[j].empty()) {
          any_updates = true;
          step_texts[j] = flushed_texts[j];
          step_token_ids[j] = run_one_step->GetResultTokenIds()[j];
        }
      }
      if (any_updates) {
        callback(Responses(TaskState::kProcessing, std::move(step_texts),
                           /*scores=*/{}, /*token_lengths=*/{},
                           std::move(step_token_ids)));
      }
    } else {
      for (int j = 0; j < num_output_candidates; ++j) {
        if (!flushed_texts[j].empty()) {
          final_texts[j] += flushed_texts[j];
          final_token_ids[j].insert(
              final_token_ids[j].end(),
              run_one_step->GetResultTokenIds()[j].begin(),
              run_one_step->GetResultTokenIds()[j].end());
        }
      }
    }
  }

  int num_decode_steps =
      executor.GetCurrentStep().value() - executor_step_before_decode;
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(
        num_decode_steps * num_output_candidates));
    auto profile_summary = executor.GetProfileSummary();
    if (profile_summary.ok()) {
      if (profile_summary->empty()) {
        ABSL_LOG(WARNING) << "Decode profile summary is empty!";
      } else {
        benchmark_info->SetProfileSummary(*profile_summary);
      }
    } else if (profile_summary.status().code() ==
               absl::StatusCode::kFailedPrecondition) {
      ABSL_VLOG(1) << "Profiling is not enabled.";
    } else {
      ABSL_LOG(WARNING) << "Failed to get decode profile summary: "
                        << profile_summary.status();
    }
  }

  if (is_custom_sampling) {
    // For external sampling, the sampled tokens are provided by the sampler. We
    // must run one prefill to add the stop token as pending token in the LLM
    // Executor when stop condition is met.
    LITERT_ASSIGN_OR_RETURN(auto duplicated_decoded_ids,
                            decoded_ids->Duplicate());
    ExecutorInputs inputs;
    inputs.SetTextData(ExecutorTextData(std::move(duplicated_decoded_ids)));
    std::optional<BenchmarkInfo> unused_benchmark_info;
    ABSL_ASSIGN_OR_RETURN(auto current_step, executor.GetCurrentStep());
    ABSL_RETURN_IF_ERROR(executor.SetCurrentStep(current_step - 1));
    auto status = Prefill(executor, inputs, /*wait_for_completion=*/true,
                          unused_benchmark_info);
    if (!status.ok()) {
      return status.status();
    }
  }

  if (is_streaming) {
    if (executor.GetCurrentStep().value() >= max_num_tokens) {
      return Responses(TaskState::kMaxNumTokensReached);
    }
    return Responses(TaskState::kDone);
  }

  // Finalize scores for non-streaming custom sampling.
  if (is_custom_sampling) {
    for (int j = 0; j < num_output_candidates; ++j) {
      if (num_decoded_tokens[j] > 0) {
        final_scores[j] = accumulated_scores[j] / num_decoded_tokens[j];
      } else {
        final_scores[j] = -std::numeric_limits<float>::infinity();
      }
    }
  }
  TaskState task_state = executor.GetCurrentStep().value() >= max_num_tokens
                             ? TaskState::kMaxNumTokensReached
                             : TaskState::kDone;
  return Responses(std::move(task_state), std::move(final_texts),
                   std::move(final_scores), /*token_lengths=*/{},
                   std::move(final_token_ids));
}

absl::StatusOr<Responses> Score(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const std::vector<absl::string_view>& target_texts, const float temperature,
    litert::TensorBuffer decoded_ids, bool store_token_lengths) {
  const int num_output_candidates = target_texts.size();
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  std::optional<BenchmarkInfo> benchmark_info;
  // Create a dummy StopTokenDetector as it's not used in ScoreCustomSampling.
  StopTokenDetector dummy_stop_token_detector(num_output_candidates);
  ABSL_ASSIGN_OR_RETURN(
      auto run_one_step,
      DecodeOneStep::Create(
          &executor, &tokenizer,
          /*num_output_candidates=*/num_output_candidates,
          dummy_stop_token_detector, benchmark_info,
          /*sampler=*/std::nullopt, RepetitionPenaltyConfig::Default(),
          NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
          /*constraint=*/nullptr));
  std::vector<std::vector<int>> ids_for_each_target_in_batch;
  ids_for_each_target_in_batch.reserve(target_texts.size());
  int max_num_tokens_of_target_texts = 0;
  for (const auto& target : target_texts) {
    ABSL_ASSIGN_OR_RETURN(std::vector<int> ids,
                          tokenizer.TextToTokenIds(target));
    max_num_tokens_of_target_texts =
        std::max(max_num_tokens_of_target_texts, static_cast<int>(ids.size()));
    ids_for_each_target_in_batch.push_back(std::move(ids));
  }
  if (max_num_tokens_of_target_texts >= max_num_tokens) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input token ids are too long. "
                     "Exceeding the maximum number of tokens allowed: ",
                     max_num_tokens_of_target_texts, " >= ", max_num_tokens));
  }

  // The scores for each candidate. The scores are accumulated over the course
  // of the decoding process.
  std::vector<float> scores(num_output_candidates);
  std::vector<std::vector<float>> token_scores(num_output_candidates);
  // We support multiple targets by padding the targets with a null token which
  // does not exist in the vocabulary and thus does not contribute to the
  // perplexity.
  std::vector<int> decoded_ids_for_each_target_in_batch(num_output_candidates,
                                                        0);
  for (int i = 0; i < max_num_tokens_of_target_texts; ++i) {
    for (int j = 0; j < num_output_candidates; ++j) {
      const int size_of_jth_target = ids_for_each_target_in_batch[j].size();
      if (i < size_of_jth_target) {
        decoded_ids_for_each_target_in_batch[j] =
            ids_for_each_target_in_batch[j][i];
      } else {
        // Pad the target with a null token. Ignore the result at this step.
        decoded_ids_for_each_target_in_batch[j] = 0;
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto decoded_ids_copy, decoded_ids.Duplicate());
    ABSL_ASSIGN_OR_RETURN(std::vector<float> step_log_likelihoods,
                          run_one_step->RunScoreStep(
                              temperature, decoded_ids_for_each_target_in_batch,
                              std::move(decoded_ids_copy)));
    for (int j = 0; j < num_output_candidates; ++j) {
      const int size_of_jth_target = ids_for_each_target_in_batch[j].size();
      // Only add the log likelihood of the non-padded tokens to the score.
      if (i < size_of_jth_target) {
        scores[j] += step_log_likelihoods[j];
        token_scores[j].push_back(step_log_likelihoods[j]);
      }
    }
  }
  std::vector<int> token_lengths;
  if (store_token_lengths) {
    // Store the token lengths of the target texts for each candidate into
    // `Responses`. This is optional.
    token_lengths.reserve(num_output_candidates);
    for (int j = 0; j < num_output_candidates; ++j) {
      token_lengths.push_back(ids_for_each_target_in_batch[j].size());
    }
  }
  auto responses = Responses(TaskState::kDone, /*response_texts=*/{},
                             std::move(scores), std::move(token_lengths),
                             std::move(ids_for_each_target_in_batch));
  responses.GetMutableTokenScores() = std::move(token_scores);
  return responses;
}

}  // namespace litert::lm::Tasks
