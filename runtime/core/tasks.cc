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
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/scoring_cpu_util.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT
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

// A wrapper class to run one step of the decode process, handling both internal
// and external sampling.
class DecodeOneStep {
 public:
  DecodeOneStep(LlmExecutor* absl_nonnull executor,
                Tokenizer* absl_nonnull tokenizer, int num_output_candidates,
                const StopTokenDetector& stop_token_detector,
                std::optional<BenchmarkInfo>& benchmark_info,
                std::optional<Sampler*> sampler, Constraint* constraint)
      : executor_(*executor),
        tokenizer_(*tokenizer),
        num_output_candidates_(num_output_candidates),
        sampler_(sampler),
        benchmark_info_(benchmark_info),
        stop_token_detector_(stop_token_detector) {
    if (constraint != nullptr) {
      constrained_decoder_ = std::make_unique<ConstrainedDecoder>(
          constraint, num_output_candidates_);
    }
    if (sampler_.has_value()) {  // External sampling setup
      auto scores_tensor = CreateTensorBuffer<float>({num_output_candidates_});
      scores_tensor_ = std::move(*scores_tensor);
    }
    result_text_ = std::vector<std::string>(num_output_candidates_, "");
    result_token_ids_ = std::vector<std::vector<int>>(num_output_candidates_);
    bpe_partial_token_ids_ =
        std::vector<std::vector<int>>(num_output_candidates_);
    pending_stop_tokens_ =
        std::vector<std::queue<std::string>>(num_output_candidates_);
    pending_stop_token_ids_ =
        std::vector<std::queue<std::vector<int>>>(num_output_candidates_);
    num_buffered_tokens_ = std::vector<int>(num_output_candidates_, 0);
  }

  // Runs one step of the decode process and returns if all stops for all
  // candidates have been found.
  // For external sampling, `decoded_ids` must be provided and will be updated.
  // For internal sampling, `decoded_ids` is ignored.
  absl::StatusOr<bool> Run(
      std::optional<litert::TensorBuffer> decoded_ids = std::nullopt) {
    ASSIGN_OR_RETURN(auto token_ids, DecodeAndSample(std::move(decoded_ids)));

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

      // Regardless of BPE, we always process the next tokens to detect stop
      // tokens.
      RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(step_tokens));

      // Merge BPE partial token ids with the next token ids if any.
      ASSIGN_OR_RETURN(step_tokens, tokenizer_.MergeTokenIds(
                                        bpe_partial_token_ids_, step_tokens));

      auto decoded_result =
          tokenizer_.TokenIdsToTexts(num_output_candidates_, step_tokens);
      for (int i = 0; i < num_output_candidates_; ++i) {
        if (Tokenizer::IsIncompleteBpeSequence(decoded_result.value()[i])) {
          bpe_partial_token_ids_[i] = step_tokens[i];
        } else if (!stop_token_detector_.GetStopTokensFound()[i]) {
          bpe_partial_token_ids_[i].clear();

          // Handle partial stop tokens.
          int max_length = stop_token_detector_.MaxPartialStopTokenLength(i);
          if (max_length > 0) {
            pending_stop_tokens_[i].push(decoded_result.value()[i].value());
            pending_stop_token_ids_[i].push(step_tokens[i]);
            num_buffered_tokens_[i] += step_tokens[i].size();
          }
          // We only need the latest max_length tokens for partial stop tokens.
          // Add the extra ones to the result text and we could keep only the
          // latest max_length stop tokens in the queue.
          while (num_buffered_tokens_[i] > max_length) {
            result_text_[i] += pending_stop_tokens_[i].front();
            pending_stop_tokens_[i].pop();

            auto& ids = pending_stop_token_ids_[i].front();
            result_token_ids_[i].insert(result_token_ids_[i].end(), ids.begin(),
                                        ids.end());
            num_buffered_tokens_[i] -= ids.size();
            pending_stop_token_ids_[i].pop();
          }

          // No partial stop token is found - add the current token to the
          // result text directly - this is the most common case.
          if (max_length == 0) {
            result_text_[i] += decoded_result.value()[i].value();
            result_token_ids_[i].insert(result_token_ids_[i].end(),
                                        step_tokens[i].begin(),
                                        step_tokens[i].end());
          }
        }
      }

      if (sampler_.has_value()) {
        LITERT_ASSIGN_OR_RETURN(scores_span_,
                                ReferTensorBufferAsSpan<float>(scores_tensor_));
      }

      is_first_step_ = false;
      ASSIGN_OR_RETURN(bool all_done, stop_token_detector_.AllDone());
      if (all_done) {
        if (step != sequence_length - 1) {
          // we are done before all the tokens are processed, so we need to
          // rollback the processed tokens in executor.
          int diff = sequence_length - step;
          ASSIGN_OR_RETURN(int current_step, executor_.GetCurrentStep());
          RETURN_IF_ERROR(executor_.SetCurrentStep(current_step - diff));
        }
        return true;
      }
    }
    return false;
  }

  absl::Span<float> GetScores() { return scores_span_; }

  const std::vector<std::string>& GetResultText() const { return result_text_; }

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
      RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
    }
    ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
    if (benchmark_info_.has_value()) {
      RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
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
      // Update constraint state only with decode ids.
      // If this is the first step, last_token_ids comes from prefill, therefore
      // should be ignored.
      if (!is_first_step_ && constrained_decoder_) {
        LITERT_ASSIGN_OR_RETURN(auto last_token_ids, decoded_ids->Duplicate());
        RETURN_IF_ERROR(
            constrained_decoder_->UpdateConstraintState(last_token_ids));
      }
      // Decoding section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      // If constrained decoding is enabled, masks the logits based on the
      // constraint state.
      if (constrained_decoder_) {
        RETURN_IF_ERROR(constrained_decoder_->MaskLogits(output_logits));
      }

      // Samping section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }
      RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(
          output_logits, decoded_ids.value(), &scores_tensor_));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }

      ASSIGN_OR_RETURN(auto token_ids,
                       tokenizer_.TensorBufferToTokenIds(decoded_ids.value()));
      return token_ids;
    } else {  // Internal sampling path
      // Benchmark executor_decode_and_sample section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      std::vector<std::vector<int>> output_tokens;
      if (constrained_decoder_) {
        auto decode_params = ExecutorDecodeParams();
        decode_params.SetConstraintDecoder(constrained_decoder_.get());
        ASSIGN_OR_RETURN(output_tokens, executor_.Decode(decode_params));
      } else {
        ASSIGN_OR_RETURN(output_tokens, executor_.Decode());
      }
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      return output_tokens;
    }
  }

  LlmExecutor& executor_;
  Tokenizer& tokenizer_;
  const int num_output_candidates_;
  std::optional<Sampler*> sampler_;
  std::unique_ptr<ConstrainedDecoder> constrained_decoder_;
  std::optional<BenchmarkInfo> benchmark_info_;
  StopTokenDetector stop_token_detector_;

  // For external sampling.
  // Holds the scores for the output candidates. Dim: {num_output_candidates}
  litert::TensorBuffer scores_tensor_;
  absl::Span<float> scores_span_;

  // Common state
  std::vector<std::vector<int>> bpe_partial_token_ids_;
  std::vector<std::queue<std::string>> pending_stop_tokens_;
  std::vector<std::queue<std::vector<int>>> pending_stop_token_ids_;
  std::vector<int> num_buffered_tokens_;
  std::vector<std::string> result_text_;
  std::vector<std::vector<int>> result_token_ids_;

  bool is_first_step_ = true;
};

}  // namespace

absl::StatusOr<Responses> Prefill(
    LlmExecutor& executor, ExecutorInputs& inputs, bool wait_for_completion,
    std::optional<BenchmarkInfo>& benchmark_info) {
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
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
  LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span, ReferTensorBufferAsSpan<int>(
                                                    text_data->GetTokenIds()));
  if (ids_buffer_span.empty()) {
    return absl::InternalError("Input token ids are empty.");
  }
  ExecutorPrefillParams params;
  // Wait for prefill to complete if benchmark mode is enabled.
  params.SetWaitForCompletion(wait_for_completion | benchmark_info.has_value());
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnStart());
  }
  RETURN_IF_ERROR(executor.Prefill(inputs, params));
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(ids_buffer_span.size()));
  }
  return Responses(TaskState::kDone);
}

absl::StatusOr<Responses> Decode(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    std::optional<BenchmarkInfo>& benchmark_info,
    std::optional<Sampler*> sampler, Constraint* constraint,
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
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
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

  ASSIGN_OR_RETURN(int executor_step_before_decode, executor.GetCurrentStep());
  const int max_num_tokens = TryGetMaxNumTokens(executor);

  std::unique_ptr<Constraint> thinking_budget_constraint;
  int vocab_size = tokenizer.GetTokens().size();
  if (thinking_token_budget.has_value() && *thinking_token_budget != 0) {
    thinking_budget_constraint = std::make_unique<ThinkingBudgetConstraint>(
        constraint, *thinking_token_budget, thinking_start_token_ids,
        thinking_end_token_ids, vocab_size);
    constraint = thinking_budget_constraint.get();
  }

  DecodeOneStep run_one_step(&executor, &tokenizer, num_output_candidates,
                             stop_token_detector, benchmark_info, sampler,
                             constraint);
  while (true) {
    if (cancelled != nullptr && cancelled->load()) {
      if (benchmark_info.has_value()) {
        ASSIGN_OR_RETURN(int current_step, executor.GetCurrentStep());
        int num_decode_steps = current_step - executor_step_before_decode;
        // If the process is cancelled, we need to end this benchmark phase.
        RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(
            num_decode_steps * num_output_candidates));
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
        ASSIGN_OR_RETURN(auto current_step, executor.GetCurrentStep());
        RETURN_IF_ERROR(executor.SetCurrentStep(current_step - 1));
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
        run_one_step.Run(std::move(decoded_ids_to_use));
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
      std::string output_text = run_one_step.GetResultText()[j];
      if (output_text.empty()) {
        // No output text for this candidate - could be due to
        // 1. early stopping.
        // 2. partial BPE sequence.
        // 3. matching partial stop tokens.
        continue;
      }
      any_updates = true;
      // The tokenizer may return a token with a special character "▁" that
      // should be replaced with a space.
      std::string result_text = absl::StrReplaceAll(output_text, {{"▁", " "}});
      if (is_streaming) {
        step_texts[j] = result_text;
        step_token_ids[j] = run_one_step.GetResultTokenIds()[j];
        if (is_custom_sampling) {
          step_scores[j] = run_one_step.GetScores()[j];
        }
      } else {
        final_texts[j] += result_text;
        final_token_ids[j].insert(final_token_ids[j].end(),
                                  run_one_step.GetResultTokenIds()[j].begin(),
                                  run_one_step.GetResultTokenIds()[j].end());
        if (is_custom_sampling) {
          accumulated_scores[j] += run_one_step.GetScores()[j];
          num_decoded_tokens[j]++;
        }
      }
    }

    if (is_streaming && any_updates) {
      callback(Responses(TaskState::kProcessing, std::move(step_texts),
                         std::move(step_scores), /*token_lengths=*/{},
                         std::move(step_token_ids)));
    }

    ASSIGN_OR_RETURN(int current_step, executor.GetCurrentStep());
    int num_decode_steps = current_step - executor_step_before_decode;
    if (ShouldStop(*all_done, benchmark_decode_token_count, num_decode_steps,
                   current_step, max_num_tokens, max_output_tokens)) {
      break;
    }
  }

  int num_decode_steps =
      executor.GetCurrentStep().value() - executor_step_before_decode;
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decode_steps *
                                                      num_output_candidates));
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
    ASSIGN_OR_RETURN(auto current_step, executor.GetCurrentStep());
    RETURN_IF_ERROR(executor.SetCurrentStep(current_step - 1));
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
  DecodeOneStep run_one_step(&executor, &tokenizer,
                             /*num_output_candidates=*/num_output_candidates,
                             dummy_stop_token_detector, benchmark_info,
                             /*sampler=*/std::nullopt,
                             /*constraint=*/nullptr);
  std::vector<std::vector<int>> ids_for_each_target_in_batch;
  ids_for_each_target_in_batch.reserve(target_texts.size());
  int max_num_tokens_of_target_texts = 0;
  for (const auto& target : target_texts) {
    ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer.TextToTokenIds(target));
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
    ASSIGN_OR_RETURN(std::vector<float> step_log_likelihoods,
                     run_one_step.RunScoreStep(
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
