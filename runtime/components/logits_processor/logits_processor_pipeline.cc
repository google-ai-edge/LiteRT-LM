// Copyright 2026 The ODML Authors.
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

#include "runtime/components/logits_processor/logits_processor_pipeline.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/components/logits_processor/no_repeat_ngram_processor.h"
#include "runtime/components/logits_processor/repetition_penalty_processor.h"
#include "runtime/components/logits_processor/suppress_tokens_processor.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

LogitsProcessorPipeline::LogitsProcessorPipeline(
    int batch_size, int vocab_size, LogitsProcessorPipelineConfig config) {
  if (config.repetition_penalty_config.enabled()) {
    processors_.push_back(std::make_unique<RepetitionPenaltyProcessor>(
        batch_size, vocab_size, std::move(config.repetition_penalty_config)));
  }
  if (config.no_repeat_ngram_config.enabled()) {
    processors_.push_back(std::make_unique<NoRepeatNgramProcessor>(
        batch_size, vocab_size, std::move(config.no_repeat_ngram_config)));
  }
  if (config.suppress_tokens_config.enabled()) {
    processors_.push_back(std::make_unique<SuppressTokensProcessor>(
        batch_size, vocab_size, std::move(config.suppress_tokens_config)));
  }
  if (config.constraint != nullptr) {
    processors_.push_back(
        std::make_unique<ConstrainedDecoder>(config.constraint, batch_size));
  }
}

LogitsProcessorPipeline::LogitsProcessorPipeline(
    std::vector<std::unique_ptr<LogitsProcessor>> processors)
    : processors_(std::move(processors)) {}

void LogitsProcessorPipeline::AddProcessor(
    std::unique_ptr<LogitsProcessor> processor) {
  if (processor != nullptr) {
    processors_.push_back(std::move(processor));
  }
}

absl::Status LogitsProcessorPipeline::ProcessLogits(
    TensorBuffer& output_logits) const {
  if (processors_.empty()) {
    return absl::OkStatus();
  }

  LITERT_ASSIGN_OR_RETURN(auto buffer_type, output_logits.BufferType());
  if (buffer_type == TensorBufferType::kHostMemory) {
    for (const auto& processor : processors_) {
      ABSL_RETURN_IF_ERROR(processor->ProcessLogits(output_logits));
    }
    return absl::OkStatus();
  }

  // For GPU/non-host buffers, copy to CPU, process in-place, write back once.
  LITERT_ASSIGN_OR_RETURN(RankedTensorType logits_tensor_type,
                          output_logits.TensorType());
  auto element_type = logits_tensor_type.ElementType();

  if (element_type == ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto logits_vector,
                            CopyFromTensorBuffer<float>(output_logits));
    auto dims = logits_tensor_type.Layout().Dimensions();
    for (const auto& processor : processors_) {
      ABSL_RETURN_IF_ERROR(processor->ProcessLogits(
          absl::MakeSpan(logits_vector.data(), logits_vector.size()), dims));
    }
    LITERT_RETURN_IF_ERROR(output_logits.Write(
        absl::MakeConstSpan(logits_vector.data(), logits_vector.size())));
    return absl::OkStatus();
  } else if (element_type == ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(auto logits_vector,
                            CopyFromTensorBuffer<tflite::half>(output_logits));
    auto dims = logits_tensor_type.Layout().Dimensions();
    for (const auto& processor : processors_) {
      ABSL_RETURN_IF_ERROR(processor->ProcessLogits(
          absl::MakeSpan(logits_vector.data(), logits_vector.size()), dims));
    }
    LITERT_RETURN_IF_ERROR(output_logits.Write(
        absl::MakeConstSpan(logits_vector.data(), logits_vector.size())));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "Output logits are not in float32 or float16 type.");
}

absl::Status LogitsProcessorPipeline::UpdateState(
    const TensorBuffer& next_token_ids) {
  for (const auto& processor : processors_) {
    ABSL_RETURN_IF_ERROR(processor->UpdateState(next_token_ids));
  }
  return absl::OkStatus();
}

absl::Status LogitsProcessorPipeline::UpdateState(
    absl::Span<const int> next_token_ids) {
  for (const auto& processor : processors_) {
    ABSL_RETURN_IF_ERROR(processor->UpdateState(next_token_ids));
  }
  return absl::OkStatus();
}

ConstrainedDecoder* LogitsProcessorPipeline::GetConstraintDecoder() const {
  for (const auto& processor : processors_) {
    if (auto* constraint_decoder =
            dynamic_cast<ConstrainedDecoder*>(processor.get());
        constraint_decoder != nullptr) {
      return constraint_decoder;
    }
  }
  return nullptr;
}

}  // namespace litert::lm
