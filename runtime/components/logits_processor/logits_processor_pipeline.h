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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_LOGITS_PROCESSOR_PIPELINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_LOGITS_PROCESSOR_PIPELINE_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/constrained_decoding/constrained_decoder.h"
#include "runtime/components/logits_processor/constrained_decoding/constraint.h"
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/components/logits_processor/suppress_tokens_config.h"

namespace litert::lm {

// Struct bundling individual logits processor configs for constructing a
// pipeline.
struct LogitsProcessorPipelineConfig {
  RepetitionPenaltyConfig repetition_penalty_config =
      RepetitionPenaltyConfig::Default();
  NoRepeatNgramConfig no_repeat_ngram_config = NoRepeatNgramConfig::Default();
  SuppressTokensConfig suppress_tokens_config = SuppressTokensConfig::Default();
  Constraint* constraint = nullptr;
};

// Manages a pipeline of LogitsProcessors, executing ProcessLogits and
// UpdateState operations sequentially across all registered processors. Handles
// CPU vs GPU host buffer transfers automatically and efficiently.
class LogitsProcessorPipeline {
 public:
  LogitsProcessorPipeline() = default;

  LogitsProcessorPipeline(LogitsProcessorPipeline&&) = default;
  LogitsProcessorPipeline& operator=(LogitsProcessorPipeline&&) = default;
  LogitsProcessorPipeline(const LogitsProcessorPipeline&) = delete;
  LogitsProcessorPipeline& operator=(const LogitsProcessorPipeline&) = delete;

  // Constructs a pipeline by creating owned processors from individual configs.
  LogitsProcessorPipeline(int batch_size, int vocab_size,
                          LogitsProcessorPipelineConfig config);

  // Constructs a pipeline taking ownership of LogitsProcessor instances.
  explicit LogitsProcessorPipeline(
      std::vector<std::unique_ptr<LogitsProcessor>> processors);

  // Adds an owned logits processor to the pipeline.
  void AddProcessor(std::unique_ptr<LogitsProcessor> processor);

  // Processes logits in-place across all processors in the pipeline.
  // Automatically handles GPU-to-CPU host copy and writeback if logits buffer
  // is on device memory.
  absl::Status ProcessLogits(::litert::TensorBuffer& output_logits) const;

  // Updates state across all processors in the pipeline.
  absl::Status UpdateState(const ::litert::TensorBuffer& next_token_ids);
  absl::Status UpdateState(absl::Span<const int> next_token_ids);

  // Returns the constraint decoder if present in the pipeline.
  // This is required for backward compatibility with the artisanal executors.
  ConstrainedDecoder* GetConstraintDecoder() const;

  bool empty() const { return processors_.empty(); }
  size_t size() const { return processors_.size(); }

 private:
  // Sequential list of active owned processors.
  std::vector<std::unique_ptr<LogitsProcessor>> processors_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_LOGITS_PROCESSOR_PIPELINE_H_
