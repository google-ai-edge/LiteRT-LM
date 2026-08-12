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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_SUPPRESS_TOKENS_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_SUPPRESS_TOKENS_PROCESSOR_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/components/logits_processor/suppress_tokens_config.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

// `SuppressTokensProcessor` modifies logits to penalize specific tokens from
// being generated.
class SuppressTokensProcessor : public LogitsProcessor {
 public:
  // Creates a SuppressTokensProcessor for a given vocabulary size and batch
  // size.
  //
  // @param batch_size The number of sequences processed in parallel.
  // @param vocab_size The vocabulary size of the model.
  // @param config The configuration containing tokens to suppress.
  explicit SuppressTokensProcessor(int batch_size, int vocab_size,
                                   SuppressTokensConfig config);

  ~SuppressTokensProcessor() override = default;

  absl::Status ProcessLogits(::litert::TensorBuffer& logits) override;

  absl::Status ProcessLogits(
      absl::Span<float> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status ProcessLogits(
      absl::Span<tflite::half> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims) override;

  absl::Status UpdateState(
      const ::litert::TensorBuffer& next_token_ids) override;

  absl::Status UpdateState(absl::Span<const int> next_token_ids) override;

 private:
  // Template implementation of ProcessLogits to support different numeric types
  // for logits.
  template <typename T>
  absl::Status ProcessLogitsImpl(
      absl::Span<T> logits,
      absl::Span<const ::litert::Layout::Dim> logits_dims);

  const int batch_size_;
  const int vocab_size_;
  const SuppressTokensConfig config_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LOGITS_PROCESSOR_SUPPRESS_TOKENS_PROCESSOR_H_
