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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_RUNNER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::omni {

// Interface for running model signatures.
class LiteRtRunner {
 public:
  virtual ~LiteRtRunner() = default;

  virtual absl::StatusOr<std::vector<::litert::TensorBuffer>>
  CreateInputBuffers(absl::string_view signature_name) = 0;

  virtual absl::StatusOr<std::vector<::litert::TensorBuffer>>
  CreateOutputBuffers(absl::string_view signature_name) = 0;

  virtual absl::Status Run(
      absl::string_view signature_name,
      absl::Span<const ::litert::TensorBuffer> input_buffers,
      absl::Span<const ::litert::TensorBuffer> output_buffers) = 0;
};

// Implementation of LiteRtRunner wrapping LiteRT CompiledModel.
class LiteRtRunnerImpl : public LiteRtRunner {
 public:
  explicit LiteRtRunnerImpl(
      ::litert::CompiledModel* absl_nonnull compiled_model);
  explicit LiteRtRunnerImpl(
      std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model);

  ~LiteRtRunnerImpl() override = default;

  absl::StatusOr<std::vector<::litert::TensorBuffer>> CreateInputBuffers(
      absl::string_view signature_name) override;

  absl::StatusOr<std::vector<::litert::TensorBuffer>> CreateOutputBuffers(
      absl::string_view signature_name) override;

  absl::Status Run(
      absl::string_view signature_name,
      absl::Span<const ::litert::TensorBuffer> input_buffers,
      absl::Span<const ::litert::TensorBuffer> output_buffers) override;

 private:
  std::unique_ptr<::litert::CompiledModel> owned_compiled_model_;
  ::litert::CompiledModel* const absl_nonnull compiled_model_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_RUNNER_H_
