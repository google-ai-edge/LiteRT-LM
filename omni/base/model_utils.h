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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_UTILS_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_engine_runner.h"
#include "omni/base/litert_lm_runner.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"

namespace litert::omni {

// Creates ExecutorInputs from a text token IDs TensorBuffer.
//
// args
// - buffer: TensorBuffer containing token IDs.
//
// returns
// - Populated ExecutorInputs with ExecutorTextData on success, or error status.
absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithText(
    const TensorBuffer& buffer);

// Creates ExecutorInputs from an audio embeddings TensorBuffer.
//
// args
// - buffer: TensorBuffer containing audio embeddings.
//
// returns
// - Populated ExecutorInputs with ExecutorAudioData on success, or error
// status.
absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithAudio(
    const TensorBuffer& buffer);

// Creates ExecutorInputs from a vision embeddings TensorBuffer.
//
// args
// - buffer: TensorBuffer containing vision embeddings.
//
// returns
// - Populated ExecutorInputs with ExecutorVisionData on success, or error
// status.
absl::StatusOr<lm::ExecutorInputs> CreateExecutorInputsWithVision(
    const TensorBuffer& buffer);

// Options for loading and compiling LiteRT models across Omni pipelines.
struct ModelOptions {
  // Directory containing model files on the filesystem.
  std::string model_dir;
  // Directory for storing backend cache (e.g., XNNPACK or GPU cache).
  std::string cache_dir;
  // Execution backend (e.g., CPU, GPU).
  lm::Backend backend = lm::Backend::CPU;
  // Number of threads for CPU execution.
  int num_threads = 4;
  // Patterns for tensor names to be declared as external tensors for GPU
  // (ml_drift).
  std::vector<std::string> external_tensor_patterns;
};

// Checks if a file exists and is readable at the specified path.
//
// args
// - path: Absolute or relative file path to check.
//
// returns
// - absl::OkStatus() if file is readable, or absl::NotFoundError otherwise.
absl::Status CheckFileReadable(absl::string_view path);

// Reads content of a file from the filesystem.
//
// args
// - path: Absolute or relative file path to read.
//
// returns
// - String containing file binary data on success, or error status.
absl::StatusOr<std::string> LoadFile(absl::string_view path);

// Reads content of a file from directory and filename.
//
// args
// - model_dir: Directory containing the file.
// - filename: Relative filename to load.
//
// returns
// - String containing file binary data on success, or error status.
absl::StatusOr<std::string> LoadFile(absl::string_view model_dir,
                                     absl::string_view filename);

// Creates and compiles a LiteRT model from filesystem path.
//
// args
// - env: LiteRT environment instance.
// - options: ModelOptions containing model directory, cache_dir, backend, etc.
// - model_filename: Model file name to load.
//
// returns
// - CompiledModel instance on success, or error status on failure.
absl::StatusOr<CompiledModel> CreateCompiledModel(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename);

// Creates and compiles a LiteRT model for use with StatefulLiteRtRunner,
// automatically declaring the recurrent state tensors as external tensors for
// GPU (ml_drift).
//
// args
// - env: LiteRT environment instance.
// - options: ModelOptions containing model directory, cache_dir, backend, etc.
// - model_filename: Model file name to load.
// - signature_name: Name of the stateful signature (or empty for default).
// - num_non_state_inputs: Number of leading non-state inputs in the signature.
// - num_non_state_outputs: Number of leading non-state outputs in the
//   signature.
//
// returns
// - CompiledModel instance on success, or error status on failure.
absl::StatusOr<CompiledModel> CreateCompiledModelForStatefulRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename, absl::string_view signature_name,
    size_t num_non_state_inputs, size_t num_non_state_outputs);

// Creates a LiteRtLmRunner from model directory and filename.
//
// args
// - env: LiteRT environment instance.
// - options: ModelOptions containing model directory, cache_dir, backend, etc.
// - model_filename: Model file name to load.
//
// returns
// - Unique pointer to LiteRtLmRunner on success, or error status on failure.
absl::StatusOr<std::unique_ptr<LiteRtLmRunner>> CreateLmRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename);

// Creates a LiteRtLmEngineRunner backed by litert::lm::Engine.
//
// args
// - env: LiteRT environment instance.
// - options: ModelOptions containing model directory, cache_dir, backend, etc.
// - model_filename: Model file name to load.
//
// returns
// - Unique pointer to LiteRtLmEngineRunner on success, or error status.
absl::StatusOr<std::unique_ptr<LiteRtLmEngineRunner>> CreateLmEngineRunner(
    Environment& env, const ModelOptions& options,
    absl::string_view model_filename);

// Resolves the input tensor index corresponding to a signature tensor name.
//
// args
// - model: CompiledModel instance.
// - tensor_name: Name of the input tensor.
//
// returns
// - Resolved tensor index, or NotFoundError if not present.
absl::StatusOr<size_t> ResolveInputIndex(const CompiledModel& model,
                                         absl::string_view tensor_name);

// Resolves the output tensor index corresponding to a signature tensor name.
//
// args
// - model: CompiledModel instance.
// - tensor_name: Name of the output tensor.
//
// returns
// - Resolved tensor index, or NotFoundError if not present.
absl::StatusOr<size_t> ResolveOutputIndex(const CompiledModel& model,
                                          absl::string_view tensor_name);

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_MODEL_UTILS_H_
