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
#include <type_traits>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_lm_engine_runner.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/omni_engine.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/tokenizer/tokenizer.h"

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
  // If true, forces FP32 precision on GPU instead of FP16.
  bool force_fp32_on_gpu = false;
  // Patterns for tensor names to be declared as external tensors for GPU
  // (ml_drift).
  std::vector<std::string> external_tensor_patterns;
};

// Parses a backend name ("cpu", "gpu", "npu", case-insensitive) into
// `OmniEngine::Settings::Backend`.
//
// args
// - backend_str: Backend string identifier.
//
// returns
// - Corresponding `OmniEngine::Settings::Backend` on success, or
//   `InvalidArgumentError` if unsupported.
absl::StatusOr<OmniEngine::Settings::Backend> ParseOmniBackend(
    absl::string_view backend_str);

// Converts `OmniEngine::Settings::Backend` to `lm::Backend`.
//
// args
// - backend: `OmniEngine::Settings::Backend` enum value.
//
// returns
// - Corresponding `lm::Backend` enum value.
lm::Backend ToLmBackend(OmniEngine::Settings::Backend backend);

// Converts `OmniEngine::Settings` to `ModelOptions` for LM runners.
//
// args
// - settings: `OmniEngine::Settings` containing model_path, cache_dir, backend,
//   and num_threads.
//
// returns
// - Populated `ModelOptions` instance.
ModelOptions ToLmModelOptions(const OmniEngine::Settings& settings);

// Creates a tokenizer from `lm_resources` if non-null, or from
// `tokenizer_path` otherwise.
//
// args
// - lm_resources: Optional opened `lm::ModelResources` container.
// - tokenizer_path: Fallback filesystem path to a tokenizer file.
//
// returns
// - Unique pointer to `Tokenizer` on success, or error status on failure.
absl::StatusOr<std::unique_ptr<::litert::support::Tokenizer>> CreateTokenizer(
    lm::ModelResources* absl_nullable lm_resources,
    absl::string_view tokenizer_path);

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

// Downloads a file from `url` to `target_path` using curl.
//
// args
// - url: Remote URL to download from.
// - target_path: Destination filesystem path.
//
// returns
// - absl::OkStatus() on success, or error status on failure.
absl::Status DownloadFileWithCurl(absl::string_view url,
                                  absl::string_view target_path);

// Resolves a `.litertlm` container file path from `model_folder`. If
// `model_folder` is a `.litertlm` regular file, returns it directly; if it is a
// directory, returns the first `.litertlm` regular file inside it; otherwise
// returns an empty string.
//
// args
// - model_folder: Path to a `.litertlm` file or directory.
//
// returns
// - Resolved `.litertlm` file path, or empty string if none is found.
std::string ResolveLitertLmPath(absl::string_view model_folder);

// Creates a shared lm::ModelResources from a .litertlm container file path.
//
// args
// - model_path: Path to the `.litertlm` container file.
//
// returns
// - Shared pointer to `lm::ModelResources` on success, or error status on
//   failure.
absl::StatusOr<std::shared_ptr<lm::ModelResources>> CreateLmModelResources(
    absl::string_view model_path);

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

// Compiles a `CompiledModel` for `model_type` from `lm_resources` if non-null,
// or from `settings.model_path` otherwise.
//
// args
// - settings: `OmniEngine::Settings` containing model_path and cache_dir.
// - env: LiteRT environment instance.
// - options: Pre-configured LiteRT compilation options.
// - lm_resources: Optional opened `lm::ModelResources` container.
// - model_type: Wire-string or enum model type identifier within the
//   `.litertlm` container.
//
// returns
// - Unique pointer to `CompiledModel` on success, or error status on failure.
absl::StatusOr<std::unique_ptr<CompiledModel>> CreateCompiledModel(
    const OmniEngine::Settings& settings, Environment& env, Options& options,
    lm::ModelResources* absl_nullable lm_resources,
    absl::string_view model_type);

template <typename ModelTypeT,
          typename = std::enable_if_t<std::is_enum_v<ModelTypeT>>>
absl::StatusOr<std::unique_ptr<CompiledModel>> CreateCompiledModel(
    const OmniEngine::Settings& settings, Environment& env, Options& options,
    lm::ModelResources* absl_nullable lm_resources, ModelTypeT model_type) {
  return CreateCompiledModel(settings, env, options, lm_resources,
                             lm::TfLiteModelTypeToWireString(model_type));
}

// Creates and compiles a LiteRT model from an in-memory buffer.
//
// args
// - env: LiteRT environment instance.
// - options: ModelOptions containing cache_dir, backend, etc.
// - model_buffer: In-memory TFLite flatbuffer bytes.
// - model_name: Logical model name used for cache naming / logging.
//
// returns
// - CompiledModel instance on success, or error status on failure.
absl::StatusOr<CompiledModel> CreateCompiledModelFromBuffer(
    Environment& env, const ModelOptions& options,
    absl::string_view model_buffer, absl::string_view model_name);

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
// `LiteRtLmRunner` exposes `DecodeLogits()` for CPU-side decoding (e.g., via
// `LmDecoder`), so `sampler_backend` is always configured to `lm::Backend::CPU`
// to preserve the model's FP32 output logits buffer even when `options.backend`
// is `lm::Backend::GPU` (which runs inference in FP16 precision by default).
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
