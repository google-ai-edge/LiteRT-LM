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
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni {

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
