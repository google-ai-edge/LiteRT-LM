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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_LITERT_LM_LIB_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_LITERT_LM_LIB_H_

#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// Configuration settings for running or benchmarking the LiteRT-LM embedding
// pipeline. Mirrors the command-line flags declared in
// `embedding_litert_lm_main.cc` and `shared_flags.h`.
struct EmbeddingLiteRtLmSettings {
  std::string backend = "cpu";
  std::optional<std::string> vision_backend = std::nullopt;
  std::optional<std::string> audio_backend = std::nullopt;
  std::string model_path;
  std::string input_prompt;
  std::string input_prompt_file;
  std::string image_path;
  std::string audio_path;
  std::string output_embedding_path;
  std::string compare_embedding_path;
  bool normalize = true;
  bool use_mmap = false;
  std::string dispatch_library_dir;
  int num_warmup = 2;
  int num_iterations = 10;
  std::string input_overflow_strategy = "truncate";
  int max_input_length = 0;
  int min_input_length = 0;
  std::string activation_data_type;
  bool benchmark = false;
  int benchmark_prefill_tokens = 0;
  int visual_token_budget = 0;
  bool report_peak_memory_footprint = false;
  int num_cpu_threads = 0;
};

// Runs the embedding pipeline with the given `settings` and reports the result.
//
// The human readable report is always written to stdout. When `report_out` is
// non-null the same text is additionally appended to it, which lets the iOS
// harness persist the report inside the application sandbox so that Mobile
// Harness can pull it back off the device.
absl::Status RunEmbedding(const EmbeddingLiteRtLmSettings& settings,
                          std::string* report_out = nullptr);

// Convenience overload that runs the embedding pipeline using the module-level
// settings populated by `SetEmbeddingFlag()`.
absl::Status RunEmbedding(std::string* report_out = nullptr);

// Sets the field on `settings` (or the module-level settings used by the
// no-settings `RunEmbedding` overload) corresponding to flag `name`.
//
// Returns `absl::NotFoundError` when `name` does not correspond to a known
// flag. Callers that iterate over the whole environment are expected to ignore
// that particular error.
absl::Status SetEmbeddingFlag(EmbeddingLiteRtLmSettings* settings,
                              absl::string_view name, absl::string_view value);
absl::Status SetEmbeddingFlag(absl::string_view name, absl::string_view value);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EMBEDDING_LITERT_LM_LIB_H_
