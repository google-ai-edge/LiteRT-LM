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

// ODML pipeline to execute an embedding model graph on device.
//
// The pipeline does the following:
// 1) Reads the embedding .litertlm file from --model_path.
// 2) Initializes the ModelResources, Tokenizer, Environment, and Settings.
// 3) Constructs an EmbeddingEngine.
// 4) Computes an embedding for --input_prompt and prints the result.

#include <iostream>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/engine/embedding_litert_lm_lib.h"
#include "runtime/engine/shared_flags.h"

ABSL_FLAG(std::string, backend, "cpu",
          "Executor backend to use for embedding execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Path to the embedding .litertlm file.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input string to compute the embedding for.");
ABSL_FLAG(std::string, input_prompt_file, "",
          "Optional file path to the input prompt.");
ABSL_FLAG(std::string, image_path, "",
          "Optional path to an image file to compute the embedding for.");
ABSL_FLAG(
    std::string, audio_path, "",
    "Optional path to an audio file (.wav) to compute the embedding for.");
ABSL_FLAG(std::string, output_embedding_path, "",
          "Optional path to save the full embedding vector as a JSON file.");
ABSL_FLAG(
    std::string, compare_embedding_path, "",
    "Optional path to a vector JSON file (e.g. golden reference) to compute "
    "and display Cosine Similarity against the current run's output vector.");
ABSL_FLAG(bool, normalize, true,
          "Whether to L2-normalize the output embedding vector.");
ABSL_FLAG(bool, use_mmap, true,
          "Whether to use memory-mapped file for model loading.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to directory containing LiteRT dispatch libraries.");
ABSL_FLAG(int, num_warmup, 2, "Number of warmup iterations for benchmarking.");
ABSL_FLAG(std::string, input_overflow_strategy, "truncate",
          "Input overflow strategy: error, truncate, or chunk_and_average.");
ABSL_FLAG(
    int, max_input_length, 0,
    "Maximum input length for embedding execution. If greater than 0, "
    "text encoder signatures will be pruned to only load signatures up to "
    "this capacity.");
ABSL_FLAG(
    std::string, activation_data_type, "",
    "Activation data type for execution: float32, float16, int16, or int8. "
    "Defaults to float32 on GPU.");
ABSL_FLAG(
    int, min_input_length, 0,
    "Minimum input length for embedding execution. If greater than 0, "
    "text encoder signatures smaller than this capacity will be excluded.");
ABSL_FLAG(
    bool, lazy_load_multimodal_encoders, true,
    "Whether to compile the vision and audio encoders bundled in the model on "
    "their first use instead of at engine creation time. Text-only runs never "
    "pay for the encoders; a run passing --image_path or --audio_path pays a "
    "one-time initialization cost on the first such request. Set to false to "
    "restore eager compilation of every bundled encoder.");

namespace {

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  litert::lm::EmbeddingLiteRtLmSettings settings;
  settings.backend = absl::GetFlag(FLAGS_backend);
  settings.vision_backend = absl::GetFlag(FLAGS_vision_backend);
  settings.audio_backend = absl::GetFlag(FLAGS_audio_backend);
  settings.model_path = absl::GetFlag(FLAGS_model_path);
  settings.input_prompt = absl::GetFlag(FLAGS_input_prompt);
  settings.input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  settings.image_path = absl::GetFlag(FLAGS_image_path);
  settings.audio_path = absl::GetFlag(FLAGS_audio_path);
  settings.output_embedding_path = absl::GetFlag(FLAGS_output_embedding_path);
  settings.compare_embedding_path = absl::GetFlag(FLAGS_compare_embedding_path);
  settings.normalize = absl::GetFlag(FLAGS_normalize);
  settings.use_mmap = absl::GetFlag(FLAGS_use_mmap);
  settings.dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);
  settings.num_warmup = absl::GetFlag(FLAGS_num_warmup);
  settings.num_iterations = absl::GetFlag(FLAGS_num_iterations);
  settings.input_overflow_strategy =
      absl::GetFlag(FLAGS_input_overflow_strategy);
  settings.max_input_length = absl::GetFlag(FLAGS_max_input_length);
  settings.min_input_length = absl::GetFlag(FLAGS_min_input_length);
  settings.activation_data_type = absl::GetFlag(FLAGS_activation_data_type);
  settings.benchmark = absl::GetFlag(FLAGS_benchmark);
  settings.benchmark_prefill_tokens =
      absl::GetFlag(FLAGS_benchmark_prefill_tokens);
  settings.visual_token_budget = absl::GetFlag(FLAGS_visual_token_budget);
  settings.report_peak_memory_footprint =
      absl::GetFlag(FLAGS_report_peak_memory_footprint);
  settings.num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
  settings.lazy_load_multimodal_encoders =
      absl::GetFlag(FLAGS_lazy_load_multimodal_encoders);

  return litert::lm::RunEmbedding(settings);
}

}  // namespace

int main(int argc, char** argv) {
  absl::Status status = MainHelper(argc, argv);
  if (!status.ok()) {
    std::cerr << "Main execution failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
