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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/model_resources.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/shared_flags.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "tflite/profiling/memory_info.h"  // from @litert

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
ABSL_FLAG(bool, use_mmap, false,
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

namespace {

using ::litert::lm::ActivationDataType;
using ::litert::lm::Backend;
using ::litert::lm::BuildLiteRtCompiledModelResources;
using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingOptions;
using ::litert::lm::EmbeddingResponse;
using ::litert::lm::InputAudio;
using ::litert::lm::InputData;
using ::litert::lm::InputImage;
using ::litert::lm::InputOverflowStrategy;
using ::litert::lm::InputText;
using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelType;
using ::litert::lm::OwnedEnvironment;
using ::litert::lm::ScopedFile;

absl::StatusOr<ModelAssets> CreateModelAssets(bool use_mmap,
                                              absl::string_view model_path) {
  if (use_mmap) {
    std::cout << "Using memory-mapped file." << std::endl;
    LITERT_ASSIGN_OR_RETURN(auto unique_memory_mapped_file,
                            MemoryMappedFile::Create(model_path));
    std::shared_ptr<MemoryMappedFile> memory_mapped_file =
        std::move(unique_memory_mapped_file);
    return ModelAssets::Create(memory_mapped_file, model_path);
  } else {
    std::cout << "Using ScopedFile." << std::endl;
    LITERT_ASSIGN_OR_RETURN(auto local_scoped_file,
                            ScopedFile::Open(model_path));
    std::shared_ptr<ScopedFile> scoped_file =
        std::make_shared<ScopedFile>(std::move(local_scoped_file));
    return ModelAssets::Create(scoped_file, model_path);
  }
}

absl::StatusOr<InputOverflowStrategy> ParseInputOverflowStrategy(
    absl::string_view strategy_str) {
  if (strategy_str == "error") {
    return InputOverflowStrategy::kError;
  } else if (strategy_str == "truncate") {
    return InputOverflowStrategy::kTruncate;
  } else if (strategy_str == "chunk_and_average") {
    return InputOverflowStrategy::kChunkAndAverage;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid --input_overflow_strategy: ", strategy_str,
                   ". Must be 'error', 'truncate', or 'chunk_and_average'."));
}

absl::StatusOr<double> ComputeCosineSimilarity(absl::Span<const float> v1,
                                               absl::Span<const float> v2) {
  if (v1.size() != v2.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Vector dimension mismatch: %d vs %d", v1.size(), v2.size()));
  }
  double dot = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for (size_t i = 0; i < v1.size(); ++i) {
    dot += static_cast<double>(v1[i]) * static_cast<double>(v2[i]);
    norm1 += static_cast<double>(v1[i]) * static_cast<double>(v1[i]);
    norm2 += static_cast<double>(v2[i]) * static_cast<double>(v2[i]);
  }
  if (norm1 > 0.0 && norm2 > 0.0) {
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
  }
  return 0.0;
}

absl::StatusOr<std::string> ReadFileToString(
    absl::string_view file_path, std::ios_base::openmode mode = std::ios::in) {
  std::string path_str(file_path);
  std::ifstream file(path_str, mode);
  if (!file.is_open()) {
    const char* build_working_dir = std::getenv("BUILD_WORKING_DIRECTORY");
    if (build_working_dir != nullptr) {
      path_str = absl::StrCat(build_working_dir, "/", file_path);
      file.open(path_str, mode);
    }
  }
  if (!file.is_open()) {
    const char* build_workspace_dir = std::getenv("BUILD_WORKSPACE_DIRECTORY");
    if (build_workspace_dir != nullptr) {
      path_str = absl::StrCat(build_workspace_dir, "/", file_path);
      file.open(path_str, mode);
    }
  }
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open file: ", file_path));
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<std::vector<float>> LoadEmbeddingFromJsonFile(
    absl::string_view file_path) {
  LITERT_ASSIGN_OR_RETURN(std::string content, ReadFileToString(file_path));
  nlohmann::json json_data =
      nlohmann::json::parse(content, nullptr, /*allow_exceptions=*/false);
  if (json_data.is_discarded() || !json_data.is_array()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid JSON array in embedding file: ", file_path));
  }
  std::vector<float> vec;
  vec.reserve(json_data.size());
  for (const auto& elem : json_data) {
    if (!elem.is_number()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "JSON array contains non-numeric element in: ", file_path));
    }
    vec.push_back(elem.get<float>());
  }
  return vec;
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  std::cout << "Loading model from: " << model_path << std::endl;

  const std::string backend_str = absl::GetFlag(FLAGS_backend);
  LITERT_ASSIGN_OR_RETURN(Backend backend,
                          ::litert::lm::GetBackendFromString(backend_str));
  const bool enable_file_backed_model_loading = (backend == Backend::NPU);

  bool use_mmap = absl::GetFlag(FLAGS_use_mmap);
  if (backend == Backend::NPU && use_mmap) {
    ABSL_LOG(WARNING)
        << "NPU backend selected. Disabling memory mapping to ensure "
           "file-backed model loading is used.";
    use_mmap = false;
  }

  LITERT_ASSIGN_OR_RETURN(auto model_assets,
                          CreateModelAssets(use_mmap, model_path));

  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              model_assets, enable_file_backed_model_loading));

  LITERT_ASSIGN_OR_RETURN(auto tokenizer, resources->GetTokenizer());
  if (!tokenizer) {
    return absl::NotFoundError("Tokenizer not found in model resources.");
  }

  std::optional<Backend> vision_backend = std::nullopt;
  if (const auto vision_backend_str = absl::GetFlag(FLAGS_vision_backend);
      vision_backend_str.has_value() && !vision_backend_str->empty()) {
    LITERT_ASSIGN_OR_RETURN(vision_backend, ::litert::lm::GetBackendFromString(
                                                *vision_backend_str));
  } else if (resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok()) {
    vision_backend = backend;
  }

  std::optional<Backend> audio_backend = std::nullopt;
  if (const auto audio_backend_str = absl::GetFlag(FLAGS_audio_backend);
      audio_backend_str.has_value() && !audio_backend_str->empty()) {
    LITERT_ASSIGN_OR_RETURN(
        audio_backend, ::litert::lm::GetBackendFromString(*audio_backend_str));
  } else if (resources->GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok() ||
             resources->GetTFLiteModel(ModelType::kTfLiteAudioFrontend).ok()) {
    audio_backend = backend;
  }

  LITERT_ASSIGN_OR_RETURN(
      auto settings, EmbeddingEngineSettings::CreateDefault(
                         model_assets, backend, vision_backend, audio_backend));

  const std::string dispatch_library_dir =
      absl::GetFlag(FLAGS_dispatch_library_dir);
  if (!dispatch_library_dir.empty()) {
    settings.GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
        dispatch_library_dir);
  }

  if (backend == Backend::CPU) {
    const int num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
    if (num_cpu_threads > 0) {
      settings.GetMutableMainExecutorSettings().SetNumThreads(num_cpu_threads);
    }
  }

  const bool is_benchmark = absl::GetFlag(FLAGS_benchmark);
  if (is_benchmark) {
    auto& benchmark_params = settings.GetMutableBenchmarkParams();
    const int benchmark_prefill_tokens =
        absl::GetFlag(FLAGS_benchmark_prefill_tokens);
    if (benchmark_prefill_tokens < 0) {
      return absl::InvalidArgumentError(
          "--benchmark_prefill_tokens must be non-negative.");
    }
    if (benchmark_prefill_tokens > 0) {
      benchmark_params.set_num_prefill_tokens(benchmark_prefill_tokens);
    }
  }

  const int visual_token_budget = absl::GetFlag(FLAGS_visual_token_budget);
  if (visual_token_budget > 0) {
    settings.SetVisionTokensPerImage(visual_token_budget);
  }

  const int max_input_length = absl::GetFlag(FLAGS_max_input_length);
  if (max_input_length > 0) {
    settings.SetMaxInputLength(max_input_length);
  }
  const int min_input_length = absl::GetFlag(FLAGS_min_input_length);
  if (min_input_length > 0) {
    settings.SetMinInputLength(min_input_length);
  }

  const std::string activation_data_type_str =
      absl::GetFlag(FLAGS_activation_data_type);
  if (!activation_data_type_str.empty()) {
    LITERT_ASSIGN_OR_RETURN(ActivationDataType act_type,
                            ::litert::lm::GetActivationDataTypeFromString(
                                activation_data_type_str));
    settings.GetMutableMainExecutorSettings().SetActivationDataType(act_type);
    if (settings.GetMutableVisionExecutorSettings().has_value()) {
      settings.GetMutableVisionExecutorSettings()->SetActivationDataType(
          act_type);
    }
    if (settings.GetMutableAudioExecutorSettings().has_value()) {
      settings.GetMutableAudioExecutorSettings()->SetActivationDataType(
          act_type);
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto owned_env,
                          CreateEnvironment(settings, resources.get()));
  auto owned_env_ptr =
      std::make_unique<OwnedEnvironment>(std::move(owned_env));

  std::cout << "Initializing EmbeddingEngine..." << std::endl;
  LITERT_ASSIGN_OR_RETURN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources),
                                  std::move(owned_env_ptr),
                                  std::move(tokenizer),
                                  std::move(settings)));

  std::string prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!prompt.empty() && !input_prompt_file.empty()) {
    return absl::InvalidArgumentError(
        "Only one of --input_prompt and --input_prompt_file can be specified.");
  }
  if (!input_prompt_file.empty()) {
    LITERT_ASSIGN_OR_RETURN(prompt, ReadFileToString(input_prompt_file));
  }

  const std::string image_path = absl::GetFlag(FLAGS_image_path);
  const std::string audio_path = absl::GetFlag(FLAGS_audio_path);
  const int benchmark_prefill_tokens =
      is_benchmark ? absl::GetFlag(FLAGS_benchmark_prefill_tokens) : 0;

  if (prompt.empty() && image_path.empty() && audio_path.empty()) {
    if (!is_benchmark || benchmark_prefill_tokens <= 0) {
      return absl::InvalidArgumentError(
          is_benchmark
              ? "At least one of --input_prompt, --input_prompt_file, "
                "--image_path, --audio_path, or --benchmark_prefill_tokens "
                "must be provided in benchmark mode."
              : "At least one of --input_prompt, --input_prompt_file, "
                "--image_path, or --audio_path must be provided.");
    }
  }

  std::vector<InputData> contents;
  if (!prompt.empty() || benchmark_prefill_tokens > 0) {
    if (prompt.empty()) {
      prompt = "benchmark";
    }
    const std::string display_prompt =
        prompt.size() > 100 ? absl::StrCat(prompt.substr(0, 100), "...")
                            : prompt;
    if (benchmark_prefill_tokens > 0) {
      std::cout << "Computing embedding for input prompt: \"" << display_prompt
                << "\" (fixed to " << benchmark_prefill_tokens
                << " prefill tokens)" << std::endl;
    } else {
      std::cout << "Computing embedding for input prompt: \"" << display_prompt
                << "\"" << std::endl;
    }
    contents.emplace_back(InputText(prompt));
  }

  if (!image_path.empty()) {
    std::cout << "Loading image from: " << image_path << std::endl;
    LITERT_ASSIGN_OR_RETURN(std::string image_bytes,
                            ReadFileToString(image_path, std::ios::binary));
    contents.emplace_back(InputImage(std::move(image_bytes)));
  }

  if (!audio_path.empty()) {
    std::cout << "Loading audio from: " << audio_path << std::endl;
    LITERT_ASSIGN_OR_RETURN(std::string audio_bytes,
                            ReadFileToString(audio_path, std::ios::binary));
    contents.emplace_back(InputAudio(std::move(audio_bytes)));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto overflow_strategy,
      ParseInputOverflowStrategy(absl::GetFlag(FLAGS_input_overflow_strategy)));
  EmbeddingOptions options = {
      .normalize = absl::GetFlag(FLAGS_normalize),
      .input_overflow_strategy = overflow_strategy,
  };

  if (absl::GetFlag(FLAGS_benchmark)) {
    const int num_warmup = absl::GetFlag(FLAGS_num_warmup);
    const int num_iterations = absl::GetFlag(FLAGS_num_iterations);
    std::cout << "Starting benchmark with " << num_warmup << " warmup and "
              << num_iterations << " measurement iterations..." << std::endl;

    for (int i = 0; i < num_warmup; ++i) {
      auto warmup_res = engine->ComputeEmbedding(contents, options);
      if (!warmup_res.ok()) {
        std::cerr << "Warmup iteration " << i
                  << " failed: " << warmup_res.status() << std::endl;
        return warmup_res.status();
      }
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(num_iterations);
    EmbeddingResponse last_response;
    for (int i = 0; i < num_iterations; ++i) {
      absl::Time start_time = absl::Now();
      auto response_result = engine->ComputeEmbedding(contents, options);
      absl::Time end_time = absl::Now();
      if (!response_result.ok()) {
        std::cerr << "Measurement iteration " << i
                  << " failed: " << response_result.status() << std::endl;
        return response_result.status();
      }
      double elapsed_ms = absl::ToDoubleMilliseconds(end_time - start_time);
      latencies_ms.push_back(elapsed_ms);
      last_response = *std::move(response_result);
    }

    double total_ms = 0.0;
    double min_ms = latencies_ms.empty() ? 0.0 : latencies_ms[0];
    double max_ms = latencies_ms.empty() ? 0.0 : latencies_ms[0];
    for (double l : latencies_ms) {
      total_ms += l;
      min_ms = std::min(min_ms, l);
      max_ms = std::max(max_ms, l);
    }
    double avg_ms =
        latencies_ms.empty() ? 0.0 : (total_ms / latencies_ms.size());

    // Print and log benchmark metrics
    std::cout << "\n================ BENCHMARK RESULT ================"
              << std::endl;
    auto benchmark_info = engine->GetBenchmarkInfo();
    if (benchmark_info.has_value()) {
      for (const auto& mark : benchmark_info->GetMarkDurations()) {
        double mark_ms = absl::ToDoubleMilliseconds(mark.second);
        ABSL_LOG(INFO) << absl::StrFormat("- %s: %.2f ms", mark.first, mark_ms);
        std::cout << "- " << mark.first << ": " << mark_ms << " ms"
                  << std::endl;
      }
    }

    ABSL_LOG(INFO) << absl::StrFormat(
        "Average Latency: %.2f ms (min: %.2f ms, max: %.2f ms)", avg_ms, min_ms,
        max_ms);
    std::cout << absl::StrFormat(
                     "Average Latency: %.2f ms (min: %.2f ms, max: %.2f ms)\n",
                     avg_ms, min_ms, max_ms);

    if (absl::GetFlag(FLAGS_report_peak_memory_footprint)) {
      auto mem_usage = tflite::profiling::memory::GetMemoryUsage();
      if (mem_usage.IsSupported()) {
        double peak_ram_mb =
            mem_usage.private_footprint_bytes / 1024.0 / 1024.0;
        ABSL_LOG(INFO) << absl::StrFormat("Peak system ram usage: %.2f MB",
                                          peak_ram_mb);
        std::cout << absl::StrFormat("Peak system ram usage: %.2f MB\n",
                                     peak_ram_mb);
      }
    }
    std::cout << "Embedding vector dimension: "
              << last_response.embedding.size() << std::endl;
    if (const std::string compare_path =
            absl::GetFlag(FLAGS_compare_embedding_path);
        !compare_path.empty()) {
      LITERT_ASSIGN_OR_RETURN(std::vector<float> golden_vec,
                              LoadEmbeddingFromJsonFile(compare_path));
      LITERT_ASSIGN_OR_RETURN(
          double cos_sim,
          ComputeCosineSimilarity(last_response.embedding, golden_vec));
      std::cout << absl::StrFormat("Cosine similarity with %s: %.10f\n",
                                   compare_path, cos_sim);
    }
    std::cout << "=================================================="
              << std::endl;
    return absl::OkStatus();
  }

  auto response_result = engine->ComputeEmbedding(contents, options);
  if (!response_result.ok()) {
    std::cerr << "ComputeEmbedding failed with error: "
              << response_result.status() << std::endl;
    return response_result.status();
  }
  EmbeddingResponse response = *std::move(response_result);

  std::cout << "\n================ RESULT ================" << std::endl;
  std::cout << "Input length: " << response.input_length << std::endl;
  if (response.truncated_length.has_value()) {
    std::cout << "Truncated length: " << *response.truncated_length
              << std::endl;
  }
  std::cout << "Number of chunks: " << response.num_chunks << std::endl;
  std::cout << "Embedding vector dimension: " << response.embedding.size()
            << std::endl;

  const int num_to_print =
      std::min(10, static_cast<int>(response.embedding.size()));
  std::cout << "First " << num_to_print << " values: [";
  for (int i = 0; i < num_to_print; ++i) {
    std::cout << response.embedding[i];
    if (i < num_to_print - 1) std::cout << ", ";
  }
  if (response.embedding.size() > num_to_print) {
    std::cout << ", ...";
  }
  std::cout << "]" << std::endl;
  std::cout << "========================================" << std::endl;

  if (const std::string compare_path =
          absl::GetFlag(FLAGS_compare_embedding_path);
      !compare_path.empty()) {
    LITERT_ASSIGN_OR_RETURN(std::vector<float> golden_vec,
                            LoadEmbeddingFromJsonFile(compare_path));
    LITERT_ASSIGN_OR_RETURN(
        double cos_sim,
        ComputeCosineSimilarity(response.embedding, golden_vec));
    std::cout << absl::StrFormat("Cosine similarity with %s: %.10f\n",
                                 compare_path, cos_sim);
  }

  if (const std::string output_path =
          absl::GetFlag(FLAGS_output_embedding_path);
      !output_path.empty()) {
    std::ofstream out_file(output_path);
    if (!out_file.is_open()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to open output embedding file: ", output_path));
    }
    out_file << "[\n";
    for (size_t i = 0; i < response.embedding.size(); ++i) {
      out_file << "  " << std::setprecision(8) << response.embedding[i];
      if (i + 1 < response.embedding.size()) {
        out_file << ",\n";
      } else {
        out_file << "\n";
      }
    }
    out_file << "]\n";
    std::cout << "Saved full embedding vector (" << response.embedding.size()
              << " dimensions) to: " << output_path << std::endl;
  }

  // Print benchmark info if enabled.
  if (auto benchmark_info = engine->GetBenchmarkInfo()) {
    std::cout << *benchmark_info << std::endl;
  }

  return absl::OkStatus();
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
