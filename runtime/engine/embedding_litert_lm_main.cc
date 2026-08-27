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
#include <fstream>
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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "cpu",
          "Executor backend to use for embedding execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "/tmp/embedding-gemma-v2.litertlm",
          "Path to the embedding .litertlm file.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input string to compute the embedding for.");
ABSL_FLAG(std::string, image_path, "",
          "Optional path to an image file to compute the embedding for.");
ABSL_FLAG(bool, normalize, true,
          "Whether to L2-normalize the output embedding vector.");
ABSL_FLAG(bool, use_mmap, true,
          "Whether to use memory-mapped file for model loading.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to directory containing LiteRT dispatch libraries.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::BuildLiteRtCompiledModelResources;
using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingResponse;
using ::litert::lm::InputData;
using ::litert::lm::InputImage;
using ::litert::lm::InputText;
using ::litert::lm::MemoryMappedFile;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelResources;
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
  if (resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder).ok()) {
    vision_backend = backend;
  }
  std::optional<Backend> audio_backend = std::nullopt;
  if (resources->GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw).ok() ||
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

  const std::string prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string image_path = absl::GetFlag(FLAGS_image_path);
  if (prompt.empty() && image_path.empty()) {
    return absl::InvalidArgumentError(
        "At least one of --input_prompt or --image_path must be provided.");
  }

  std::vector<InputData> contents;
  if (!prompt.empty()) {
    std::cout << "Computing embedding for input prompt: \"" << prompt << "\""
              << std::endl;
    contents.emplace_back(InputText(prompt));
  }

  if (!image_path.empty()) {
    std::cout << "Loading image from: " << image_path << std::endl;
    std::ifstream file(image_path, std::ios::binary);
    if (!file.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Failed to open image file: ", image_path));
    }
    std::string image_bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    contents.emplace_back(InputImage(std::move(image_bytes)));
  }

  auto response_result = engine->ComputeEmbedding(
      contents, {.normalize = absl::GetFlag(FLAGS_normalize)});
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
