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
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "support/preprocessor/image_preprocessor.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "cpu",
          "Executor backend to use for embedding execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "/tmp/embedding_gemma_v2.litertlm",
          "Path to the embedding .litertlm file.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input string to compute the embedding for.");
ABSL_FLAG(std::string, image_path, "",
          "Optional path to an image file to compute the embedding for.");
ABSL_FLAG(bool, normalize, false,
          "Whether to L2-normalize the output embedding vector.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::EmbeddingEngineImpl;
using ::litert::lm::EmbeddingEngineSettings;
using ::litert::lm::EmbeddingResponse;
using ::litert::lm::InputData;
using ::litert::lm::InputImage;
using ::litert::lm::InputImageEnd;
using ::litert::lm::InputText;
using ::litert::lm::LitertLmLoader;
using ::litert::lm::ModelAssets;
using ::litert::lm::ModelResourcesLitertLm;
using ::litert::lm::ModelType;
using ::litert::lm::OwnedEnvironment;

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  std::cout << "Loading model from: " << model_path << std::endl;

  LITERT_ASSIGN_OR_RETURN(auto scoped_file,
                          ::litert::ScopedFile::Open(model_path));
  LITERT_ASSIGN_OR_RETURN(auto loader,
                          LitertLmLoader::Create(std::move(scoped_file)));
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          ModelResourcesLitertLm::Create(std::move(loader)));

  LITERT_ASSIGN_OR_RETURN(auto tokenizer, resources->GetTokenizer());
  if (!tokenizer) {
    return absl::NotFoundError("Tokenizer not found in model resources.");
  }

  LITERT_ASSIGN_OR_RETURN(auto env, ::litert::Environment::Create({}));
  auto owned_env = std::make_unique<OwnedEnvironment>(OwnedEnvironment{
      /*magic_number_configs_helper=*/nullptr, std::move(env)});

  const std::string backend_str = absl::GetFlag(FLAGS_backend);
  LITERT_ASSIGN_OR_RETURN(Backend backend,
                          ::litert::lm::GetBackendFromString(backend_str));

  LITERT_ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(model_path));
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

  std::cout << "Initializing EmbeddingEngine..." << std::endl;
  LITERT_ASSIGN_OR_RETURN(
      auto engine,
      EmbeddingEngineImpl::Create(std::move(resources), std::move(owned_env),
                                  std::move(tokenizer), std::move(settings)));

  const std::string prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string image_path = absl::GetFlag(FLAGS_image_path);
  if (prompt.empty() && image_path.empty()) {
    return absl::InvalidArgumentError(
        "At least one of --input_prompt or --image_path must be provided.");
  }

  std::vector<InputData> contents;
  if (!image_path.empty()) {
    std::cout << "Loading image from: " << image_path << std::endl;
    std::ifstream file(image_path, std::ios::binary);
    if (!file.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Failed to open image file: ", image_path));
    }
    std::string image_bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    InputImage raw_image(std::move(image_bytes));

    auto preprocessor = ::litert::support::ImagePreprocessor::Create();
    if (preprocessor == nullptr) {
      return absl::InternalError("Failed to create image preprocessor.");
    }

    ::litert::support::ImagePreprocessParameter preprocess_param;
    preprocess_param.SetPatchifyConfig(
        ::litert::support::ImagePreprocessParameter::PatchifyConfig{
            .patch_width = 16,
            .patch_height = 16,
            .max_num_patches = 2520,
            .pooling_kernel_size = 3,
        });

    LITERT_ASSIGN_OR_RETURN(
        InputImage processed_image,
        preprocessor->Preprocess(raw_image, preprocess_param));

    contents.emplace_back(std::move(processed_image));
    contents.emplace_back(InputImageEnd());
  }

  if (!prompt.empty()) {
    std::cout << "Computing embedding for input prompt: \"" << prompt << "\""
              << std::endl;
    contents.emplace_back(InputText(prompt));
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
