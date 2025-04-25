// Copyright 2025 The ODML Authors.
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

// ODML pipeline to execute or benchmark LLM graph on device.
//
// The pipeline does the following
// 1) Read the corresponding parameters, weight and model file paths.
// 2) Construct a graph model with the setting.
// 3) Execute model inference and generate the output.
//
// Consider run_llm_inference_engine.sh as an example to run on android device.

#include <memory>
#include <optional>
#include <string>

#include "third_party/absl/flags/flag.h"
#include "third_party/absl/flags/parse.h"
#include "third_party/absl/log/absl_check.h"
#include "third_party/absl/log/absl_log.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/odml/litert_lm/runtime/engine/engine.h"
#include "third_party/odml/litert_lm/runtime/engine/llm_model_settings.h"
#include "third_party/odml/litert_lm/runtime/executor/llm_executor_config.h"

ABSL_FLAG(std::optional<std::string>, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt, "What is the highest building in Paris?",
          "Input prompt to use for testing LLM execution.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::CpuConfig;
using ::litert::lm::GpuConfig;
using ::litert::lm::LlmExecutorConfig;
using ::litert::lm::LlmModelSettings;
using ::litert::lm::ModelAssets;

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  ABSL_LOG(INFO) << "Model path: " << model_path;
  ModelAssets model_assets;
  model_assets.model_paths.push_back(model_path);
  LlmExecutorConfig executor_settings(model_assets);

  std::string backend_str = absl::GetFlag(FLAGS_backend).value();
  ABSL_LOG(INFO) << "Choose backend: " << backend_str;
  Backend backend;
  if (backend_str == "cpu") {
    backend = Backend::CPU;
    CpuConfig config;
    config.number_of_threads = 4;
    executor_settings.SetBackendConfig(config);
  } else if (backend_str == "gpu") {
    backend = Backend::GPU;
    GpuConfig config;
    config.max_top_k = 1;
    executor_settings.SetBackendConfig(config);
  } else {
    return absl::InvalidArgumentError("Unsupported backend: " + backend_str);
  }
  executor_settings.SetBackend(backend);
  // TODO(b/397975034) Set the max num tokens based on the model.
  executor_settings.SetMaxNumTokens(160);

  LlmModelSettings model_settings(executor_settings);

  absl::StatusOr<std::unique_ptr<litert::lm::Engine>> llm =
      litert::lm::Engine::CreateEngine(model_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<litert::lm::Engine::Session>> session =
      (*llm)->CreateSession();
  ABSL_CHECK_OK(session);

  ABSL_LOG(INFO) << "Adding prompt: " << absl::GetFlag(FLAGS_input_prompt);
  absl::Status status =
      (*session)->AddTextPrompt(absl::GetFlag(FLAGS_input_prompt));
  ABSL_CHECK_OK(status);

  auto responses = (*session)->PredictSync();

  ABSL_CHECK_OK(responses);
  for (int i = 0; i < responses->GetNumOutputCandidates(); ++i) {
    ABSL_LOG(INFO) << "Response " << i << ": "
                   << responses->GetResponseTextAt(i).value();
    ABSL_LOG(INFO) << "Response " << i << " score: "
                   << responses->GetScoreAt(i).value();
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}

