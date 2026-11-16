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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#ifndef LITERT_LM_MAIN_NO_ENTRY_POINT
#include "absl/flags/parse.h"  // from @com_google_absl
#endif  // LITERT_LM_MAIN_NO_ENTRY_POINT
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/shared_flags.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(std::string, input_prompt_file, "", "File path to the input prompt.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::litert::lm::OptionalArgs;
using ::nlohmann::json;

constexpr absl::string_view kUsage =
    "Usage: litert_lm_main --model_path=<path> "
    "[--backend=cpu|gpu|npu] [--input_prompt=\"<prompt>\"] "
    "[--input_prompt_file=<path>]";

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateMessageCallback(
    std::ostream& out) {
  return [&out](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      out << "Error: " << message.status() << std::endl;
      return;
    }
    if (message->is_null()) {
      out << std::endl << std::flush;
      return;
    }
    for (const auto& content : (*message)["content"]) {
      out << content["text"].get<std::string>();
    }
    out << std::flush;
  };
}

// Gets the input prompt from the command line flag or file.
absl::StatusOr<std::string> GetInputPrompt() {
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!input_prompt.empty() && !input_prompt_file.empty()) {
    return absl::InvalidArgumentError(
        "Only one of --input_prompt and --input_prompt_file can be specified.");
  }
  if (!input_prompt.empty()) {
    return input_prompt;
  }
  if (!input_prompt_file.empty()) {
    std::ifstream file(input_prompt_file);
    if (!file.is_open()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not open input_prompt_file: ", input_prompt_file));
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
  // If no input prompt is provided, use the default prompt.
  return "What is the tallest building in the world?";
}

absl::Status MainHelper(std::ostream& out) {
  // Overrides the default for FLAGS_minloglevel to error.
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("--model_path must be specified.\n", kUsage));
  }
  ABSL_ASSIGN_OR_RETURN(const std::string input_prompt, GetInputPrompt());
  ABSL_ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                        ModelAssets::Create(model_path));
  auto backend_str = absl::GetFlag(FLAGS_backend);
  ABSL_ASSIGN_OR_RETURN(Backend backend,
                        litert::lm::GetBackendFromString(backend_str));
  ABSL_ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));
  // Enable benchmark by default.
  engine_settings.GetMutableBenchmarkParams() =
      litert::lm::proto::BenchmarkParams();
  const int visual_token_budget = absl::GetFlag(FLAGS_visual_token_budget);
  if (visual_token_budget > 0) {
    engine_settings.SetMaxVisionTokensPerImage(visual_token_budget);
  }
  ABSL_ASSIGN_OR_RETURN(auto use_ir_model,
                        litert::lm::GetUseIrModelFromString(
                            absl::GetFlag(FLAGS_gpu_use_ir_model)));
  engine_settings.GetMutableMainExecutorSettings().SetUseIrModel(use_ir_model);

  if (backend == Backend::CPU) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ABSL_ASSIGN_OR_RETURN(
        auto cpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::CpuConfig>());
    const int num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
    if (num_cpu_threads > 0) {
      cpu_settings.number_of_threads = num_cpu_threads;
    }
    cpu_settings.enable_ynnpack = absl::GetFlag(FLAGS_enable_ynnpack);
    executor_settings.SetBackendConfig(cpu_settings);
  }
  if (!absl::GetFlag(FLAGS_custom_backend_options).empty()) {
    ABSL_RETURN_IF_ERROR(engine_settings.GetMutableMainExecutorSettings()
                             .SetCustomBackendOptionsFromStrings(
                                 absl::GetFlag(FLAGS_custom_backend_options)));
  }

  // Create the engine.
  ABSL_ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateDefault(
                                         std::move(engine_settings)));

  // Create the conversation.
  std::unique_ptr<Conversation> conversation;
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  ABSL_ASSIGN_OR_RETURN(auto conversation_config,
                        ConversationConfig::Builder()
                            .SetSessionConfig(session_config)
                            .Build(*engine));
  ABSL_ASSIGN_OR_RETURN(conversation,
                        Conversation::Create(*engine, conversation_config));

  // Prepare the message to send.
  json content_list = json::array();
  out << "input_prompt: " << input_prompt << std::endl;
  content_list.push_back({{"type", "text"}, {"text", input_prompt}});

  // Limit the number of decoded tokens when --max_output_tokens is positive.
  OptionalArgs optional_args;
  const int max_output_tokens = absl::GetFlag(FLAGS_max_output_tokens);
  if (max_output_tokens > 0) {
    optional_args.max_output_tokens = max_output_tokens;
  }

  // Send the message and wait for the response, asynchronously log the
  // response.
  ABSL_RETURN_IF_ERROR(conversation->SendMessageAsync(
      json::object({{"role", "user"}, {"content", content_list}}),
      CreateMessageCallback(out), std::move(optional_args)));
  ABSL_RETURN_IF_ERROR(engine->WaitUntilDone(absl::Minutes(10)));

  // Print the benchmark info.
  auto benchmark_info = conversation->GetBenchmarkInfo();
  if (benchmark_info.ok()) {
    out << std::endl << *benchmark_info << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace

namespace litert::lm {

int RunLiteRtLmFromFlags(std::ostream& out, std::ostream& err);

int RunLiteRtLmFromFlags(std::ostream& out, std::ostream& err) {
  const absl::Status status = MainHelper(out);
  if (!status.ok()) {
    err << "Error: " << status.message() << std::endl;
    if (!absl::GetFlag(FLAGS_model_path).empty() &&
        absl::GetFlag(FLAGS_backend) == "gpu") {
      err << "Hint: pass --backend=cpu to run on CPU when a GPU delegate is "
             "unavailable."
          << std::endl;
    }
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace litert::lm

#ifndef LITERT_LM_MAIN_NO_ENTRY_POINT
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  return litert::lm::RunLiteRtLmFromFlags(std::cout, std::cerr);
}
#endif  // LITERT_LM_MAIN_NO_ENTRY_POINT
