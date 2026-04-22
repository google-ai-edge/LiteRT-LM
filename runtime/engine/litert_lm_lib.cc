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

#include "runtime/engine/litert_lm_lib.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log_sink_registry.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "tflite/profiling/memory_info.h"  // from @litert
#include "tflite/profiling/memory_usage_monitor.h"  // from @litert

namespace litert {
namespace lm {

using ::litert::ScopedFile;
using ::litert::lm::Backend;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::InputText;

using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::nlohmann::json;

// Memory check interval in milliseconds.
constexpr int kMemoryCheckIntervalMs = 50;
// Timeout duration for waiting until the engine is done with all the tasks.
const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(10);

namespace {
// Creates the ModelAssets from the LiteRtLmSettings.
absl::StatusOr<ModelAssets> CreateModelAssets(
    const LiteRtLmSettings& settings) {
  if (settings.model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  ABSL_LOG(INFO) << "Model path: " << settings.model_path;
  if (!settings.load_model_from_descriptor) {
    return ModelAssets::Create(settings.model_path);
  }
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(settings.model_path));
  return ModelAssets::Create(
      std::make_shared<ScopedFile>(std::move(scoped_file)));
}

// Helper to process the sampler backend string and return a sampler backend
// if possible. Otherwise, return std::nullopt.
std::optional<Backend> GetSamplerBackend(const LiteRtLmSettings& settings) {
  const std::string& sampler_backend_str = settings.sampler_backend;
  if (sampler_backend_str.empty()) {
    return std::nullopt;
  }
  const absl::StatusOr<Backend> sampler_backend =
      GetBackendFromString(sampler_backend_str);
  if (!sampler_backend.ok()) {
    ABSL_LOG(WARNING) << "Ignore invalid sampler backend string: "
                      << sampler_backend.status();
    return std::nullopt;
  }
  return *sampler_backend;
}

absl::Status PrintMessage(const Message& message,
                          std::stringstream& captured_output,
                          bool streaming = false) {
  std::stringstream output;
  if (message.contains("content")) {
    if (message["content"].is_array()) {
      for (const auto& content : message["content"]) {
        if (content.contains("type") && content["type"] == "text" &&
            content.contains("text")) {
          captured_output << content["text"].get<std::string>();
          output << content["text"].get<std::string>();
        }
      }

    } else if (message["content"].is_object() &&
               message["content"].contains("text") &&
               message["content"]["text"].is_string()) {
      captured_output << message["content"]["text"].get<std::string>();
      output << message["content"]["text"].get<std::string>();
    }

    if (streaming) {
      std::cout << output.str() << std::flush;
    } else {
      captured_output << std::endl;
      std::cout << output.str() << std::endl;
    }
    return absl::OkStatus();
  }

  if (message.contains("tool_calls") ||
      (message.contains("type") && message["type"] == "function")) {
    // Gracefully handle function calls without throwing or failing
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Invalid message: " + message.dump());
}

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreatePrintMessageCallback(
    std::stringstream& captured_output) {
  return [&captured_output](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      std::cout << message.status().message() << std::endl;
      return;
    }
    if (message->is_null()) {
      std::cout << std::endl << std::flush;
      return;
    }
    auto status = PrintMessage(*message, captured_output,
                               /*streaming=*/true);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to print message: " << status;
    }
  };
}

void CheckExpectedOutput(const std::string& captured_output,
                         const LiteRtLmSettings& settings) {
  // Skip printing the output when using fake prefill tokens.
  bool should_print_output = settings.benchmark_prefill_tokens == 0;
  if (should_print_output) {
    ABSL_LOG(INFO) << "Captured model output: " << captured_output;
  }
  if (settings.expected_output.has_value()) {
    if (!absl::StrContainsIgnoreCase(captured_output,
                                     *settings.expected_output)) {
      ABSL_LOG(FATAL) << "Expected output: " << *settings.expected_output
                      << " was not found in response: " << captured_output;
    }
  }
}

absl::StatusOr<std::unique_ptr<Constraint>> CreateRegexConstraint(
    const Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    std::string constraint_regex) {
  ASSIGN_OR_RETURN(
      auto constraint_provider,
      CreateConstraintProvider(LlGuidanceConfig(), tokenizer, stop_token_ids));
  return constraint_provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kRegex,
                              .constraint_string = constraint_regex});
}

absl::StatusOr<Message> RunSingleTurnConversation(
    const json& content_list, const LiteRtLmSettings& settings,
    litert::lm::Engine* engine, Conversation* conversation) {
  std::stringstream captured_output;
  OptionalArgs optional_args;
  if (settings.max_output_tokens > 0) {
    optional_args.max_output_tokens = settings.max_output_tokens;
  }

  // Skip printing the output when using fake prefill tokens.
  bool should_print_output = settings.benchmark_prefill_tokens == 0;
  if (settings.async) {
    auto print_message_callback =
        should_print_output
            ? CreatePrintMessageCallback(captured_output)
            : [](absl::StatusOr<Message> message) {};
    RETURN_IF_ERROR(conversation->SendMessageAsync(
        json::object({{"role", "user"}, {"content", content_list}}),
        std::move(print_message_callback), std::move(optional_args)));
    RETURN_IF_ERROR(engine->WaitUntilDone(kWaitUntilDoneTimeout));
    CheckExpectedOutput(captured_output.str(), settings);
    return conversation->GetHistory().back();
  } else {
    ASSIGN_OR_RETURN(
        auto model_message,
        conversation->SendMessage(
            json::object({{"role", "user"}, {"content", content_list}}),
            std::move(optional_args)));
    if (should_print_output) {
      RETURN_IF_ERROR(PrintMessage(model_message, captured_output));
    }
    CheckExpectedOutput(captured_output.str(), settings);
    return model_message;
  }
}

absl::Status RunMultiTurnConversation(const LiteRtLmSettings& settings,
                                      litert::lm::Engine* engine,
                                      Conversation* conversation) {
  std::string input_prompt;
  std::stringstream captured_output;
  do {
    std::cout << "Please enter the prompt (or press Enter to end): ";
    std::getline(std::cin, input_prompt);
    if (input_prompt.empty()) {
      break;
    }
    json content_list = json::array();

    // If there is an error building the content list, skip the prompt and
    // continue.
    std::vector<InputData> input_data;
    input_data.push_back(InputText(input_prompt));
    auto status = BuildContentList(input_data, settings, content_list);
    if (!status.ok()) {
      std::cout << status.message() << std::endl;
      continue;
    }
    if (content_list.empty()) {
      continue;
    }
    OptionalArgs optional_args;
    if (settings.max_output_tokens > 0) {
      optional_args.max_output_tokens = settings.max_output_tokens;
    }

    if (settings.async) {
      RETURN_IF_ERROR(conversation->SendMessageAsync(
          json::object({{"role", "user"}, {"content", content_list}}),
          CreatePrintMessageCallback(captured_output),
          std::move(optional_args)));
      RETURN_IF_ERROR(engine->WaitUntilDone(kWaitUntilDoneTimeout));
    } else {
      ASSIGN_OR_RETURN(
          auto model_message,
          conversation->SendMessage(
              json::object({{"role", "user"}, {"content", content_list}}),
              std::move(optional_args)));
      RETURN_IF_ERROR(PrintMessage(model_message, captured_output));
    }
  } while (true);
  CheckExpectedOutput(captured_output.str(), settings);
  return absl::OkStatus();
}

absl::Status RunSingleTurnSession(const std::string& input_prompt,
                                  const LiteRtLmSettings& settings,
                                  Engine* engine, Engine::Session* session) {
  std::stringstream captured_output;
  if (settings.async) {
    return absl::UnimplementedError(
        "Async mode is not supported for single turn session.");
  }

  ABSL_LOG(INFO) << "Running single turn session with prompt: " << input_prompt;
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  if (settings.max_output_tokens > 0) {
    decode_config.SetMaxOutputTokens(settings.max_output_tokens);
  }

  std::unique_ptr<Constraint> constraint;
  if (!settings.constraint_regex.empty()) {
    ASSIGN_OR_RETURN(
        constraint,
        CreateRegexConstraint(engine->GetTokenizer(),
                              session->GetSessionConfig().GetStopTokenIds(),
                              settings.constraint_regex));
    decode_config.SetConstraint(constraint.get());
  }

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(input_prompt));
  RETURN_IF_ERROR(session->RunPrefill(inputs));
  ASSIGN_OR_RETURN(auto responses, session->RunDecode(decode_config));
  for (const auto& response : responses.GetTexts()) {
    captured_output << response << std::endl << std::flush;
  }
  ABSL_LOG(INFO) << "output: " << captured_output.str();
  CheckExpectedOutput(captured_output.str(), settings);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<litert::lm::ScorerOutput>> RunScoreText(
    litert::lm::Engine* llm, litert::lm::Engine::Session* session,
    absl::string_view input_prompt,
    const std::vector<absl::string_view>& target_text_vector,
    bool store_char_and_token_lengths = false) {
  std::vector<litert::lm::InputData> inputs;
  inputs.emplace_back(InputText(std::string(input_prompt)));
  RETURN_IF_ERROR(session->RunPrefill(inputs));
  ASSIGN_OR_RETURN(litert::lm::Responses response,
                   session->RunTextScoring(target_text_vector,
                                           store_char_and_token_lengths));
  const std::vector<float>& scores = response.GetScores();
  if (scores.empty()) {
    ABSL_LOG(WARNING) << "No score found.";
  } else {
    // Multiply by -1 to get the negative log likelihood.
    ABSL_LOG(INFO) << "Score: " << -1 * (scores[0]) << std::endl;
  }
  if (scores.size() != target_text_vector.size()) {
    return absl::InternalError(absl::StrCat("Scores size ", scores.size(),
                                            " does not match target text size ",
                                            target_text_vector.size()));
  }
  const std::optional<std::vector<int>>& token_lengths =
      response.GetTokenLengths();
  if (store_char_and_token_lengths) {
    if (!token_lengths.has_value()) {
      return absl::InternalError("Token lengths are not available.");
    }
    if (scores.size() != token_lengths->size()) {
      return absl::InternalError(absl::StrCat(
          "Scores size ", scores.size(), " does not match token lengths size ",
          token_lengths->size()));
    }
  }
  // Write the scores and char/token lengths (if requested) to `ScorerOutputs`.
  std::vector<litert::lm::ScorerOutput> scorer_outputs;
  scorer_outputs.reserve(scores.size());
  for (int i = 0; i < scores.size(); ++i) {
    litert::lm::ScorerOutput& scorer_output = scorer_outputs.emplace_back();
    scorer_output.score = scores[i];
    if (store_char_and_token_lengths) {
      scorer_output.option_text_char_length = target_text_vector[i].size();
      scorer_output.option_text_token_length = (*token_lengths)[i];
    }
  }
  return scorer_outputs;
}

void LogBenchmarkInfo(const litert::lm::BenchmarkInfo& benchmark_info,
                      const LiteRtLmSettings& settings) {
  if (!settings.log_sink_file.has_value()) {
    ABSL_LOG(INFO) << benchmark_info;
  } else {
    ABSL_LOG(INFO) << absl::StrFormat(
        "Benchmark flags: "
        "benchmark_prefill_tokens=%d,benchmark_decode_tokens=%d,backend=%s",
        benchmark_info.GetBenchmarkParams().num_prefill_tokens(),
        benchmark_info.GetBenchmarkParams().num_decode_tokens(),
        settings.backend);
    for (const auto& phase : benchmark_info.GetInitPhases()) {
      ABSL_LOG(INFO) << absl::StrFormat(
          "%s: %.2f ms", phase.first, absl::ToDoubleMilliseconds(phase.second));
    }
    ABSL_LOG(INFO) << absl::StrFormat("Time to first token: %.2f s",
                                      benchmark_info.GetTimeToFirstToken());
    for (int i = 0; i < benchmark_info.GetTotalPrefillTurns(); ++i) {
      ABSL_LOG(INFO) << absl::StrFormat(
          "Prefill speed turn %d: %.2f tk/s", i,
          benchmark_info.GetPrefillTokensPerSec(0));
      ABSL_LOG(INFO) << absl::StrFormat(
          "Decode speed turn %d: %.2f tk/s", i,
          benchmark_info.GetDecodeTokensPerSec(0));
    }
  }
}

void LogMemoryUsage(const LiteRtLmSettings& settings, float peak_mem_mb,
                    float peak_private_mb) {
  if (!settings.log_sink_file.has_value()) {
    ABSL_LOG(INFO) << "Peak system ram usage: " << peak_mem_mb << "MB.";
    ABSL_LOG(INFO) << "Memory usage: "
                   << tflite::profiling::memory::GetMemoryUsage();
    ABSL_LOG(INFO) << "Peak private footprint: " << peak_private_mb << "MB.";
  } else {
    ABSL_LOG(INFO) << absl::StrFormat("Peak system ram usage: %.2f MB",
                                      peak_private_mb);
    ABSL_LOG(INFO) << absl::StrFormat("Peak private footprint: %.2f MB",
                                      peak_private_mb);
    auto memory_usage = tflite::profiling::memory::GetMemoryUsage();
    if (memory_usage.IsSupported()) {
      ABSL_LOG(INFO) << absl::StrFormat("Physical footprint: %.2f MB",
                                        memory_usage.mem_footprint_kb / 1000.0);
      ABSL_LOG(INFO) << absl::StrFormat(
          "Total non-mmapped heap size: %.2f MB",
          memory_usage.total_allocated_bytes / 1000.0 / 1000.0);
      ABSL_LOG(INFO) << absl::StrFormat(
          "In-use heap size: %.2f MB",
          memory_usage.in_use_allocated_bytes / 1000.0 / 1000.0);
      ABSL_LOG(INFO) << absl::StrFormat(
          "Private footprint: %.2f MB",
          memory_usage.private_footprint_bytes / 1000.0 / 1000.0);
    }
  }
}
}  // namespace

absl::StatusOr<EngineSettings> CreateEngineSettings(
    const LiteRtLmSettings& settings) {
  ASSIGN_OR_RETURN(ModelAssets model_assets, CreateModelAssets(settings));
  auto backend_str = settings.backend;
  ABSL_LOG(INFO) << "Choose backend: " << backend_str;
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(backend_str));
  std::optional<Backend> vision_backend = std::nullopt;
  if (settings.vision_backend.has_value()) {
    ABSL_LOG(INFO) << "Provided vision backend: " << *settings.vision_backend;
    ASSIGN_OR_RETURN(vision_backend, litert::lm::GetBackendFromString(
                                         *settings.vision_backend));
  }
  std::optional<Backend> audio_backend = std::nullopt;
  if (settings.audio_backend.has_value()) {
    ABSL_LOG(INFO) << "Provided audio backend: " << *settings.audio_backend;
    ASSIGN_OR_RETURN(audio_backend,
                     litert::lm::GetBackendFromString(*settings.audio_backend));
  }

  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend,
                                    vision_backend, audio_backend));
  if (settings.max_num_tokens > 0) {
    engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(
        settings.max_num_tokens);
  }
  if (settings.force_f32) {
    engine_settings.GetMutableMainExecutorSettings().SetActivationDataType(
        litert::lm::ActivationDataType::FLOAT32);
    if (settings.vision_backend.has_value()) {
      engine_settings.GetMutableVisionExecutorSettings()->SetActivationDataType(
          litert::lm::ActivationDataType::FLOAT32);
    }
    if (settings.audio_backend.has_value()) {
      engine_settings.GetMutableAudioExecutorSettings()->SetActivationDataType(
          litert::lm::ActivationDataType::FLOAT32);
    }
  }
  if (settings.disable_cache) {
    engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
    if (settings.vision_backend.has_value()) {
      engine_settings.GetMutableVisionExecutorSettings()->SetCacheDir(
          ":nocache");
    }
    if (settings.audio_backend.has_value()) {
      engine_settings.GetMutableAudioExecutorSettings()->SetCacheDir(
          ":nocache");
    }
  } else if (!settings.cache_dir.empty()) {
    engine_settings.GetMutableMainExecutorSettings().SetCacheDir(
        settings.cache_dir);
    if (settings.vision_backend.has_value()) {
      engine_settings.GetMutableVisionExecutorSettings()->SetCacheDir(
          settings.cache_dir);
    }
    if (settings.audio_backend.has_value()) {
      engine_settings.GetMutableAudioExecutorSettings()->SetCacheDir(
          settings.cache_dir);
    }
  }
  if (!settings.litert_dispatch_lib_dir.empty()) {
    engine_settings.GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
        settings.litert_dispatch_lib_dir);
  }
  if (backend == Backend::CPU) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ASSIGN_OR_RETURN(
        auto cpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::CpuConfig>());
    if (settings.num_cpu_threads > 0) {
      cpu_settings.number_of_threads = settings.num_cpu_threads;
    }
    cpu_settings.prefill_chunk_size = settings.prefill_chunk_size;
    executor_settings.SetBackendConfig(cpu_settings);
  }
  if (backend == Backend::GPU) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ASSIGN_OR_RETURN(
        auto gpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::GpuConfig>());
    gpu_settings.external_tensor_mode = settings.gpu_external_tensor_mode;
    executor_settings.SetBackendConfig(gpu_settings);
  }
  if (backend == Backend::GPU_ARTISAN) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    executor_settings.SetMaxNumImages(settings.max_num_images);
    ASSIGN_OR_RETURN(
        auto gpu_artisan_settings,
        executor_settings.MutableBackendConfig<litert::lm::GpuArtisanConfig>());
    gpu_artisan_settings.use_submodel = settings.use_submodel;
    executor_settings.SetBackendConfig(gpu_artisan_settings);
  }
  const std::optional<Backend> sampler_backend = GetSamplerBackend(settings);
  if (sampler_backend.has_value()) {
    engine_settings.GetMutableMainExecutorSettings().SetSamplerBackend(
        *sampler_backend);
  }

  AdvancedSettings advanced_settings{
      .prefill_batch_sizes = settings.prefill_batch_sizes,
      .num_output_candidates = settings.num_output_candidates,
      .configure_magic_numbers = settings.configure_magic_numbers,
      .verify_magic_numbers = settings.verify_magic_numbers,
      .clear_kv_cache_before_prefill = settings.clear_kv_cache_before_prefill,
      .num_logits_to_print_after_decode =
          static_cast<uint32_t>(settings.num_logits_to_print_after_decode),
      .gpu_madvise_original_shared_tensors =
          settings.gpu_madvise_original_shared_tensors,
      .is_benchmark = settings.benchmark,
      .preferred_device_substr = settings.preferred_device_substr,
      .num_threads_to_upload = settings.num_threads_to_upload,
      .num_threads_to_compile = settings.num_threads_to_compile,
      .convert_weights_on_gpu = settings.convert_weights_on_gpu,
      .wait_for_weights_conversion_complete_in_benchmark =
          settings.wait_for_weights_conversion_complete_in_benchmark,
      .optimize_shader_compilation = settings.optimize_shader_compilation,
      .cache_compiled_shaders_only = settings.cache_compiled_shaders_only,
      .share_constant_tensors = settings.share_constant_tensors,
      .sampler_handles_input = settings.sampler_handles_input,
      .enable_speculative_decoding = settings.enable_speculative_decoding,
  };
  if (settings.conv_type == ConvType::kFloat) {
    advanced_settings.allow_src_quantized_fc_conv_ops = false;
  } else if (settings.conv_type == ConvType::kInt8) {
    advanced_settings.allow_src_quantized_fc_conv_ops = true;
  }
  if (advanced_settings != AdvancedSettings()) {
    engine_settings.GetMutableMainExecutorSettings().SetAdvancedSettings(
        advanced_settings);
  }

  ABSL_LOG(INFO) << "executor_settings: "
                 << engine_settings.GetMainExecutorSettings();

  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    ABSL_LOG(INFO) << "vision_executor_settings: "
                   << engine_settings.GetVisionExecutorSettings().value();
  } else {
    ABSL_LOG(INFO) << "vision_executor_settings: not set";
  }
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    ABSL_LOG(INFO) << "audio_executor_settings: "
                   << engine_settings.GetAudioExecutorSettings().value();
  } else {
    ABSL_LOG(INFO) << "audio_executor_settings: not set";
  }

  if (settings.benchmark) {
    if (settings.multi_turns && settings.async) {
      // TODO(b/483699181) - Support benchmarking for multi-turns and async.
      ABSL_LOG(ERROR) << "Benchmark with multi-turns and async do not show "
                         "results, use sync mode instead.";
    }

    litert::lm::proto::BenchmarkParams benchmark_params;
    benchmark_params.set_num_prefill_tokens(settings.benchmark_prefill_tokens);
    benchmark_params.set_num_decode_tokens(settings.benchmark_decode_tokens);
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }

  return engine_settings;
}

absl::StatusOr<std::unique_ptr<litert::lm::Engine>> CreateEngine(
    const LiteRtLmSettings& settings, const EngineSettings& engine_settings) {
  ABSL_LOG(INFO) << "Creating engine";
  ASSIGN_OR_RETURN(auto engine,
                   litert::lm::EngineFactory::CreateAny(
                       std::move(engine_settings), settings.input_prompt));
  if (settings.vision_backend.has_value()) {
    ASSIGN_OR_RETURN(auto vision_executor_properties,
                     engine->GetVisionExecutorProperties());
    ABSL_LOG(INFO) << "Vision executor properties: "
                   << vision_executor_properties;
  }
  if (settings.audio_backend.has_value()) {
    ASSIGN_OR_RETURN(auto audio_executor_properties,
                     engine->GetAudioExecutorProperties());
    ABSL_LOG(INFO) << "Audio executor properties: "
                   << audio_executor_properties;
  }
  return engine;
}

SessionConfig CreateSessionConfig(const LiteRtLmSettings& settings) {
  // Set the session config.
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  session_config.SetNumOutputCandidates(settings.num_output_candidates);
  const std::optional<Backend> sampler_backend = GetSamplerBackend(settings);
  if (sampler_backend.has_value()) {
    session_config.SetSamplerBackend(*sampler_backend);
  }
  if (settings.vision_backend.has_value()) {
    session_config.SetVisionModalityEnabled(true);
  }
  if (settings.audio_backend.has_value()) {
    session_config.SetAudioModalityEnabled(true);
  }
  return session_config;
}

// TODO(b/453071109): Check if returning the content list is more appropriate.
absl::Status BuildContentList(const std::vector<InputData>& input_data,
                              const LiteRtLmSettings& settings,
                              nlohmann::json& content_list) {
  // We expect the media path to be in the format of [image:/path/to/image.jpg]
  // or [audio:/path/to/audio.wav]
  //
  // So the prompt can be like:
  // 1. Briefly describe the two images [image:/path/to/image1.jpg] and
  // [image:/path/to/image2.jpg]
  //
  // 2. Transcribe the audio [audio:/path/to/audio.wav]
  //
  // 3. First transcribe the [audio:/path/to/audio.wav] then describe the
  // content in the [image:/path/to/image.jpg]
  RE2 re_media("\\[(image|audio):([^\\s\\]]+)\\]");  // Regex to find image
                                                     // or audio paths
  constexpr int kBracketShift = 3;  // account for [] in the string

  for (const auto& data : input_data) {
    if (const auto* text = std::get_if<InputText>(&data)) {
      ASSIGN_OR_RETURN(auto prompt_view, text->GetRawTextString());
      absl::string_view whole_prompt(prompt_view);
      int last_pos = 0;
      std::string media_type;
      std::string media_path;

      while (RE2::FindAndConsume(&prompt_view, re_media, &media_type,
                                 &media_path)) {
        if (!std::filesystem::exists(media_path)) {
          return absl::NotFoundError(absl::StrCat(
              "[ERROR] Media path ", media_path, " does not exist."));
        }
        // Calculate the position of the match in the original string
        const int media_string_size =
            media_type.size() + media_path.size() + kBracketShift;
        int match_pos =
            whole_prompt.size() - prompt_view.size() - media_string_size;
        // Add text part before the media path
        if (match_pos > last_pos) {
          content_list.push_back(
              {{"type", "text"},
               {"text", whole_prompt.substr(last_pos, match_pos - last_pos)}});
        }
        if (media_type == "image" && !settings.vision_backend.has_value()) {
          return absl::InvalidArgumentError(
              "Image backend is not specified. Please specify the vision "
              "backend "
              "with --vision_backend=<cpu|gpu>");
        }
        if (media_type == "audio" && !settings.audio_backend.has_value()) {
          return absl::InvalidArgumentError(
              "Audio backend is not specified. Please specify the audio "
              "backend "
              "with --audio_backend=<cpu|gpu>");
        }
        // Add media part
        content_list.push_back({{"type", media_type}, {"path", media_path}});
        last_pos = match_pos + media_string_size;
      }
      // Add any remaining text part
      if (!prompt_view.empty()) {
        content_list.push_back({{"type", "text"}, {"text", prompt_view}});
      }
    } else if (const auto* image = std::get_if<InputImage>(&data)) {
      ASSIGN_OR_RETURN(auto raw_bytes, image->GetRawImageBytes());
      content_list.push_back(
          {{"type", "image"}, {"blob", absl::Base64Escape(raw_bytes)}});
    }
    // TODO(b/453071109): Add support for audio.
  }

  return absl::OkStatus();
}

absl::Status RunLiteRtLm(const LiteRtLmSettings& settings,
                         std::vector<LitertLmMetrics>* metrics) {
  std::unique_ptr<FileLogSink> log_sink;
  if (settings.log_sink_file.has_value()) {
    log_sink = std::make_unique<FileLogSink>(settings.log_sink_file.value());
    absl::AddLogSink(log_sink.get());
  }

  ASSIGN_OR_RETURN(EngineSettings engine_settings,
                   CreateEngineSettings(settings));
  ASSIGN_OR_RETURN(auto engine, CreateEngine(settings, engine_settings));

  // Get the session config.
  SessionConfig session_config = CreateSessionConfig(settings);

  for (int i = 0; i < settings.num_iterations; ++i) {
    std::unique_ptr<tflite::profiling::memory::MemoryUsageMonitor> mem_monitor;
    if (settings.report_peak_memory_footprint) {
      mem_monitor =
          std::make_unique<tflite::profiling::memory::MemoryUsageMonitor>(
              kMemoryCheckIntervalMs);
      mem_monitor->Start();
    }

    // Session and Conversation are mutually exclusive. Only when
    // settings.score_target_text is set, we will create a Session to run the
    // scoring. Otherwise, we will create a Conversation.
    std::unique_ptr<Engine::Session> session;
    std::unique_ptr<Conversation> conversation;
      if (settings.score_target_text.has_value() &&
          !settings.score_target_text->empty()) {
      ABSL_LOG(INFO) << "Creating session";
      ASSIGN_OR_RETURN(session, engine->CreateSession(session_config));
      std::string input_prompt = settings.input_prompt;
      std::string score_target_text = settings.score_target_text.value();
      RETURN_IF_ERROR(RunScoreText(engine.get(), session.get(), input_prompt,
                                   {score_target_text},
                                   /*store_char_and_token_lengths=*/false)
                          .status());
    } else if (settings.use_session) {
      ABSL_LOG(INFO) << "Creating session";
      ASSIGN_OR_RETURN(session, engine->CreateSession(session_config));
      if (settings.multi_turns) {
        return absl::UnimplementedError(
            "Multi-turns is not supported with Session.");
      } else {
        RETURN_IF_ERROR(RunSingleTurnSession(settings.input_prompt, settings,
                                             engine.get(), session.get()));
      }
    } else {
      ABSL_LOG(INFO) << "Creating conversation";
      ASSIGN_OR_RETURN(auto conversation_config,
                       ConversationConfig::Builder()
                           .SetSessionConfig(session_config)
                           .Build(*engine));
      ASSIGN_OR_RETURN(conversation,
                       Conversation::Create(*engine, conversation_config));
      if (settings.multi_turns) {
        ABSL_LOG(INFO) << "Running multi-turns conversation";
        RETURN_IF_ERROR(RunMultiTurnConversation(settings, engine.get(),
                                                 conversation.get()));
      } else {
        ABSL_LOG(INFO) << "Running single-turn conversation";
        json content_list = json::array();
        std::vector<InputData> input_data;
        input_data.push_back(InputText(settings.input_prompt));
        RETURN_IF_ERROR(BuildContentList(input_data, settings, content_list));
        RETURN_IF_ERROR(RunSingleTurnConversation(content_list, settings,
                                                  engine.get(),
                                                  conversation.get())
                            .status());
      }
    }
    LitertLmMetrics metric;
    if (settings.benchmark) {
      absl::StatusOr<BenchmarkInfo> benchmark_info;
      if (conversation != nullptr) {
        benchmark_info = conversation->GetBenchmarkInfo();
      } else if (session != nullptr) {
        benchmark_info = session->GetBenchmarkInfo();
      } else {
        return absl::InternalError("No session or conversation to benchmark.");
      }
      if (benchmark_info.ok()) {
        LogBenchmarkInfo(*benchmark_info, settings);
        if (metrics != nullptr) {
          metric.benchmark_info = *benchmark_info;
        }
      }
    }

    // Manually resetting the session to ensure that memory usage from
    // `GetMemoryUsage()` is reporting idle engine state without active
    // sessions.
    conversation.reset();
    session.reset();

    if (settings.report_peak_memory_footprint) {
      float peak_mem_mb = 0.0f;
      float peak_private_mb = 0.0f;
      if (mem_monitor != nullptr) {
        mem_monitor->Stop();
        peak_mem_mb = mem_monitor->GetPeakMemUsageInMB();
        peak_private_mb = mem_monitor->GetPeakPrivateFootprintInMB();
        if (metrics != nullptr) {
          metric.peak_mem_mb = peak_mem_mb;
          metric.peak_private_mb = peak_private_mb;
        }
      }
      LogMemoryUsage(settings, peak_mem_mb, peak_private_mb);
    }
    if (metrics != nullptr) {
      metrics->push_back(metric);
    }
  }

  if (log_sink) {
    absl::RemoveLogSink(log_sink.get());
  }

  return absl::OkStatus();
}

}  // namespace lm
}  // namespace litert
