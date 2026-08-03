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

#include "c/engine.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#if defined(_WIN32)
#include <io.h>
#endif
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/logging.h"
#include "runtime/util/scoped_file.h"

struct LiteRtLmInputData {
  explicit LiteRtLmInputData(litert::lm::InputData d) : data(std::move(d)) {}
  litert::lm::InputData data;
};

struct LiteRtLmSamplerParams {
  LiteRtLmSamplerType type;
  int32_t top_k;
  float top_p;
  float temperature;
  int32_t seed;
};

namespace {

absl::AnyInvocable<void(absl::StatusOr<litert::lm::Responses>)> CreateCallback(
    LiteRtLmStreamCallback callback, void* callback_data) {
  return [callback,
          callback_data](absl::StatusOr<litert::lm::Responses> responses) {
    if (!responses.ok()) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               responses.status().ToString().c_str());
      return;
    }
    if (responses->GetTaskState() == litert::lm::TaskState::kDone) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               /*error_message=*/nullptr);
    } else if (responses->GetTaskState() ==
               litert::lm::TaskState::kMaxNumTokensReached) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               "Max number of tokens reached.");
    } else if (responses->GetTaskState() == litert::lm::TaskState::kCancelled) {
      callback(callback_data, /*text=*/nullptr, /*is_final=*/true,
               "CANCELLED.");
    } else {
      for (const auto& text : responses->GetTexts()) {
        callback(callback_data, text.data(), /*is_final=*/false,
                 /*error_message=*/nullptr);
      }
    }
  };
}

absl::AnyInvocable<void(absl::StatusOr<litert::lm::Message>)>
CreateConversationCallback(LiteRtLmStreamCallback callback, void* user_data) {
  return [callback, user_data](absl::StatusOr<litert::lm::Message> message) {
    if (!message.ok()) {
      std::string error_str = message.status().ToString();
      callback(user_data, nullptr, true, const_cast<char*>(error_str.c_str()));
      return;
    }
    if (message->empty()) {  // End of stream marker
      callback(user_data, nullptr, true, nullptr);
    } else {
      std::string json_str = message->dump();
      callback(user_data, const_cast<char*>(json_str.c_str()), false, nullptr);
    }
  };
}

std::optional<litert::lm::DataProcessorArguments> GetDataProcessorArguments(
    const litert::lm::Conversation* conversation,
    const int visual_token_budget) {
  bool is_gemma4 = conversation->GetConfig()
                       .GetSessionConfig()
                       .GetLlmModelType()
                       .has_gemma4();
  if (is_gemma4) {
    return litert::lm::Gemma4DataProcessorArguments{.visual_token_budget =
                                                        visual_token_budget};
  }
  return std::nullopt;
}

litert::lm::OptionalArgs CreateOptionalArgs(
    const litert::lm::Conversation* conversation, const char* extra_context,
    std::optional<int> visual_token_budget,
    std::optional<int> max_output_tokens) {
  litert::lm::OptionalArgs litert_lm_optional_args;
  if (extra_context) {
    auto extra_context_json =
        nlohmann::ordered_json::parse(extra_context, nullptr, false);
    if (!extra_context_json.is_null() && !extra_context_json.empty()) {
      litert_lm_optional_args.extra_context = extra_context_json;
    }
  }
  if (visual_token_budget.has_value()) {
    litert_lm_optional_args.args =
        GetDataProcessorArguments(conversation, *visual_token_budget);
  }
  if (max_output_tokens.has_value()) {
    litert_lm_optional_args.max_output_tokens = max_output_tokens;
  }
  return litert_lm_optional_args;
}

absl::StatusOr<std::vector<litert::lm::InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs) {
  std::vector<litert::lm::InputData> engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i] != nullptr) {
      auto copy_status = litert::lm::CreateInputDataCopy(inputs[i]->data);
      if (!copy_status.ok()) {
        return copy_status.status();
      }
      engine_inputs.push_back(std::move(*copy_status));
    }
  }
  return engine_inputs;
}

}  // namespace

using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::Engine;
using ::litert::lm::EngineFactory;
using ::litert::lm::EngineSettings;
using ::litert::lm::OptionalArgs;

using ::litert::lm::ScopedFile;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::litert::lm::Responses;
using ::litert::lm::SessionConfig;
using ::litert::lm::proto::SamplerParameters;

LiteRtLmInputData* litert_lm_input_data_create(LiteRtLmInputDataType type,
                                               const void* data, size_t size) {
  switch (type) {
    case kLiteRtLmInputDataTypeText:
      return std::make_unique<LiteRtLmInputData>(
                 litert::lm::InputText(
                     std::string(static_cast<const char*>(data), size)))
          .release();
    case kLiteRtLmInputDataTypeImage:
      return std::make_unique<LiteRtLmInputData>(
                 litert::lm::InputImage(
                     std::string(static_cast<const char*>(data), size)))
          .release();
    case kLiteRtLmInputDataTypeImageEnd:
      return std::make_unique<LiteRtLmInputData>(litert::lm::InputImageEnd())
          .release();
    case kLiteRtLmInputDataTypeAudio:
      return std::make_unique<LiteRtLmInputData>(
                 litert::lm::InputAudio(
                     std::string(static_cast<const char*>(data), size)))
          .release();
    case kLiteRtLmInputDataTypeAudioEnd:
      return std::make_unique<LiteRtLmInputData>(litert::lm::InputAudioEnd())
          .release();
    default:
      return nullptr;
  }
}

void litert_lm_input_data_delete(LiteRtLmInputData* input_data) {
  delete input_data;
}

struct LiteRtLmEngineSettings {
  std::unique_ptr<EngineSettings> settings;
};

static LiteRtLmEngineSettings* CreateEngineSettingsHelper(
    ModelAssets model_assets, absl::string_view backend_str,
    absl::string_view vision_backend_str, absl::string_view audio_backend_str) {
  auto backend = litert::lm::GetBackendFromString(backend_str);
  if (!backend.ok()) {
    ABSL_LOG(ERROR) << "Failed to parse backend: " << backend.status();
    return nullptr;
  }

  std::optional<litert::lm::Backend> vision_backend;
  if (!vision_backend_str.empty()) {
    auto backend = litert::lm::GetBackendFromString(vision_backend_str);
    if (!backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse vision backend: " << backend.status();
      return nullptr;
    }
    vision_backend = *backend;
  }

  std::optional<litert::lm::Backend> audio_backend;
  if (!audio_backend_str.empty()) {
    auto backend = litert::lm::GetBackendFromString(audio_backend_str);
    if (!backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse audio backend: " << backend.status();
      return nullptr;
    }
    audio_backend = *backend;
  }

  auto engine_settings = EngineSettings::CreateDefault(
      std::move(model_assets), *backend, vision_backend, audio_backend);
  if (!engine_settings.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine settings: "
                    << engine_settings.status();
    return nullptr;
  }

  auto* c_settings = new LiteRtLmEngineSettings;
  c_settings->settings =
      std::make_unique<EngineSettings>(std::move(*engine_settings));
  return c_settings;
}

struct LiteRtLmEngine {
  std::unique_ptr<Engine> engine;
};

struct LiteRtLmSession {
  std::unique_ptr<Engine::Session> session;
};

struct LiteRtLmResponses {
  Responses responses;
};

struct LiteRtLmBenchmarkInfo {
  litert::lm::BenchmarkInfo benchmark_info;
};

struct LiteRtLmConversation {
  std::unique_ptr<Conversation> conversation;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_message_to_string`. This ties the lifetime
  // of the returned `const char*` to the `LiteRtLmConversation` object,
  // ensuring memory safety for the C API caller without requiring explicit
  // per-call deallocation.
  std::string last_rendered_message;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_preface_to_string`.
  std::string last_rendered_preface;
};

struct LiteRtLmJsonResponse {
  std::string json_string;
};

// TODO: b/483172229 - Migrate to use SessionConfig instead of unique_ptr to
// SessionConfig for consistency and efficiency.
struct LiteRtLmSessionConfig {
  std::unique_ptr<SessionConfig> config;
};

struct LiteRtLmConversationConfig {
  std::optional<SessionConfig> session_config;
  std::string system_message_json;
  std::string tools_json;
  std::string messages_json;
  std::string extra_context_json;
  bool enable_constrained_decoding = false;
  bool filter_channel_content_from_kv_cache = false;
  bool stream_tool_calls = false;
  std::string stream_tool_calls_channel_name = "tool_call";
};

struct LiteRtLmConversationOptionalArgs {
  std::optional<int> visual_token_budget;
  std::optional<int> max_output_tokens;
};

struct LiteRtLmDetokenizeResult {
  std::string text;
};

struct LiteRtLmTokenizeResult {
  std::vector<int> tokens;
};

struct LiteRtLmTokenUnion {
  litert::lm::proto::TokenUnion token_union;
};

struct LiteRtLmTokenUnions {
  std::vector<litert::lm::proto::TokenUnion> tokens;
};

extern "C" {

void litert_lm_set_min_log_level(int level) {
  litert::lm::SetMinLogSeverity(static_cast<litert::lm::LogSeverity>(level));
}

SamplerParameters::Type ToSamplerParametersType(LiteRtLmSamplerType type) {
  switch (type) {
    case kLiteRtLmSamplerTypeTopK:
      return SamplerParameters::TOP_K;
    case kLiteRtLmSamplerTypeTopP:
      return SamplerParameters::TOP_P;
    case kLiteRtLmSamplerTypeGreedy:
      return SamplerParameters::GREEDY;
  }
  return SamplerParameters::TYPE_UNSPECIFIED;
}

LiteRtLmSamplerParams* litert_lm_sampler_params_create(
    LiteRtLmSamplerType type) {
  auto params = std::make_unique<LiteRtLmSamplerParams>();
  params->type = type;
  params->top_k = 0;
  params->top_p = 0.0f;
  params->temperature = 0.0f;
  params->seed = 0;
  return params.release();
}

void litert_lm_sampler_params_delete(LiteRtLmSamplerParams* params) {
  delete params;
}

void litert_lm_sampler_params_set_top_k(LiteRtLmSamplerParams* params,
                                        int32_t top_k) {
  if (params) {
    params->top_k = top_k;
  }
}

void litert_lm_sampler_params_set_top_p(LiteRtLmSamplerParams* params,
                                        float top_p) {
  if (params) {
    params->top_p = top_p;
  }
}

void litert_lm_sampler_params_set_temperature(LiteRtLmSamplerParams* params,
                                              float temperature) {
  if (params) {
    params->temperature = temperature;
  }
}

void litert_lm_sampler_params_set_seed(LiteRtLmSamplerParams* params,
                                       int32_t seed) {
  if (params) {
    params->seed = seed;
  }
}

LiteRtLmSessionConfig* litert_lm_session_config_create() {
  auto* c_config = new LiteRtLmSessionConfig;
  c_config->config =
      std::make_unique<SessionConfig>(SessionConfig::CreateDefault());
  return c_config;
}

void litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens) {
  if (config && config->config) {
    config->config->SetMaxOutputTokens(max_output_tokens);
  }
}

void litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig* config, bool apply_prompt_template) {
  if (config && config->config) {
    config->config->SetApplyPromptTemplateInSession(apply_prompt_template);
  }
}

void litert_lm_session_config_set_sampler_params(
    LiteRtLmSessionConfig* config,
    const LiteRtLmSamplerParams* sampler_params) {
  if (config && config->config && sampler_params) {
    SamplerParameters& params = config->config->GetMutableSamplerParams();

    params.set_type(ToSamplerParametersType(sampler_params->type));

    params.set_k(sampler_params->top_k);
    params.set_p(sampler_params->top_p);
    params.set_temperature(sampler_params->temperature);
    params.set_seed(sampler_params->seed);
  }
}

void litert_lm_session_config_delete(LiteRtLmSessionConfig* config) {
  delete config;
}

int litert_lm_session_config_set_lora_path(LiteRtLmSessionConfig* config,
                                           const char* lora_path) {
  if (!config || !config->config || !lora_path) {
    return -1;
  }
  absl::string_view path_view(lora_path);
  if (path_view.empty()) {
    return -1;
  }
  auto lora_file = litert::lm::ScopedFile::Open(path_view);
  if (!lora_file.ok()) {
    ABSL_LOG(ERROR) << "Failed to open LoRA file: " << lora_file.status();
    return -1;
  }
  config->config->SetScopedLoraFile(
      std::make_shared<litert::lm::ScopedFile>(std::move(*lora_file)));
  return 0;
}

int litert_lm_session_config_set_audio_lora_path(LiteRtLmSessionConfig* config,
                                                 const char* audio_lora_path) {
  if (!config || !config->config || !audio_lora_path) {
    return -1;
  }
  absl::string_view path_view(audio_lora_path);
  if (path_view.empty()) {
    return -1;
  }
  auto lora_file = litert::lm::ScopedFile::Open(path_view);
  if (!lora_file.ok()) {
    ABSL_LOG(ERROR) << "Failed to open Audio LoRA file: " << lora_file.status();
    return -1;
  }
  config->config->SetAudioScopedLoraFile(
      std::make_shared<litert::lm::ScopedFile>(std::move(*lora_file)));
  return 0;
}

LiteRtLmConversationConfig* litert_lm_conversation_config_create() {
  return new LiteRtLmConversationConfig;
}

void litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config) {
  if (config && session_config && session_config->config) {
    config->session_config = *session_config->config;
  }
}

void litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json) {
  if (config && system_message_json) {
    config->system_message_json = system_message_json;
  }
}

void litert_lm_conversation_config_set_tools(LiteRtLmConversationConfig* config,
                                             const char* tools_json) {
  if (config && tools_json) {
    config->tools_json = tools_json;
  }
}

void litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json) {
  if (config && messages_json) {
    config->messages_json = messages_json;
  }
}

void litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json) {
  if (config && extra_context_json) {
    config->extra_context_json = extra_context_json;
  }
}

void litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding) {
  if (config) {
    config->enable_constrained_decoding = enable_constrained_decoding;
  }
}

void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache) {
  if (config) {
    config->filter_channel_content_from_kv_cache =
        filter_channel_content_from_kv_cache;
  }
}

void litert_lm_conversation_config_set_stream_tool_calls(
    LiteRtLmConversationConfig* config, bool stream_tool_calls,
    const char* channel_name) {
  if (config) {
    config->stream_tool_calls = stream_tool_calls;
    if (channel_name != nullptr) {
      config->stream_tool_calls_channel_name = channel_name;
    }
  }
}

void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config) {
  delete config;
}

LiteRtLmConversationOptionalArgs*
litert_lm_conversation_optional_args_create() {
  return new LiteRtLmConversationOptionalArgs;
}

void litert_lm_conversation_optional_args_set_visual_token_budget(
    LiteRtLmConversationOptionalArgs* args, int visual_token_budget) {
  if (args) {
    args->visual_token_budget = visual_token_budget;
  }
}

void litert_lm_conversation_optional_args_set_max_output_tokens(
    LiteRtLmConversationOptionalArgs* args, int max_output_tokens) {
  if (args) {
    args->max_output_tokens = max_output_tokens;
  }
}

void litert_lm_conversation_optional_args_delete(
    LiteRtLmConversationOptionalArgs* args) {
  delete args;
}

LiteRtLmEngineSettings* litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str) {
  auto model_assets = ModelAssets::Create(model_path);
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets: "
                    << model_assets.status();
    return nullptr;
  }
  return CreateEngineSettingsHelper(
      std::move(*model_assets), absl::NullSafeStringView(backend_str),
      absl::NullSafeStringView(vision_backend_str),
      absl::NullSafeStringView(audio_backend_str));
}

LiteRtLmEngineSettings*
litert_lm_engine_settings_create_from_raw_file_descriptor(
    int fd, const char* backend_str, const char* vision_backend_str,
    const char* audio_backend_str) {
  if (fd < 0) {
    ABSL_LOG(ERROR) << "Invalid file descriptor: " << fd;
    return nullptr;
  }
  auto model_assets = ModelAssets::Create(
#if defined(_WIN32)
      std::make_shared<litert::lm::ScopedFile>(litert::lm::ScopedFile(
          reinterpret_cast<litert::lm::ScopedFile::PlatformFile>(
              _get_osfhandle(fd)))));
#else
      std::make_shared<litert::lm::ScopedFile>(litert::lm::ScopedFile(fd)));
#endif
  if (!model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create model assets from raw FD: "
                    << model_assets.status();
    return nullptr;
  }
  ABSL_LOG(INFO) << "LiteRT-LM successfully created EngineSettings directly "
                    "from raw File Descriptor: "
                 << fd;
  return CreateEngineSettingsHelper(
      std::move(*model_assets), absl::NullSafeStringView(backend_str),
      absl::NullSafeStringView(vision_backend_str),
      absl::NullSafeStringView(audio_backend_str));
}
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings* settings) {
  delete settings;
}

void litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
        max_num_tokens);
  }
}

void litert_lm_engine_settings_set_num_threads(LiteRtLmEngineSettings* settings,
                                               int num_threads) {
  if (settings && settings->settings) {
    auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
    auto config = main_settings.MutableBackendConfig<litert::lm::CpuConfig>();
    if (config.ok()) {
      litert::lm::CpuConfig cpu_config = *config;
      cpu_config.number_of_threads = num_threads;
      main_settings.SetBackendConfig(cpu_config);
    } else {
      ABSL_LOG(WARNING) << "Failed to get CpuConfig to set num threads: "
                        << config.status();
    }
  }
}

void litert_lm_engine_settings_set_audio_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads) {
  if (settings && settings->settings) {
    auto& audio_settings =
        settings->settings->GetMutableAudioExecutorSettings();
    if (audio_settings.has_value()) {
      audio_settings->SetNumThreads(num_threads);
    }
  }
}

void litert_lm_engine_settings_set_parallel_file_section_loading(
    LiteRtLmEngineSettings* settings, bool parallel_file_section_loading) {
  if (settings && settings->settings) {
    settings->settings->SetParallelFileSectionLoading(
        parallel_file_section_loading);
  }
}

void litert_lm_engine_settings_set_max_num_images(
    LiteRtLmEngineSettings* settings, int max_num_images) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetMaxNumImages(
        max_num_images);
  }
}

void litert_lm_engine_settings_set_cache_dir(LiteRtLmEngineSettings* settings,
                                             const char* cache_dir) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir);

    if (settings->settings->GetVisionExecutorSettings().has_value()) {
      settings->settings->GetMutableVisionExecutorSettings()->SetCacheDir(
          cache_dir);
    }

    if (settings->settings->GetAudioExecutorSettings().has_value()) {
      settings->settings->GetMutableAudioExecutorSettings()->SetCacheDir(
          cache_dir);
    }
  }
}

void litert_lm_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEngineSettings* settings, const char* lib_dir) {
  if (settings && settings->settings && lib_dir) {
    settings->settings->GetMutableMainExecutorSettings()
        .SetLitertDispatchLibDir(lib_dir);
  }
}

void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings) {
  if (settings && settings->settings) {
    settings->settings->GetMutableBenchmarkParams();
  }
}

void litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings* settings, int num_prefill_tokens) {
  if (settings && settings->settings) {
    settings->settings->GetMutableBenchmarkParams().set_num_prefill_tokens(
        num_prefill_tokens);
  }
}

void litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings* settings, int num_decode_tokens) {
  if (settings && settings->settings) {
    settings->settings->GetMutableBenchmarkParams().set_num_decode_tokens(
        num_decode_tokens);
  }
}

void litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding) {
  if (settings && settings->settings) {
    auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
    auto advanced_settings = main_settings.GetAdvancedSettings().value_or(
        litert::lm::AdvancedSettings());
    advanced_settings.enable_speculative_decoding = enable_speculative_decoding;
    main_settings.SetAdvancedSettings(advanced_settings);
  }
}

void litert_lm_engine_settings_set_lora_rank(LiteRtLmEngineSettings* settings,
                                             int lora_rank) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetLoraRank(lora_rank);
  }
}

int litert_lm_engine_settings_set_supported_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks) {
  if (!settings || !settings->settings || !lora_ranks || num_ranks == 0) {
    return -1;
  }
  std::vector<uint32_t> ranks;
  ranks.reserve(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    ranks.push_back(static_cast<uint32_t>(lora_ranks[i]));
  }
  auto status = settings->settings->GetMutableMainExecutorSettings()
                    .SetSupportedLoraRanks(ranks);
  return status.ok() ? 0 : -1;
}

void litert_lm_engine_settings_set_audio_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank) {
  if (settings && settings->settings &&
      settings->settings->GetAudioExecutorSettings().has_value()) {
    settings->settings->GetMutableAudioExecutorSettings()->SetLoraRank(
        lora_rank);
  }
}

int litert_lm_engine_settings_set_supported_audio_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks) {
  if (!settings || !settings->settings || !lora_ranks || num_ranks == 0) {
    return -1;
  }
  if (!settings->settings->GetAudioExecutorSettings().has_value()) {
    return -1;
  }
  std::vector<uint32_t> ranks;
  ranks.reserve(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    ranks.push_back(static_cast<uint32_t>(lora_ranks[i]));
  }
  auto status = settings->settings->GetMutableAudioExecutorSettings()
                    ->SetSupportedLoraRanks(ranks);
  return status.ok() ? 0 : -1;
}

void litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings, int activation_data_type_int) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetActivationDataType(
        static_cast<litert::lm::ActivationDataType>(activation_data_type_int));
  }
}

void litert_lm_engine_settings_set_prefill_chunk_size(
    LiteRtLmEngineSettings* settings, int prefill_chunk_size) {
  if (settings && settings->settings) {
    auto& main_settings = settings->settings->GetMutableMainExecutorSettings();
    auto config = main_settings.MutableBackendConfig<litert::lm::CpuConfig>();
    if (!config.ok()) {
      ABSL_LOG(WARNING) << "Failed to get CpuConfig to set prefill chunk size: "
                        << config.status();
      return;
    }
    config->prefill_chunk_size = prefill_chunk_size;
    main_settings.SetBackendConfig(*config);
  }
}

LiteRtLmEngine* litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings) {
  if (!settings || !settings->settings) {
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<Engine>> engine =
      EngineFactory::CreateDefault(*settings->settings);

  if (!engine.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine: " << engine.status();
    return nullptr;
  }

  auto* c_engine = new LiteRtLmEngine;
  c_engine->engine = *std::move(engine);
  return c_engine;
}

void litert_lm_engine_delete(LiteRtLmEngine* engine) { delete engine; }

LiteRtLmSession* litert_lm_engine_create_session(
    LiteRtLmEngine* engine, LiteRtLmSessionConfig* config) {
  if (!engine || !engine->engine) {
    return nullptr;
  }

  SessionConfig session_config = config && config->config
                                     ? *config->config
                                     : SessionConfig::CreateDefault();
  if (engine->engine->GetEngineSettings()
          .GetAudioExecutorSettings()
          .has_value()) {
    session_config.SetAudioModalityEnabled(true);
  }
  if (engine->engine->GetEngineSettings()
          .GetVisionExecutorSettings()
          .has_value()) {
    session_config.SetVisionModalityEnabled(true);
  }

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      engine->engine->CreateSession(session_config);
  if (!session.ok()) {
    ABSL_LOG(ERROR) << "Failed to create session: " << session.status();
    return nullptr;
  }

  auto* c_session = new LiteRtLmSession;
  c_session->session = *std::move(session);
  return c_session;
}

void litert_lm_session_delete(LiteRtLmSession* session) { delete session; }

void litert_lm_session_cancel_process(LiteRtLmSession* session) {
  if (session && session->session) {
    session->session->CancelProcess();
  }
}

LiteRtLmResponses* litert_lm_session_run_text_scoring(
    LiteRtLmSession* session, const char** target_text, size_t num_targets,
    bool store_token_lengths) {
  if (!session || !session->session || !target_text || num_targets <= 0) {
    return nullptr;
  }
  std::vector<absl::string_view> target_text_views;
  target_text_views.reserve(num_targets);
  for (size_t i = 0; i < num_targets; ++i) {
    target_text_views.push_back(target_text[i]);
  }
  auto responses =
      session->session->RunTextScoring(target_text_views, store_token_lengths);
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to run text scoring: " << responses.status();
    return nullptr;
  }
  auto* c_responses = new LiteRtLmResponses{std::move(*responses)};
  if (c_responses->responses.GetTexts().empty()) {
    auto& mutable_texts = c_responses->responses.GetMutableTexts();
    mutable_texts.reserve(num_targets);
    for (size_t i = 0; i < num_targets; ++i) {
      mutable_texts.emplace_back(target_text[i]);
    }
  }
  return c_responses;
}

int litert_lm_session_run_prefill(LiteRtLmSession* session,
                                  const LiteRtLmInputData* const* inputs,
                                  size_t num_inputs) {
  if (!session || !session->session || !inputs || num_inputs <= 0) {
    return -1;
  }
  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to copy inputs: " << engine_inputs.status();
    return -1;
  }
  auto status = session->session->RunPrefill(*engine_inputs);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run prefill: " << status;
    return -1;
  }
  return 0;
}

LiteRtLmResponses* litert_lm_session_run_decode(LiteRtLmSession* session) {
  if (!session || !session->session) {
    return nullptr;
  }
  auto responses = session->session->RunDecode();
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to run decode: " << responses.status();
    return nullptr;
  }
  return new LiteRtLmResponses{std::move(*responses)};
}

int litert_lm_session_run_decode_async(LiteRtLmSession* session,
                                       LiteRtLmStreamCallback callback,
                                       void* callback_data) {
  if (!session || !session->session) {
    return -1;
  }
  auto status =
      session->session->RunDecodeAsync(CreateCallback(callback, callback_data));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start decode stream: " << status.status();
    return static_cast<int>(status.status().code());
  }
  return 0;
}

LiteRtLmResponses* litert_lm_session_generate_content(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs) {
  if (!session || !session->session) {
    return nullptr;
  }
  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to copy inputs: " << engine_inputs.status();
    return nullptr;
  }
  auto responses = session->session->GenerateContent(std::move(*engine_inputs));
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to generate content: " << responses.status();
    return nullptr;
  }

  auto* c_responses = new LiteRtLmResponses{std::move(*responses)};
  return c_responses;
}

int litert_lm_session_generate_content_stream(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmStreamCallback callback, void* callback_data) {
  if (!session || !session->session) {
    return -1;
  }
  auto engine_inputs = ToEngineInputData(inputs, num_inputs);
  if (!engine_inputs.ok()) {
    ABSL_LOG(ERROR) << "Failed to copy inputs: " << engine_inputs.status();
    return -1;
  }

  absl::Status status = session->session->GenerateContentStream(
      std::move(*engine_inputs), CreateCallback(callback, callback_data));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start content stream: " << status;
    // No need to delete callbacks, unique_ptr handles it if not moved.
    return static_cast<int>(status.code());
  }
  return 0;  // The call is non-blocking and returns immediately.
}

void litert_lm_responses_delete(LiteRtLmResponses* responses) {
  delete responses;
}

int litert_lm_responses_get_num_candidates(const LiteRtLmResponses* responses) {
  if (!responses) {
    return 0;
  }
  const auto& r = responses->responses;
  size_t num_candidates = r.GetTexts().size();
  if (num_candidates == 0) {
    num_candidates = r.GetScores().size();
  }
  if (num_candidates == 0 && r.GetTokenLengths().has_value()) {
    num_candidates = r.GetTokenLengths()->size();
  }
  return static_cast<int>(num_candidates);
}

const char* litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index) {
  if (!responses || index < 0 ||
      index >= responses->responses.GetTexts().size()) {
    return nullptr;
  }

  // The string_view's data is valid as long as the responses object is alive.
  return responses->responses.GetTexts()[index].data();
}

bool litert_lm_responses_has_score_at(const LiteRtLmResponses* responses,
                                      int index) {
  if (!responses || index < 0 ||
      index >= responses->responses.GetScores().size()) {
    return false;
  }
  return true;
}

float litert_lm_responses_get_score_at(const LiteRtLmResponses* responses,
                                       int index) {
  if (!litert_lm_responses_has_score_at(responses, index)) {
    return 0.0f;
  }
  return responses->responses.GetScores()[index];
}

bool litert_lm_responses_has_token_length_at(const LiteRtLmResponses* responses,
                                             int index) {
  if (!responses || !responses->responses.GetTokenLengths().has_value() ||
      index < 0 || index >= responses->responses.GetTokenLengths()->size()) {
    return false;
  }
  return true;
}

int litert_lm_responses_get_token_length_at(const LiteRtLmResponses* responses,
                                            int index) {
  if (!litert_lm_responses_has_token_length_at(responses, index)) {
    return 0;
  }
  return (*responses->responses.GetTokenLengths())[index];
}

bool litert_lm_responses_has_token_scores_at(const LiteRtLmResponses* responses,
                                             int index) {
  if (!responses || !responses->responses.GetTokenScores().has_value() ||
      index < 0 || index >= responses->responses.GetTokenScores()->size()) {
    return false;
  }
  return true;
}

int litert_lm_responses_get_num_token_scores_at(
    const LiteRtLmResponses* responses, int index) {
  if (!litert_lm_responses_has_token_scores_at(responses, index)) {
    return 0;
  }
  return (*responses->responses.GetTokenScores())[index].size();
}

const float* litert_lm_responses_get_token_scores_at(
    const LiteRtLmResponses* responses, int index) {
  if (!litert_lm_responses_has_token_scores_at(responses, index)) {
    return nullptr;
  }
  return (*responses->responses.GetTokenScores())[index].data();
}

LiteRtLmBenchmarkInfo* litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session) {
  if (!session || !session->session) {
    return nullptr;
  }
  auto benchmark_info = session->session->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get benchmark info: "
                    << benchmark_info.status();
    return nullptr;
  }
  return new LiteRtLmBenchmarkInfo{std::move(*benchmark_info)};
}

void litert_lm_benchmark_info_delete(LiteRtLmBenchmarkInfo* benchmark_info) {
  delete benchmark_info;
}

double litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0.0;
  }
  return benchmark_info->benchmark_info.GetTimeToFirstToken();
}

double litert_lm_benchmark_info_get_total_init_time_in_second(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0.0;
  }
  double total_init_time_ms = 0.0;
  for (const auto& phase : benchmark_info->benchmark_info.GetInitPhases()) {
    total_init_time_ms += absl::ToDoubleMilliseconds(phase.second);
  }
  return total_init_time_ms / 1000.0;
}

int litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0;
  }
  return benchmark_info->benchmark_info.GetTotalPrefillTurns();
}

int litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info) {
  if (!benchmark_info) {
    return 0;
  }
  return benchmark_info->benchmark_info.GetTotalDecodeTurns();
}

int litert_lm_benchmark_info_get_prefill_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index) {
  if (!benchmark_info) {
    return 0;
  }
  auto turn = benchmark_info->benchmark_info.GetPrefillTurn(index);
  if (!turn.ok()) {
    return 0;
  }
  return static_cast<int>(turn->num_tokens);
}

int litert_lm_benchmark_info_get_decode_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index) {
  if (!benchmark_info) {
    return 0;
  }
  auto turn = benchmark_info->benchmark_info.GetDecodeTurn(index);
  if (!turn.ok()) {
    return 0;
  }
  return static_cast<int>(turn->num_tokens);
}

double litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index) {
  if (!benchmark_info) {
    return 0.0;
  }
  return benchmark_info->benchmark_info.GetPrefillTokensPerSec(index);
}

double litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index) {
  if (!benchmark_info) {
    return 0.0;
  }
  return benchmark_info->benchmark_info.GetDecodeTokensPerSec(index);
}

LiteRtLmConversation* litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* c_config) {
  if (!engine || !engine->engine) {
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<Conversation>> conversation;
  if (c_config) {
    litert::lm::JsonPreface json_preface;
    if (!c_config->system_message_json.empty()) {
      nlohmann::ordered_json system_message;
      system_message["role"] = "system";
      auto content = nlohmann::ordered_json::parse(
          c_config->system_message_json, nullptr, false);
      if (content.is_discarded()) {
        system_message["content"] = c_config->system_message_json;
      } else {
        system_message["content"] = content;
      }
      json_preface.messages = nlohmann::ordered_json::array({system_message});
    }

    if (!c_config->messages_json.empty()) {
      auto messages = nlohmann::ordered_json::parse(c_config->messages_json,
                                                    nullptr, false);
      if (messages.is_discarded()) {
        ABSL_LOG(ERROR) << "Failed to parse messages JSON.";
      } else if (!messages.is_array()) {
        ABSL_LOG(ERROR) << "Messages JSON is not an array.";
      } else {
        if (json_preface.messages.is_array()) {
          json_preface.messages.insert(json_preface.messages.end(),
                                       messages.begin(), messages.end());
        } else {
          json_preface.messages = std::move(messages);
        }
      }
    }

    if (!c_config->tools_json.empty()) {
      auto tool_json_parsed =
          nlohmann::ordered_json::parse(c_config->tools_json, nullptr, false);
      if (!tool_json_parsed.is_discarded() && tool_json_parsed.is_array()) {
        json_preface.tools = tool_json_parsed;
      } else {
        ABSL_LOG(ERROR) << "Failed to parse tools JSON or not an array: "
                        << c_config->tools_json;
      }
    }

    if (!c_config->extra_context_json.empty()) {
      auto extra_context_parsed = nlohmann::ordered_json::parse(
          c_config->extra_context_json, nullptr, false);
      if (!extra_context_parsed.is_discarded() &&
          extra_context_parsed.is_object()) {
        json_preface.extra_context = std::move(extra_context_parsed);
      } else {
        ABSL_LOG(ERROR)
            << "Failed to parse extra context JSON or not an object: "
            << c_config->extra_context_json;
      }
    }

    auto builder = litert::lm::ConversationConfig::Builder();
    SessionConfig session_config = c_config->session_config
                                       ? *c_config->session_config
                                       : SessionConfig::CreateDefault();
    if (engine->engine->GetEngineSettings()
            .GetAudioExecutorSettings()
            .has_value()) {
      session_config.SetAudioModalityEnabled(true);
    }
    if (engine->engine->GetEngineSettings()
            .GetVisionExecutorSettings()
            .has_value()) {
      session_config.SetVisionModalityEnabled(true);
    }
    builder.SetSessionConfig(session_config);

    builder.SetPreface(json_preface);
    builder.SetEnableConstrainedDecoding(c_config->enable_constrained_decoding);
    builder.SetFilterChannelContentFromKvCache(
        c_config->filter_channel_content_from_kv_cache);
    builder.SetStreamToolCalls(c_config->stream_tool_calls,
                               c_config->stream_tool_calls_channel_name);
    auto config = builder.Build(*engine->engine);

    if (!config.ok()) {
      ABSL_LOG(ERROR) << "Failed to create conversation config: "
                      << config.status();
      return nullptr;
    }
    conversation = Conversation::Create(*engine->engine, *config);
  } else {
    auto default_conversation_config =
        ConversationConfig::CreateDefault(*engine->engine);
    if (!default_conversation_config.ok()) {
      ABSL_LOG(ERROR) << "Failed to create default conversation config: "
                      << default_conversation_config.status();
      return nullptr;
    }
    conversation =
        Conversation::Create(*engine->engine, *default_conversation_config);
  }

  if (!conversation.ok()) {
    ABSL_LOG(ERROR) << "Failed to create conversation: "
                    << conversation.status();
    return nullptr;
  }
  auto* c_conversation = new LiteRtLmConversation;
  c_conversation->conversation = *std::move(conversation);
  return c_conversation;
}

void litert_lm_conversation_delete(LiteRtLmConversation* conversation) {
  delete conversation;
}

LiteRtLmConversation* litert_lm_conversation_clone(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  auto cloned = conversation->conversation->Clone();
  if (!cloned.ok()) {
    ABSL_LOG(ERROR) << "Failed to clone conversation: " << cloned.status();
    return nullptr;
  }
  auto c_conversation = std::make_unique<LiteRtLmConversation>();
  c_conversation->conversation = std::move(*cloned);
  return c_conversation.release();
}

LiteRtLmJsonResponse* litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return nullptr;
  }

  OptionalArgs litert_lm_optional_args = CreateOptionalArgs(
      conversation->conversation.get(), extra_context,
      optional_args ? optional_args->visual_token_budget : std::nullopt,
      optional_args ? optional_args->max_output_tokens : std::nullopt);

  auto response = conversation->conversation->SendMessage(
      json_message, std::move(litert_lm_optional_args));
  if (!response.ok()) {
    ABSL_LOG(ERROR) << "Failed to send message: " << response.status();
    return nullptr;
  }
  auto* c_response = new LiteRtLmJsonResponse;
  c_response->json_string = response->dump();
  return c_response;
}

void litert_lm_json_response_delete(LiteRtLmJsonResponse* response) {
  delete response;
}

const char* litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response) {
  if (!response) {
    return nullptr;
  }
  return response->json_string.c_str();
}

int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmStreamCallback callback, void* callback_data) {
  if (!conversation || !conversation->conversation) {
    return -1;
  }
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return -1;
  }

  litert::lm::OptionalArgs litert_lm_optional_args = CreateOptionalArgs(
      conversation->conversation.get(), extra_context,
      optional_args ? optional_args->visual_token_budget : std::nullopt,
      optional_args ? optional_args->max_output_tokens : std::nullopt);

  absl::Status status = conversation->conversation->SendMessageAsync(
      json_message, CreateConversationCallback(callback, callback_data),
      std::move(litert_lm_optional_args));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start message stream: " << status;
    return static_cast<int>(status.code());
  }
  return 0;
}

const char* litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json) {
  if (!conversation || !conversation->conversation || !message_json) {
    return nullptr;
  }
  nlohmann::json json_message =
      nlohmann::json::parse(message_json, /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (json_message.is_discarded()) {
    ABSL_LOG(ERROR) << "Failed to parse message JSON.";
    return nullptr;
  }

  auto rendered = conversation->conversation->RenderMessageIntoString(
      json_message, litert::lm::OptionalArgs());
  if (!rendered.ok()) {
    ABSL_LOG(ERROR) << "Failed to render message: " << rendered.status();
    return nullptr;
  }
  conversation->last_rendered_message = std::move(*rendered);
  return conversation->last_rendered_message.c_str();
}

const char* litert_lm_conversation_render_preface_to_string(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  auto rendered = conversation->conversation->RenderPrefaceIntoString(
      litert::lm::OptionalArgs());
  if (!rendered.ok()) {
    ABSL_LOG(ERROR) << "Failed to render preface: " << rendered.status();
    return nullptr;
  }
  conversation->last_rendered_preface = std::move(*rendered);
  return conversation->last_rendered_preface.c_str();
}

void litert_lm_conversation_cancel_process(LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return;
  }
  conversation->conversation->CancelProcess();
}

LiteRtLmBenchmarkInfo* litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return nullptr;
  }
  auto benchmark_info = conversation->conversation->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get benchmark info: "
                    << benchmark_info.status();
    return nullptr;
  }
  return new LiteRtLmBenchmarkInfo{std::move(*benchmark_info)};
}

int litert_lm_conversation_get_token_count(LiteRtLmConversation* conversation) {
  if (!conversation || !conversation->conversation) {
    return -1;
  }
  absl::StatusOr<int> token_count = conversation->conversation->GetTokenCount();
  if (!token_count.ok()) {
    ABSL_LOG(ERROR) << "Failed to get token count: " << token_count.status();
    return -1;
  }
  return *token_count;
}

LiteRtLmTokenizeResult* litert_lm_engine_tokenize(LiteRtLmEngine* engine,
                                                  const char* text) {
  if (!engine || !engine->engine || !text) {
    return nullptr;
  }
  const auto& tokenizer = engine->engine->GetTokenizer();
  auto token_ids =
      const_cast<litert::lm::Tokenizer&>(tokenizer).TextToTokenIds(text);
  if (!token_ids.ok()) {
    ABSL_LOG(ERROR) << "Failed to tokenize: " << token_ids.status();
    return nullptr;
  }
  return new LiteRtLmTokenizeResult{std::move(*token_ids)};
}

void litert_lm_tokenize_result_delete(LiteRtLmTokenizeResult* result) {
  delete result;
}

const int* litert_lm_tokenize_result_get_tokens(
    const LiteRtLmTokenizeResult* result) {
  if (!result) {
    return nullptr;
  }
  return result->tokens.data();
}

size_t litert_lm_tokenize_result_get_num_tokens(
    const LiteRtLmTokenizeResult* result) {
  if (!result) {
    return 0;
  }
  return result->tokens.size();
}

LiteRtLmDetokenizeResult* litert_lm_engine_detokenize(LiteRtLmEngine* engine,
                                                      const int* tokens,
                                                      size_t num_tokens) {
  if (!engine || !engine->engine || !tokens) {
    return nullptr;
  }
  const auto& tokenizer = engine->engine->GetTokenizer();
  std::vector<int> token_ids(tokens, tokens + num_tokens);
  auto text =
      const_cast<litert::lm::Tokenizer&>(tokenizer).TokenIdsToText(token_ids);
  if (!text.ok()) {
    ABSL_LOG(ERROR) << "Failed to detokenize: " << text.status();
    return nullptr;
  }
  return new LiteRtLmDetokenizeResult{std::move(*text)};
}

void litert_lm_detokenize_result_delete(LiteRtLmDetokenizeResult* result) {
  delete result;
}

const char* litert_lm_detokenize_result_get_string(
    const LiteRtLmDetokenizeResult* result) {
  if (!result) {
    return nullptr;
  }
  return result->text.c_str();
}

void litert_lm_token_union_delete(LiteRtLmTokenUnion* token_union) {
  delete token_union;
}

LiteRtLmTokenUnionType litert_lm_token_union_get_type(
    const LiteRtLmTokenUnion* token_union) {
  if (token_union && token_union->token_union.has_token_str()) {
    return kLiteRtLmTokenUnionTypeString;
  }
  return kLiteRtLmTokenUnionTypeIds;
}

const char* litert_lm_token_union_get_string(
    const LiteRtLmTokenUnion* token_union) {
  if (token_union && token_union->token_union.has_token_str()) {
    return token_union->token_union.token_str().c_str();
  }
  return nullptr;
}

int litert_lm_token_union_get_ids(const LiteRtLmTokenUnion* token_union,
                                  const int** out_tokens,
                                  size_t* out_num_tokens) {
  if (!token_union || !token_union->token_union.has_token_ids() ||
      !out_tokens || !out_num_tokens) {
    return -1;
  }
  *out_tokens = token_union->token_union.token_ids().ids().data();
  *out_num_tokens = token_union->token_union.token_ids().ids_size();
  return 0;
}

void litert_lm_token_unions_delete(LiteRtLmTokenUnions* tokens) {
  delete tokens;
}

size_t litert_lm_token_unions_get_num_tokens(
    const LiteRtLmTokenUnions* tokens) {
  if (!tokens) {
    return 0;
  }
  return tokens->tokens.size();
}

LiteRtLmTokenUnion* litert_lm_token_unions_get_token_at(
    const LiteRtLmTokenUnions* tokens, size_t index) {
  if (!tokens || index >= tokens->tokens.size()) {
    return nullptr;
  }
  auto* result = new LiteRtLmTokenUnion();
  result->token_union = tokens->tokens[index];
  return result;
}

LiteRtLmTokenUnion* litert_lm_engine_get_start_token(LiteRtLmEngine* engine) {
  if (!engine || !engine->engine) {
    return nullptr;
  }
  const auto& metadata = engine->engine->GetEngineSettings().GetLlmMetadata();
  if (!metadata.has_value() || !metadata->has_start_token()) {
    return nullptr;
  }
  return new LiteRtLmTokenUnion{metadata->start_token()};
}

LiteRtLmTokenUnions* litert_lm_engine_get_stop_tokens(LiteRtLmEngine* engine) {
  if (!engine || !engine->engine) {
    return nullptr;
  }
  const auto& metadata = engine->engine->GetEngineSettings().GetLlmMetadata();
  if (!metadata.has_value() || metadata->stop_tokens_size() == 0) {
    return nullptr;
  }
  auto* c_tokens = new LiteRtLmTokenUnions;
  c_tokens->tokens.assign(metadata->stop_tokens().begin(),
                          metadata->stop_tokens().end());
  return c_tokens;
}

}  // extern "C"
