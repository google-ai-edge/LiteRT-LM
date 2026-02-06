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
#include <cstring>
#include <memory>
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
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/mezo.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

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
    if (auto* json_msg = std::get_if<litert::lm::JsonMessage>(&*message)) {
      if (json_msg->is_null()) {  // End of stream marker
        callback(user_data, nullptr, true, nullptr);
      } else {
        std::string json_str = json_msg->dump();
        callback(user_data, const_cast<char*>(json_str.c_str()), false,
                 nullptr);
      }
    } else {
      std::string error_str = "Unsupported message type";
      callback(user_data, nullptr, true, const_cast<char*>(error_str.c_str()));
    }
  };
}

}  // namespace

using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::Engine;
using ::litert::lm::EngineFactory;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputText;
using ::litert::lm::JsonMessage;
using ::litert::lm::Message;
using ::litert::lm::MeZoConfig;
using ::litert::lm::MeZoFineTuner;
using ::litert::lm::ModelAssets;
using ::litert::lm::NamedParameter;
using ::litert::lm::Responses;
using ::litert::lm::SessionConfig;
using ::litert::lm::proto::SamplerParameters;

struct LiteRtLmEngineSettings {
  std::unique_ptr<EngineSettings> settings;
};

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
};

struct LiteRtLmJsonResponse {
  std::string json_string;
};

struct LiteRtLmSessionConfig {
  std::unique_ptr<SessionConfig> config;
};

struct LiteRtLmConversationConfig {
  std::unique_ptr<ConversationConfig> config;
};

struct LiteRtLmMeZoConfig {
  MeZoConfig config;
};

struct LiteRtLmMeZoFineTuner {
  std::unique_ptr<MeZoFineTuner> finetuner;
};

extern "C" {

SamplerParameters::Type ToSamplerParametersType(Type type) {
  switch (type) {
    case kTypeUnspecified:
      return SamplerParameters::TYPE_UNSPECIFIED;
    case kTopK:
      return SamplerParameters::TOP_K;
    case kTopP:
      return SamplerParameters::TOP_P;
    case kGreedy:
      return SamplerParameters::GREEDY;
  }
  return SamplerParameters::TYPE_UNSPECIFIED;
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

void litert_lm_session_config_set_lora_id(LiteRtLmSessionConfig* config,
                                          int32_t lora_id) {
  if (config && config->config) {
    if (lora_id >= 0) {
      config->config->SetLoraId(static_cast<uint32_t>(lora_id));
    } else {
      config->config->SetLoraId(std::nullopt);
    }
  }
}

int32_t litert_lm_session_config_get_lora_id(
    const LiteRtLmSessionConfig* config) {
  if (!config || !config->config) {
    return -1;
  }
  auto lora_id = config->config->GetLoraId();
  return lora_id.has_value() ? static_cast<int32_t>(*lora_id) : -1;
}

void litert_lm_session_config_delete(LiteRtLmSessionConfig* config) {
  delete config;
}

LiteRtLmConversationConfig* litert_lm_conversation_config_create(
    LiteRtLmEngine* engine, const LiteRtLmSessionConfig* session_config,
    const char* system_message_json) {
  if (!engine || !engine->engine) {
    return nullptr;
  }

  litert::lm::JsonPreface json_preface;
  if (system_message_json) {
    nlohmann::ordered_json system_message;
    system_message["role"] = "system";
    auto content =
        nlohmann::ordered_json::parse(system_message_json, nullptr, false);
    if (content.is_discarded()) {
      // If JSON parsing fails, assume it's a plain string.
      system_message["content"] = system_message_json;
    } else {
      system_message["content"] = content;
    }
    json_preface.messages = nlohmann::ordered_json::array({system_message});
  }

  std::unique_ptr<SessionConfig> default_session_config;
  const SessionConfig* config_to_use;

  if (session_config && session_config->config) {
    config_to_use = session_config->config.get();
  } else {
    default_session_config =
        std::make_unique<SessionConfig>(SessionConfig::CreateDefault());
    config_to_use = default_session_config.get();
  }

  auto conversation_config = litert::lm::ConversationConfig::Builder()
                                 .SetSessionConfig(*config_to_use)
                                 .SetPreface(json_preface)
                                 .Build(*engine->engine);

  if (!conversation_config.ok()) {
    ABSL_LOG(ERROR) << "Failed to create conversation config: "
                    << conversation_config.status();
    return nullptr;
  }

  auto* c_config = new LiteRtLmConversationConfig;
  c_config->config =
      std::make_unique<ConversationConfig>(*std::move(conversation_config));
  return c_config;
}

void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config) {
  delete config;
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
  auto backend = litert::lm::GetBackendFromString(backend_str);
  if (!backend.ok()) {
    ABSL_LOG(ERROR) << "Failed to parse backend: " << backend.status();
    return nullptr;
  }

  std::optional<litert::lm::Backend> vision_backend;
  if (vision_backend_str) {
    auto backend = litert::lm::GetBackendFromString(vision_backend_str);
    if (!backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse vision backend: " << backend.status();
      return nullptr;
    }
    vision_backend = *backend;
  }

  std::optional<litert::lm::Backend> audio_backend;
  if (audio_backend_str) {
    auto backend = litert::lm::GetBackendFromString(audio_backend_str);
    if (!backend.ok()) {
      ABSL_LOG(ERROR) << "Failed to parse audio backend: " << backend.status();
      return nullptr;
    }
    audio_backend = *backend;
  }

  auto engine_settings = EngineSettings::CreateDefault(
      *std::move(model_assets), *backend, vision_backend, audio_backend);
  if (!engine_settings.ok()) {
    ABSL_LOG(ERROR) << "Failed to create engine settings: "
                    << engine_settings.status();
    return nullptr;
  }

  auto* c_settings = new LiteRtLmEngineSettings;
  c_settings->settings =
      std::make_unique<EngineSettings>(*std::move(engine_settings));
  return c_settings;
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

void litert_lm_engine_settings_set_cache_dir(LiteRtLmEngineSettings* settings,
                                             const char* cache_dir) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir);
  }
}

void litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings) {
  if (settings && settings->settings) {
    settings->settings->GetMutableBenchmarkParams();
  }
}

void litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings, int activation_data_type_int) {
  if (settings && settings->settings) {
    settings->settings->GetMutableMainExecutorSettings().SetActivationDataType(
        static_cast<litert::lm::ActivationDataType>(activation_data_type_int));
  }
}

LiteRtLmEngine* litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings) {
  if (!settings || !settings->settings) {
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<Engine>> engine;
    engine = EngineFactory::CreateDefault(*settings->settings);


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
  absl::StatusOr<std::unique_ptr<Engine::Session>> session;
  if (config && config->config) {
    session = engine->engine->CreateSession(*config->config);
  } else {
    session = engine->engine->CreateSession(SessionConfig::CreateDefault());
  }
  if (!session.ok()) {
    ABSL_LOG(ERROR) << "Failed to create session: " << session.status();
    return nullptr;
  }

  auto* c_session = new LiteRtLmSession;
  c_session->session = *std::move(session);
  return c_session;
}

void litert_lm_session_delete(LiteRtLmSession* session) { delete session; }

LiteRtLmResponses* litert_lm_session_generate_content(LiteRtLmSession* session,
                                                      const InputData* inputs,
                                                      size_t num_inputs) {
  if (!session || !session->session) {
    return nullptr;
  }
  std::vector<std::variant<litert::lm::InputText, litert::lm::InputImage,
                           litert::lm::InputAudio, litert::lm::InputAudioEnd>>
      engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case kInputText:
        engine_inputs.emplace_back(InputText(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputImage:
        engine_inputs.emplace_back(litert::lm::InputImage(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputAudio:
        engine_inputs.emplace_back(litert::lm::InputAudio(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputAudioEnd:
        engine_inputs.emplace_back(litert::lm::InputAudioEnd());
        break;
    }
  }
  auto responses = session->session->GenerateContent(std::move(engine_inputs));
  if (!responses.ok()) {
    ABSL_LOG(ERROR) << "Failed to generate content: " << responses.status();
    return nullptr;
  }

  auto* c_responses = new LiteRtLmResponses{std::move(*responses)};
  return c_responses;
}

int litert_lm_session_generate_content_stream(LiteRtLmSession* session,
                                              const InputData* inputs,
                                              size_t num_inputs,
                                              LiteRtLmStreamCallback callback,
                                              void* callback_data) {
  if (!session || !session->session) {
    return -1;
  }
  std::vector<std::variant<litert::lm::InputText, litert::lm::InputImage,
                           litert::lm::InputAudio, litert::lm::InputAudioEnd>>
      engine_inputs;
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    switch (inputs[i].type) {
      case kInputText:
        engine_inputs.emplace_back(InputText(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputImage:
        engine_inputs.emplace_back(litert::lm::InputImage(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputAudio:
        engine_inputs.emplace_back(litert::lm::InputAudio(std::string(
            static_cast<const char*>(inputs[i].data), inputs[i].size)));
        break;
      case kInputAudioEnd:
        engine_inputs.emplace_back(litert::lm::InputAudioEnd());
        break;
    }
  }

  absl::Status status = session->session->GenerateContentStream(
      std::move(engine_inputs), CreateCallback(callback, callback_data));

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
  return responses->responses.GetTexts().size();
}

const char* litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index) {
  if (!responses) {
    return nullptr;
  }
  if (index < 0 || index >= responses->responses.GetTexts().size()) {
    return nullptr;
  }

  // The string_view's data is valid as long as the responses object is alive.
  return responses->responses.GetTexts()[index].data();
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
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* conversation_config) {
  if (!engine || !engine->engine) {
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<Conversation>> conversation;
  if (conversation_config && conversation_config->config) {
    conversation =
        Conversation::Create(*engine->engine, *conversation_config->config);
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

LiteRtLmJsonResponse* litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json) {
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
  auto response = conversation->conversation->SendMessage(json_message);
  if (!response.ok()) {
    ABSL_LOG(ERROR) << "Failed to send message: " << response.status();
    return nullptr;
  }
  auto* json_response = std::get_if<JsonMessage>(&*response);
  if (!json_response) {
    ABSL_LOG(ERROR) << "Response is not a JSON message.";
    return nullptr;
  }
  auto* c_response = new LiteRtLmJsonResponse;
  c_response->json_string = json_response->dump();
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

  absl::Status status = conversation->conversation->SendMessageAsync(
      json_message, CreateConversationCallback(callback, callback_data));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to start message stream: " << status;
    return static_cast<int>(status.code());
  }
  return 0;
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

// ---------------------------------------------------------------------------
// MeZO Fine-Tuning C API
// ---------------------------------------------------------------------------

LiteRtLmMeZoConfig* litert_lm_mezo_config_create() {
  return new LiteRtLmMeZoConfig;
}

void litert_lm_mezo_config_delete(LiteRtLmMeZoConfig* config) {
  delete config;
}

void litert_lm_mezo_config_set_learning_rate(LiteRtLmMeZoConfig* config,
                                             float learning_rate) {
  if (config) {
    config->config.SetLearningRate(learning_rate);
  }
}

void litert_lm_mezo_config_set_epsilon(LiteRtLmMeZoConfig* config,
                                       float epsilon) {
  if (config) {
    config->config.SetEpsilon(epsilon);
  }
}

void litert_lm_mezo_config_set_weight_decay(LiteRtLmMeZoConfig* config,
                                            float weight_decay) {
  if (config) {
    config->config.SetWeightDecay(weight_decay);
  }
}

void litert_lm_mezo_config_set_seed(LiteRtLmMeZoConfig* config,
                                    uint64_t seed) {
  if (config) {
    config->config.SetSeed(seed);
  }
}

LiteRtLmMeZoFineTuner* litert_lm_mezo_finetuner_create(
    const LiteRtLmMeZoConfig* config) {
  if (!config) {
    return nullptr;
  }
  auto finetuner = MeZoFineTuner::Create(config->config);
  if (!finetuner.ok()) {
    ABSL_LOG(ERROR) << "Failed to create MeZO fine-tuner: "
                    << finetuner.status();
    return nullptr;
  }
  auto* c_finetuner = new LiteRtLmMeZoFineTuner;
  c_finetuner->finetuner = *std::move(finetuner);
  return c_finetuner;
}

void litert_lm_mezo_finetuner_delete(LiteRtLmMeZoFineTuner* finetuner) {
  delete finetuner;
}

int litert_lm_mezo_finetuner_step(LiteRtLmMeZoFineTuner* finetuner,
                                  const LiteRtLmMeZoParameter* parameters,
                                  size_t num_parameters,
                                  LiteRtLmMeZoLossFn loss_fn, void* user_data,
                                  float* loss_out) {
  if (!finetuner || !finetuner->finetuner || !parameters || !loss_fn ||
      !loss_out) {
    return -1;
  }

  // Convert C parameters to C++ NamedParameter vector.
  std::vector<NamedParameter> cpp_params;
  cpp_params.reserve(num_parameters);
  for (size_t i = 0; i < num_parameters; ++i) {
    NamedParameter p;
    p.name = parameters[i].name ? parameters[i].name : "";
    p.data = parameters[i].data;
    p.num_elements = parameters[i].num_elements;
    p.is_bias_or_layernorm = !parameters[i].apply_weight_decay;
    cpp_params.push_back(std::move(p));
  }

  // Wrap the C loss callback as a C++ invocable.
  auto cpp_loss_fn =
      [loss_fn, user_data]() -> absl::StatusOr<float> {
    float loss = 0.0f;
    int result = loss_fn(user_data, &loss);
    if (result != 0) {
      return absl::InternalError("Loss function callback failed.");
    }
    return loss;
  };

  auto result =
      finetuner->finetuner->Step(cpp_params, std::move(cpp_loss_fn));
  if (!result.ok()) {
    ABSL_LOG(ERROR) << "MeZO step failed: " << result.status();
    return static_cast<int>(result.status().code());
  }
  *loss_out = *result;
  return 0;
}

uint64_t litert_lm_mezo_finetuner_get_step_count(
    const LiteRtLmMeZoFineTuner* finetuner) {
  if (!finetuner || !finetuner->finetuner) {
    return 0;
  }
  return finetuner->finetuner->GetStepCount();
}

void litert_lm_mezo_finetuner_set_learning_rate(
    LiteRtLmMeZoFineTuner* finetuner, float learning_rate) {
  if (finetuner && finetuner->finetuner) {
    finetuner->finetuner->SetLearningRate(learning_rate);
  }
}

float litert_lm_mezo_finetuner_get_learning_rate(
    const LiteRtLmMeZoFineTuner* finetuner) {
  if (!finetuner || !finetuner->finetuner) {
    return 0.0f;
  }
  return finetuner->finetuner->GetLearningRate();
}

}  // extern "C"
