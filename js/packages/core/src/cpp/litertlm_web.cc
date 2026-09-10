/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Without this ifdef, presubmits will fail when compiling this for non-web
// platforms.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>

#include "research/drishti/app/pursuit/wasm/wasm_logging.h"
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "litert/js/packages/core/src/cpp/global_error_reporter.h"  // from @litert
#include "js/packages/core/src/cpp/readable_stream_data_stream.h"
#include "js/packages/core/src/cpp/unwrap_statusor.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/core/embedding_engine_impl.h"
#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/streamed_weights_manager.h"

using emscripten::optional_override;
using emscripten::val;

namespace litertlm_web {

// Stored callback for streaming weights.
static std::optional<emscripten::val> stream_weights_callback;

void RegisterStreamWeightsCallback(emscripten::val callback) {
  if (callback.isNull() || callback.isUndefined()) {
    stream_weights_callback = std::nullopt;
  } else {
    stream_weights_callback = callback;
  }
}

/**
 * Get the callback that, when called, will stream weights to WebGPU.
 *
 * This is used by litert/weight_loader/external_weights_loader.cc
 */
emscripten::val GetStreamWeightsCallback() {
  // TODO: b/555784020 - Use a better approach to identifying which model is
  // being loaded.
  if (!stream_weights_callback.has_value()) {
    return emscripten::val::undefined();
  }
  return stream_weights_callback.value();
}

void SetupLogging() { drishti::wasm::InitializeLog(); }

/**
 * Fill the prefill_batch_sizes field in AdvancedSettings from the given JS
 * array.
 */
void setPrefills(litert::lm::AdvancedSettings& advanced_settings,
                 val prefill_batch_sizes) {
  // embind does not have a builtin binding for std::set, so we use a manual
  // conversion to JS array and back to std::set.
  val js_array = val::global("Array").call<val>("from", prefill_batch_sizes);
  std::vector<int> prefill_batch_sizes_vec =
      emscripten::vecFromJSArray<int>(js_array);
  advanced_settings.prefill_batch_sizes.clear();
  for (int size : prefill_batch_sizes_vec) {
    advanced_settings.prefill_batch_sizes.insert(size);
  }
}

/**
 * Get the prefill_batch_sizes field in AdvancedSettings as a JS array.
 */
val getPrefills(const litert::lm::AdvancedSettings& advanced_settings) {
  val jsArray = val::global("Array").new_();
  for (int prefill_batch_size : advanced_settings.prefill_batch_sizes) {
    jsArray.call<void>("push", prefill_batch_size);
  }
  return jsArray;
}

struct JsBenchmarkInfo {
  double lastPrefillTokensPerSecond = 0.0;
  int lastPrefillTokenCount = 0;
  double lastDecodeTokensPerSecond = 0.0;
  int lastDecodeTokenCount = 0;
  double timeToFirstTokenInSecond = 0.0;
};

static litert::lm::EmbeddingOptions ParseEmbeddingOptions(
    emscripten::val options_val) {
  litert::lm::EmbeddingOptions options;
  if (!options_val.isUndefined() && !options_val.isNull()) {
    auto normalize_val = options_val["normalize"];
    if (!normalize_val.isUndefined() && !normalize_val.isNull()) {
      options.normalize = normalize_val.as<bool>();
    }
    auto insert_special_tokens_val = options_val["insertSpecialTokens"];
    if (!insert_special_tokens_val.isUndefined() &&
        !insert_special_tokens_val.isNull()) {
      options.insert_special_tokens = insert_special_tokens_val.as<bool>();
    }
    auto overflow_val = options_val["inputOverflowStrategy"];
    if (!overflow_val.isUndefined() && !overflow_val.isNull()) {
      options.input_overflow_strategy =
          overflow_val.as<litert::lm::InputOverflowStrategy>();
    }
    auto vision_tokens_val = options_val["visionTokensPerImage"];
    if (!vision_tokens_val.isUndefined() && !vision_tokens_val.isNull()) {
      options.vision_tokens_per_image = vision_tokens_val.as<int>();
    }
    auto output_size_val = options_val["outputSize"];
    if (!output_size_val.isUndefined() && !output_size_val.isNull()) {
      options.output_size = output_size_val.as<int>();
    }
  }
  return options;
}

static std::string ReadBytesFromVal(emscripten::val data_val) {
  if (data_val.isUndefined() || data_val.isNull()) {
    return "";
  }
  if (data_val.instanceof(emscripten::val::global("Uint8Array"))) {
    size_t length = data_val["length"].as<size_t>();
    std::string result(length, '\0');
    emscripten::val memory_view = emscripten::val(emscripten::typed_memory_view(
        length, reinterpret_cast<uint8_t*>(result.data())));
    memory_view.call<void>("set", data_val);
    return result;
  }
  if (data_val.instanceof(emscripten::val::global("ArrayBuffer"))) {
    emscripten::val uint8_array =
        emscripten::val::global("Uint8Array").new_(data_val);
    size_t length = uint8_array["length"].as<size_t>();
    std::string result(length, '\0');
    emscripten::val memory_view = emscripten::val(emscripten::typed_memory_view(
        length, reinterpret_cast<uint8_t*>(result.data())));
    memory_view.call<void>("set", uint8_array);
    return result;
  }
  return data_val.as<std::string>();
}

static absl::StatusOr<std::vector<litert::lm::InputData>> ParseInputData(
    emscripten::val input_val) {
  std::vector<litert::lm::InputData> contents;
  if (input_val.isUndefined() || input_val.isNull()) {
    return contents;
  }

  auto parse_single_item = [&](emscripten::val item) -> absl::Status {
    if (item.isString()) {
      contents.push_back(litert::lm::InputText(item.as<std::string>()));
      return absl::OkStatus();
    }
    if (item.typeOf().as<std::string>() == "object") {
      std::string type = "text";
      if (item.hasOwnProperty("type") && item["type"].isString()) {
        type = item["type"].as<std::string>();
      }
      if (type == "text") {
        std::string text = "";
        if (item.hasOwnProperty("text") && item["text"].isString()) {
          text = item["text"].as<std::string>();
        }
        contents.push_back(litert::lm::InputText(std::move(text)));
      } else if (type == "image" || type == "audio") {
        if (!item.hasOwnProperty("data") || item["data"].isNull() ||
            item["data"].isUndefined()) {
          return absl::InvalidArgumentError(
              absl::StrCat(type, " content part must have data."));
        }
        std::string bytes;
        emscripten::val data = item["data"];
        if (data.isString()) {
          if (!absl::Base64Unescape(data.as<std::string>(), &bytes)) {
            return absl::InvalidArgumentError(
                absl::StrCat("Failed to decode base64 data for ", type, "."));
          }
        } else {
          bytes = ReadBytesFromVal(data);
        }
        if (type == "image") {
          contents.push_back(litert::lm::InputImage(std::move(bytes)));
        } else {
          contents.push_back(litert::lm::InputAudio(std::move(bytes)));
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported content type: ", type));
      }
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError("Input item must be string or object.");
  };

  if (input_val.isArray()) {
    size_t length = input_val["length"].as<size_t>();
    for (size_t i = 0; i < length; ++i) {
      ABSL_RETURN_IF_ERROR(parse_single_item(input_val[i]));
    }
  } else {
    ABSL_RETURN_IF_ERROR(parse_single_item(input_val));
  }
  return contents;
}

EMSCRIPTEN_BINDINGS(litertlm_web) {
  emscripten::function("setupLogging", &SetupLogging);
  emscripten::function("setErrorReporter", &litert_web::SetErrorReporter);
  emscripten::function("registerStreamWeightsCallback",
                       &RegisterStreamWeightsCallback);
  emscripten::function("getStreamWeightsCallback", &GetStreamWeightsCallback);
  emscripten::function(
      "readStoredWeights",
      optional_override([](int model_type, double offset_double,
                           double size_double, uintptr_t dest_address) {
        uint64_t offset = static_cast<uint64_t>(offset_double);
        uint64_t size = static_cast<uint64_t>(size_double);
        void* buffer = reinterpret_cast<void*>(dest_address);
        UnwrapStatus(
            litert::lm::ReadStoredWeights(model_type, offset, size, buffer));
      }),
      emscripten::async());
  emscripten::function("clearStoredWeightsStreams", optional_override([]() {
                         UnwrapStatus(litert::lm::ClearStoredWeightsStreams());
                       }),
                       emscripten::async());
  emscripten::function(
      "getCurrentlyCompilingModel", optional_override([]() {
        return static_cast<int>(litert::lm::GetCurrentlyCompilingModel());
      }));

  emscripten::enum_<litert::lm::Backend>("Backend")
      .value("UNSPECIFIED", litert::lm::Backend::UNSPECIFIED)
      .value("CPU_ARTISAN", litert::lm::Backend::CPU_ARTISAN)
      .value("GPU_ARTISAN", litert::lm::Backend::GPU_ARTISAN)
      .value("CPU", litert::lm::Backend::CPU)
      .value("GPU", litert::lm::Backend::GPU)
      .value("GOOGLE_TENSOR_ARTISAN",
             litert::lm::Backend::GOOGLE_TENSOR_ARTISAN)
      .value("NPU", litert::lm::Backend::NPU);

  emscripten::class_<litert::lm::ModelAssets>("ModelAssets")
      .class_function(
          "create", optional_override([](std::string model_path) {
            return UnwrapStatusOr(litert::lm::ModelAssets::Create(model_path));
          }),
          emscripten::return_value_policy::take_ownership())
      .class_function(
          "createStreaming",
          optional_override(
              [](std::shared_ptr<litert::lm::ReadableStreamDataStream>
                     data_stream) {
                return UnwrapStatusOr(
                    litert::lm::ModelAssets::Create(data_stream));
              }),
          emscripten::return_value_policy::take_ownership())
      .function("getPath",
                optional_override([](litert::lm::ModelAssets& model_assets) {
                  // Convert from string_view to string for embind.
                  return std::string(UnwrapStatusOr(model_assets.GetPath()));
                }));

  emscripten::class_<litert::lm::EngineSettings>("EngineSettings")
      .class_function(
          "createDefault",
          optional_override([](litert::lm::ModelAssets model_assets,
                               litert::lm::Backend backend) {
            return UnwrapStatusOr(litert::lm::EngineSettings::CreateDefault(
                std::move(model_assets), backend));
          }),
          emscripten::return_value_policy::take_ownership())
      .function("getParallelFileSectionLoading",
                &litert::lm::EngineSettings::GetParallelFileSectionLoading)
      .function("setParallelFileSectionLoading",
                &litert::lm::EngineSettings::SetParallelFileSectionLoading)
      .function("getSingleThreadedExecution",
                &litert::lm::EngineSettings::GetSingleThreadedExecution)
      .function("setSingleThreadedExecution",
                &litert::lm::EngineSettings::SetSingleThreadedExecution)
      .function("enableBenchmark",
                optional_override([](litert::lm::EngineSettings& settings) {
                  settings.GetMutableBenchmarkParams();
                }))
      .function("getMutableMainExecutorSettings",
                &litert::lm::EngineSettings::GetMutableMainExecutorSettings,
                emscripten::return_value_policy::reference());

  emscripten::value_object<litert::lm::CpuConfig>("CpuConfig")
      .field("kv_increment_size", &litert::lm::CpuConfig::kv_increment_size)
      .field("prefill_chunk_size", &litert::lm::CpuConfig::prefill_chunk_size)
      .field("number_of_threads", &litert::lm::CpuConfig::number_of_threads);
  emscripten::value_object<litert::lm::GpuConfig>("GpuConfig")
      .field("max_top_k", &litert::lm::GpuConfig::max_top_k)
      .field("external_tensor_mode",
             &litert::lm::GpuConfig::external_tensor_mode);

  emscripten::register_vector<uint32_t>("VectorUint32");
  emscripten::value_object<litert::lm::GpuArtisanConfig>("GpuArtisanConfig")
      .field("num_output_candidates",
             &litert::lm::GpuArtisanConfig::num_output_candidates)
      .field("wait_for_weight_uploads",
             &litert::lm::GpuArtisanConfig::wait_for_weight_uploads)
      .field("num_decode_steps_per_sync",
             &litert::lm::GpuArtisanConfig::num_decode_steps_per_sync)
      .field("sequence_batch_size",
             &litert::lm::GpuArtisanConfig::sequence_batch_size)
      .field("supported_lora_ranks",
             &litert::lm::GpuArtisanConfig::supported_lora_ranks)
      .field("max_top_k", &litert::lm::GpuArtisanConfig::max_top_k)
      .field("enable_decode_logits",
             &litert::lm::GpuArtisanConfig::enable_decode_logits)
      .field("enable_external_embeddings",
             &litert::lm::GpuArtisanConfig::enable_external_embeddings)
      .field("use_submodel", &litert::lm::GpuArtisanConfig::use_submodel)
      .field("use_autosized_ringbuffers",
             &litert::lm::GpuArtisanConfig::use_autosized_ringbuffers);

  emscripten::value_object<litert::lm::AdvancedSettings>("AdvancedSettings")
      .field("prefill_batch_sizes", &getPrefills, &setPrefills)
      .field("num_output_candidates",
             &litert::lm::AdvancedSettings::num_output_candidates)
      .field("configure_magic_numbers",
             &litert::lm::AdvancedSettings::configure_magic_numbers)
      .field("verify_magic_numbers",
             &litert::lm::AdvancedSettings::verify_magic_numbers)
      .field("clear_kv_cache_before_prefill",
             &litert::lm::AdvancedSettings::clear_kv_cache_before_prefill)
      .field("num_logits_to_print_after_decode",
             &litert::lm::AdvancedSettings::num_logits_to_print_after_decode)
      .field("gpu_madvise_original_shared_tensors",
             &litert::lm::AdvancedSettings::gpu_madvise_original_shared_tensors)
      .field("is_benchmark", &litert::lm::AdvancedSettings::is_benchmark)
      .field("preferred_device_substr",
             &litert::lm::AdvancedSettings::preferred_device_substr)
      .field("num_threads_to_upload",
             &litert::lm::AdvancedSettings::num_threads_to_upload)
      .field("num_threads_to_compile",
             &litert::lm::AdvancedSettings::num_threads_to_compile)
      .field("convert_weights_on_gpu",
             &litert::lm::AdvancedSettings::convert_weights_on_gpu)
      .field("optimize_shader_compilation",
             &litert::lm::AdvancedSettings::optimize_shader_compilation)
      .field("share_constant_tensors",
             &litert::lm::AdvancedSettings::share_constant_tensors);

  emscripten::register_optional<litert::lm::AdvancedSettings>();

  emscripten::class_<litert::lm::ExecutorSettingsBase>("ExecutorSettingsBase")
      .function("getCacheDir", &litert::lm::ExecutorSettingsBase::GetCacheDir)
      .function("setCacheDir", &litert::lm::ExecutorSettingsBase::SetCacheDir)
      .function("getBackend", &litert::lm::ExecutorSettingsBase::GetBackend);

  emscripten::class_<litert::lm::LlmExecutorSettings,
                     emscripten::base<litert::lm::ExecutorSettingsBase>>(
      "LlmExecutorSettings")
      .function("getMaxNumTokens",
                &litert::lm::LlmExecutorSettings::GetMaxNumTokens)
      .function("setMaxNumTokens",
                &litert::lm::LlmExecutorSettings::SetMaxNumTokens)
      .function("getBackendConfigCpu",
                optional_override(
                    [](litert::lm::LlmExecutorSettings& executor_settings) {
                      return UnwrapStatusOr(
                          executor_settings
                              .GetBackendConfig<litert::lm::CpuConfig>());
                    }))
      .function("setBackendConfigCpu",
                optional_override(
                    [](litert::lm::LlmExecutorSettings& executor_settings,
                       litert::lm::CpuConfig cpu_config) {
                      executor_settings.SetBackendConfig(std::move(cpu_config));
                    }))
      .function("getBackendConfigGpu",
                optional_override(
                    [](litert::lm::LlmExecutorSettings& executor_settings) {
                      return UnwrapStatusOr(
                          executor_settings
                              .GetBackendConfig<litert::lm::GpuConfig>());
                    }))
      .function("setBackendConfigGpu",
                optional_override(
                    [](litert::lm::LlmExecutorSettings& executor_settings,
                       litert::lm::GpuConfig gpu_config) {
                      executor_settings.SetBackendConfig(std::move(gpu_config));
                    }))
      .function(
          "getBackendConfigGpuArtisan",
          optional_override(
              [](litert::lm::LlmExecutorSettings& executor_settings) {
                return UnwrapStatusOr(
                    executor_settings
                        .GetBackendConfig<litert::lm::GpuArtisanConfig>());
              }))
      .function("setBackendConfigGpuArtisan",
                optional_override(
                    [](litert::lm::LlmExecutorSettings& executor_settings,
                       litert::lm::GpuArtisanConfig gpu_artisan_config) {
                      executor_settings.SetBackendConfig(
                          std::move(gpu_artisan_config));
                    }))
      .function("setSamplerBackend",
                &litert::lm::LlmExecutorSettings::SetSamplerBackend)
      .function("getSamplerBackend",
                &litert::lm::LlmExecutorSettings::GetSamplerBackend)
      .function("setAdvancedSettings",
                &litert::lm::LlmExecutorSettings::SetAdvancedSettings)
      .function("getAdvancedSettings",
                &litert::lm::LlmExecutorSettings::GetAdvancedSettings);

  emscripten::class_<litert::lm::Engine>("Engine")
      .class_function(
          "createEngine",
          optional_override([](litert::lm::EngineSettings& engine_settings,
                               std::string input_prompt_as_hint = "") {
            return UnwrapStatusOr(litert::lm::EngineFactory::Create(
                litert::lm::EngineFactory::EngineType::
                    kAdvancedLiteRTCompiledModel,
                engine_settings, input_prompt_as_hint));
          }),
          emscripten::return_value_policy::take_ownership(),
          emscripten::async())
      .class_function(
          "createStreaming",
          optional_override([](litert::lm::EngineSettings& engine_settings,
                               std::string input_prompt_as_hint = "") {
            return UnwrapStatusOr(litert::lm::EngineFactory::Create(
                litert::lm::EngineFactory::EngineType::kAdvancedLegacyTfLite,
                engine_settings, input_prompt_as_hint));
          }),
          emscripten::return_value_policy::take_ownership(),
          emscripten::async())
      // We intentionally copy this value since embind does not handle const
      // references well.
      .function("getEngineSettings", &litert::lm::Engine::GetEngineSettings)
      .function(
          "waitUntilDone", optional_override([](litert::lm::Engine& engine) {
            UnwrapStatus(
                engine.WaitUntilDone(litert::lm::Engine::kDefaultTimeout));
          }),
          emscripten::async())
      .function(
          "createSession",
          optional_override([](litert::lm::Engine& engine,
                               litert::lm::SessionConfig& session_config) {
            return UnwrapStatusOr(engine.CreateSession(session_config));
          }),
          emscripten::return_value_policy::take_ownership());  // returns
                                                               // unique ptr

  emscripten::enum_<litert::lm::proto::SamplerParameters::Type>("SamplerType")
      .value("TYPE_UNSPECIFIED",
             litert::lm::proto::SamplerParameters::TYPE_UNSPECIFIED)
      .value("TOP_K", litert::lm::proto::SamplerParameters::TOP_K)
      .value("TOP_P", litert::lm::proto::SamplerParameters::TOP_P)
      .value("GREEDY", litert::lm::proto::SamplerParameters::GREEDY);

  emscripten::class_<litert::lm::proto::SamplerParameters>("SamplerParameters")
      .function("type", &litert::lm::proto::SamplerParameters::type)
      .function("setType", &litert::lm::proto::SamplerParameters::set_type)
      .function("k", &litert::lm::proto::SamplerParameters::k)
      .function("setK", &litert::lm::proto::SamplerParameters::set_k)
      .function("p", &litert::lm::proto::SamplerParameters::p)
      .function("setP", &litert::lm::proto::SamplerParameters::set_p)
      .function("temperature",
                &litert::lm::proto::SamplerParameters::temperature)
      .function("setTemperature",
                &litert::lm::proto::SamplerParameters::set_temperature)
      .function("seed", &litert::lm::proto::SamplerParameters::seed)
      .function("setSeed", &litert::lm::proto::SamplerParameters::set_seed);

  emscripten::register_vector<int>("VectorInt");
  emscripten::register_vector<std::vector<int>>("VectorVectorInt");
  emscripten::class_<litert::lm::SessionConfig>("SessionConfig")
      .class_function("createDefault",
                      &litert::lm::SessionConfig::CreateDefault,
                      emscripten::return_value_policy::take_ownership())
      .function("getAudioModalityEnabled",
                &litert::lm::SessionConfig::AudioModalityEnabled)
      .function("setAudioModalityEnabled",
                &litert::lm::SessionConfig::SetAudioModalityEnabled)
      .function("getVisionModalityEnabled",
                &litert::lm::SessionConfig::VisionModalityEnabled)
      .function("setVisionModalityEnabled",
                &litert::lm::SessionConfig::SetVisionModalityEnabled)
      .function("getMutableSamplerParams",
                &litert::lm::SessionConfig::GetMutableSamplerParams,
                emscripten::return_value_policy::reference())
      .function("getStopTokenIds", &litert::lm::SessionConfig::GetStopTokenIds)
      // Skipping getMutableStopTokenIds for now.
      .function("getStartTokenId", &litert::lm::SessionConfig::GetStartTokenId)
      .function("setStartTokenId", &litert::lm::SessionConfig::SetStartTokenId)
      .function("getNumOutputCandidates",
                &litert::lm::SessionConfig::GetNumOutputCandidates)
      .function("setNumOutputCandidates",
                &litert::lm::SessionConfig::SetNumOutputCandidates)
      .function("getSamplerBackend",
                &litert::lm::SessionConfig::GetSamplerBackend)
      .function("setSamplerBackend",
                &litert::lm::SessionConfig::SetSamplerBackend)
      // Skipping PromptTemplates for now.
      // Skipping LlmModelType for now.
      .function("getApplyPromptTemplateInSession",
                &litert::lm::SessionConfig::GetApplyPromptTemplateInSession)
      .function("setApplyPromptTemplateInSession",
                &litert::lm::SessionConfig::SetApplyPromptTemplateInSession)
      .function("getUseExternalSampler",
                &litert::lm::SessionConfig::UseExternalSampler)
      .function("setUseExternalSampler",
                &litert::lm::SessionConfig::SetUseExternalSampler)
      // Skipping ScopedLoraFile for now.
      .function("getMaxOutputTokens",
                &litert::lm::SessionConfig::GetMaxOutputTokens)
      .function("setMaxOutputTokens",
                &litert::lm::SessionConfig::SetMaxOutputTokens);

  emscripten::class_<litert::lm::Engine::Session>("Session")
      // We intentionally copy this value since embind does not handle const
      // references well.
      .function("getSessionConfig",
                &litert::lm::Engine::Session::GetSessionConfig)
      .function(
          "runPrefill",
          optional_override([](litert::lm::Engine::Session& session,
                               val inputs_array) {
            // Embind does not support std::variant, so we use a JS array
            // instead of a std::vector<std::variant<...>> and parse the
            // inputs manually.
            if (!val::global("Array").call<bool>("isArray", inputs_array)) {
              litert_web::GetGlobalErrorReporter()->ReportAndThrowError(
                  "inputs_array must be an Array");
            }
            int size = inputs_array["length"].as<int>();
            std::vector<litert::lm::InputData> inputs;
            inputs.reserve(size);
            for (int i = 0; i < size; ++i) {
              if (!inputs_array[i].isString()) {
                litert_web::GetGlobalErrorReporter()->ReportAndThrowError(
                    absl::StrFormat("inputs_array[%d] must be a string", i));
              }
              std::string input = inputs_array[i].as<std::string>();
              auto input_text = litert::lm::InputText(std::move(input));
              inputs.push_back(std::move(input_text));
            }
            return UnwrapStatus(session.RunPrefill(inputs));
          }),
          emscripten::async())
      .function("runDecode",
                optional_override([](litert::lm::Engine::Session& session) {
                  return UnwrapStatusOr(session.RunDecode());
                }),
                emscripten::async())
      .function("cancelProcess", &litert::lm::Engine::Session::CancelProcess)
      .function("clone",
                optional_override([](litert::lm::Engine::Session& session) {
                  return UnwrapStatusOr(session.Clone());
                }),
                emscripten::return_value_policy::take_ownership());

  emscripten::register_vector<std::string>("VectorString");

  emscripten::class_<litert::lm::Responses>("Responses")
      .function("getTexts", &litert::lm::Responses::GetTexts);

  emscripten::value_object<JsBenchmarkInfo>("BenchmarkInfo")
      .field("lastPrefillTokensPerSecond",
             &JsBenchmarkInfo::lastPrefillTokensPerSecond)
      .field("lastPrefillTokenCount", &JsBenchmarkInfo::lastPrefillTokenCount)
      .field("lastDecodeTokensPerSecond",
             &JsBenchmarkInfo::lastDecodeTokensPerSecond)
      .field("lastDecodeTokenCount", &JsBenchmarkInfo::lastDecodeTokenCount)
      .field("timeToFirstTokenInSecond",
             &JsBenchmarkInfo::timeToFirstTokenInSecond);

  emscripten::class_<litert::lm::ConversationConfig>("ConversationConfig")
      .class_function(
          "createDefault",
          optional_override([](const litert::lm::Engine& engine) {
            return UnwrapStatusOr(
                litert::lm::ConversationConfig::CreateDefault(engine));
          }),
          emscripten::return_value_policy::take_ownership())
      .class_function(
          "createCustom",
          optional_override([](const litert::lm::Engine& engine,
                               const litert::lm::SessionConfig& session_config,
                               bool enable_constrained_decoding,
                               bool prefill_preface_on_init,
                               bool filter_channel_content_from_kv_cache,
                               std::string preface_json) {
            litert::lm::ConversationConfig::Builder builder;
            builder.SetSessionConfig(session_config);
            builder.SetEnableConstrainedDecoding(enable_constrained_decoding);
            builder.SetPrefillPrefaceOnInit(prefill_preface_on_init);
            builder.SetFilterChannelContentFromKvCache(
                filter_channel_content_from_kv_cache);

            if (!preface_json.empty()) {
              auto json = nlohmann::ordered_json::parse(preface_json);
              litert::lm::JsonPreface json_preface;
              if (json.contains("messages")) {
                json_preface.messages = json["messages"];
              }
              if (json.contains("tools")) {
                json_preface.tools = json["tools"];
              }
              if (json.contains("extra_context")) {
                json_preface.extra_context = json["extra_context"];
              }
              builder.SetPreface(litert::lm::Preface(json_preface));
            }
            return UnwrapStatusOr(builder.Build(engine));
          }),
          emscripten::return_value_policy::take_ownership());

  emscripten::class_<litert::lm::Conversation>("Conversation")
      .class_function(
          "create",
          optional_override([](litert::lm::Engine& engine,
                               const litert::lm::ConversationConfig& config) {
            return UnwrapStatusOr(
                litert::lm::Conversation::Create(engine, config));
          }),
          emscripten::return_value_policy::take_ownership(),
          emscripten::async())
      .function("sendMessage",
                optional_override([](litert::lm::Conversation& conversation,
                                     std::string message_json) {
                  auto message = nlohmann::ordered_json::parse(message_json);
                  auto result = conversation.SendMessage(message);
                  return UnwrapStatusOr(result).dump();
                }),
                emscripten::async())
      .function(
          "sendMessageAsync",
          optional_override([](litert::lm::Conversation& conversation,
                               std::string message_json,
                               emscripten::val callback) {
            auto message = nlohmann::ordered_json::parse(message_json);
            auto status = conversation.SendMessageAsync(
                message,
                // Note: This capture is not thread-safe, but we're only using
                // a single thread for now.
                [callback](absl::StatusOr<litert::lm::Message> result) mutable {
                  if (!result.ok()) {
                    callback(emscripten::val::null(), true,
                             std::string(result.status().message()));
                    return;
                  }
                  if (result->is_null()) {
                    callback(emscripten::val::null(), true,
                             emscripten::val::null());
                  } else {
                    callback(result->dump(), false, emscripten::val::null());
                  }
                });
            UnwrapStatus(status);
          }),
          emscripten::async())
      .function(
          "getHistory",
          optional_override([](const litert::lm::Conversation& conversation) {
            nlohmann::ordered_json json_history =
                nlohmann::ordered_json::array();
            conversation.AccessHistory([&json_history](const auto& history) {
              for (const auto& msg : history) {
                json_history.push_back(msg);
              }
            });
            return json_history.dump();
          }))
      .function(
          "getTokenCount",
          optional_override([](const litert::lm::Conversation& conversation) {
            return UnwrapStatusOr(conversation.GetTokenCount());
          }))
      .function("getBenchmarkInfo",
                optional_override([](litert::lm::Conversation& conversation) {
                  auto info_or = conversation.GetBenchmarkInfo();
                  JsBenchmarkInfo js_info;
                  if (!info_or.ok()) return js_info;

                  auto& info = *info_or;
                  int totalPrefillTurns = info.GetTotalPrefillTurns();
                  if (totalPrefillTurns > 0) {
                    js_info.lastPrefillTokensPerSecond =
                        info.GetPrefillTokensPerSec(totalPrefillTurns - 1);
                    auto prefill_turn =
                        info.GetPrefillTurn(totalPrefillTurns - 1);
                    if (prefill_turn.ok())
                      js_info.lastPrefillTokenCount = prefill_turn->num_tokens;
                  }
                  int totalDecodeTurns = info.GetTotalDecodeTurns();
                  if (totalDecodeTurns > 0) {
                    js_info.lastDecodeTokensPerSecond =
                        info.GetDecodeTokensPerSec(totalDecodeTurns - 1);
                    auto decode_turn = info.GetDecodeTurn(totalDecodeTurns - 1);
                    if (decode_turn.ok())
                      js_info.lastDecodeTokenCount = decode_turn->num_tokens;
                  }
                  js_info.timeToFirstTokenInSecond = info.GetTimeToFirstToken();
                  return js_info;
                }))
      .function("cancelProcess", &litert::lm::Conversation::CancelProcess)
      .function("clone",
                optional_override([](litert::lm::Conversation& conversation) {
                  return UnwrapStatusOr(conversation.Clone());
                }),
                emscripten::return_value_policy::take_ownership());

  emscripten::class_<litert::lm::DataStream>("DataStream");
  emscripten::class_<litert::lm::ReadableStreamDataStream,
                     emscripten::base<litert::lm::DataStream>>(
      "ReadableStreamDataStream")
      .smart_ptr<std::shared_ptr<litert::lm::ReadableStreamDataStream>>(
          "shared_ptr<ReadableStreamDataStream>")
      .class_function("create", &litert::lm::ReadableStreamDataStream::Create);

  emscripten::enum_<litert::lm::InputOverflowStrategy>("InputOverflowStrategy")
      .value("ERROR", litert::lm::InputOverflowStrategy::kError)
      .value("TRUNCATE", litert::lm::InputOverflowStrategy::kTruncate)
      .value("CHUNK_AND_AVERAGE",
             litert::lm::InputOverflowStrategy::kChunkAndAverage);

  emscripten::register_optional<int>();
  emscripten::register_vector<float>("VectorFloat");

  emscripten::value_object<litert::lm::EmbeddingOptions>("EmbeddingOptions")
      .field("normalize", &litert::lm::EmbeddingOptions::normalize)
      .field("insertSpecialTokens",
             &litert::lm::EmbeddingOptions::insert_special_tokens)
      .field("inputOverflowStrategy",
             &litert::lm::EmbeddingOptions::input_overflow_strategy)
      .field("visionTokensPerImage",
             &litert::lm::EmbeddingOptions::vision_tokens_per_image)
      .field("outputSize", &litert::lm::EmbeddingOptions::output_size);

  emscripten::value_object<litert::lm::EmbeddingResponse>("EmbeddingResponse")
      .field("embedding", &litert::lm::EmbeddingResponse::embedding)
      .field("inputLength", &litert::lm::EmbeddingResponse::input_length)
      .field("truncatedLength",
             &litert::lm::EmbeddingResponse::truncated_length)
      .field("numChunks", &litert::lm::EmbeddingResponse::num_chunks);

  emscripten::register_vector<litert::lm::EmbeddingResponse>(
      "VectorEmbeddingResponse");

  emscripten::class_<litert::lm::EmbeddingExecutorSettings,
                     emscripten::base<litert::lm::ExecutorSettingsBase>>(
      "EmbeddingExecutorSettings")
      .function("getNumThreads",
                &litert::lm::EmbeddingExecutorSettings::GetNumThreads)
      .function("setNumThreads",
                &litert::lm::EmbeddingExecutorSettings::SetNumThreads);

  emscripten::class_<litert::lm::EmbeddingEngineSettings>(
      "EmbeddingEngineSettings")
      .class_function(
          "createDefault",
          optional_override([](litert::lm::ModelAssets model_assets,
                               litert::lm::Backend backend) {
            auto settings = UnwrapStatusOr(
                litert::lm::EmbeddingEngineSettings::CreateDefault(
                    std::move(model_assets), backend));
            settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
            return settings;
          }),
          emscripten::return_value_policy::take_ownership())
      .class_function(
          "createDefaultMultimodal",
          optional_override([](litert::lm::ModelAssets model_assets,
                               litert::lm::Backend backend,
                               emscripten::val vision_backend_val,
                               emscripten::val audio_backend_val) {
            std::optional<litert::lm::Backend> vision_backend;
            if (!vision_backend_val.isUndefined() &&
                !vision_backend_val.isNull()) {
              int val_int = 0;
              if (vision_backend_val.hasOwnProperty("value")) {
                val_int = vision_backend_val["value"].as<int>();
              } else {
                val_int = vision_backend_val.as<int>();
              }
              vision_backend = static_cast<litert::lm::Backend>(val_int);
            }
            std::optional<litert::lm::Backend> audio_backend;
            if (!audio_backend_val.isUndefined() &&
                !audio_backend_val.isNull()) {
              int val_int = 0;
              if (audio_backend_val.hasOwnProperty("value")) {
                val_int = audio_backend_val["value"].as<int>();
              } else {
                val_int = audio_backend_val.as<int>();
              }
              audio_backend = static_cast<litert::lm::Backend>(val_int);
            }
            auto settings = UnwrapStatusOr(
                litert::lm::EmbeddingEngineSettings::CreateDefault(
                    std::move(model_assets), backend, vision_backend,
                    audio_backend));
            settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
            if (settings.GetVisionExecutorSettings().has_value()) {
              settings.GetMutableVisionExecutorSettings()->SetCacheDir(
                  ":nocache");
            }
            if (settings.GetAudioExecutorSettings().has_value()) {
              settings.GetMutableAudioExecutorSettings()->SetCacheDir(
                  ":nocache");
            }
            return settings;
          }),
          emscripten::return_value_policy::take_ownership())
      .function("getMaxInputLength",
                &litert::lm::EmbeddingEngineSettings::GetMaxInputLength)
      .function("setMaxInputLength",
                &litert::lm::EmbeddingEngineSettings::SetMaxInputLength)
      .function("getVisionTokensPerImage",
                &litert::lm::EmbeddingEngineSettings::GetVisionTokensPerImage)
      .function("setVisionTokensPerImage",
                &litert::lm::EmbeddingEngineSettings::SetVisionTokensPerImage)
      .function(
          "getMutableMainExecutorSettings",
          &litert::lm::EmbeddingEngineSettings::GetMutableMainExecutorSettings,
          emscripten::return_value_policy::reference());

  emscripten::class_<litert::lm::EmbeddingEngine>("EmbeddingEngine")
      .class_function(
          "createEngine",
          optional_override([](litert::lm::EmbeddingEngineSettings& settings) {
            return UnwrapStatusOr(
                litert::lm::EmbeddingEngineImpl::Create(std::move(settings)));
          }),
          emscripten::return_value_policy::take_ownership(),
          emscripten::async())
      .function(
          "computeEmbedding",
          optional_override([](litert::lm::EmbeddingEngine& engine,
                               emscripten::val input_val,
                               emscripten::val options_val) {
            auto options = ParseEmbeddingOptions(options_val);
            auto contents = UnwrapStatusOr(ParseInputData(input_val));
            return UnwrapStatusOr(engine.ComputeEmbedding(contents, options));
          }),
          emscripten::async())
      .function(
          "computeEmbeddingBatch",
          optional_override([](litert::lm::EmbeddingEngine& engine,
                               emscripten::val batch_val,
                               emscripten::val options_val) {
            auto options = ParseEmbeddingOptions(options_val);
            std::vector<std::vector<litert::lm::InputData>> batch_contents;
            if (!batch_val.isArray()) {
              auto contents = UnwrapStatusOr(ParseInputData(batch_val));
              batch_contents.push_back(std::move(contents));
            } else {
              size_t length = batch_val["length"].as<size_t>();
              batch_contents.reserve(length);
              for (size_t i = 0; i < length; ++i) {
                auto contents = UnwrapStatusOr(ParseInputData(batch_val[i]));
                batch_contents.push_back(std::move(contents));
              }
            }
            return UnwrapStatusOr(
                engine.ComputeEmbeddingBatch(batch_contents, options));
          }),
          emscripten::async());
}
}  // namespace litertlm_web
#endif  // __EMSCRIPTEN__
