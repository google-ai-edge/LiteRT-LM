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

#include "omni/tts/kokoro/kokoro_factory.h"

#include <algorithm>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/tts/kokoro/common.h"
#include "omni/tts/kokoro/espeak_assets.h"
#include "omni/tts/kokoro/kokoro_acoustic_stage.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/kokoro/kokoro_vocoder_stage.h"
#include "omni/tts/kokoro/phonemizer.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/tts_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::tts {

namespace {

ModelOptions MakeModelOptions(absl::string_view model_dir,
                              absl::string_view cache_dir, lm::Backend backend,
                              int num_threads) {
  ModelOptions options;
  options.model_dir = model_dir;
  options.cache_dir = cache_dir;
  options.backend = backend;
  options.num_threads = num_threads;
  return options;
}

}  // namespace

absl::Status InitKokoroResources(KokoroModelConfig& config,
                                 absl::string_view model_folder,
                                 absl::string_view cache_dir,
                                 lm::Backend backend, int num_threads,
                                 ::litert::Environment& env,
                                 ModelResources& resources) {
  // The acoustic model is slower on GPU than on CPU in practice, so a GPU
  // engine backend still runs it on CPU unless explicitly configured; only the
  // vocoder follows the engine backend.
  const lm::Backend acoustic_backend = config.acoustic_backend.value_or(
      backend == lm::Backend::GPU ? lm::Backend::CPU : backend);
  const lm::Backend vocoder_backend = config.vocoder_backend.value_or(backend);

  ModelOptions acoustic_options =
      MakeModelOptions(model_folder, cache_dir, acoustic_backend, num_threads);
  ModelOptions vocoder_options =
      MakeModelOptions(model_folder, cache_dir, vocoder_backend, num_threads);

  std::optional<CompiledModel> acoustic_opt;
  std::optional<CompiledModel> vocoder_opt;
  if (resources.HasLmModelResources()) {
    auto lm_resources = resources.GetLmModelResources();
    LITERT_ASSIGN_OR_RETURN(
        absl::string_view acoustic_buffer,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::TtsMetadata::TF_LITE_ACOUSTIC));
    LITERT_ASSIGN_OR_RETURN(
        auto acoustic_compiled,
        CreateCompiledModelFromBuffer(env, acoustic_options, acoustic_buffer,
                                      "kokoro_acoustic"));
    LITERT_ASSIGN_OR_RETURN(
        absl::string_view vocoder_buffer,
        lm_resources->GetTFLiteModelBuffer(
            lm::proto::TtsMetadata::TF_LITE_VOCODER));
    LITERT_ASSIGN_OR_RETURN(
        auto vocoder_compiled,
        CreateCompiledModelFromBuffer(env, vocoder_options, vocoder_buffer,
                                      "kokoro_vocoder"));
    acoustic_opt.emplace(std::move(acoustic_compiled));
    vocoder_opt.emplace(std::move(vocoder_compiled));

    // espeak-ng only reads its data from the filesystem, so data shipped in
    // the container has to be unpacked before the phonemizer starts. A
    // container without espeak-ng data is fine: the phonemizer then falls back
    // to data next to the model.
    if (config.espeak_data_dir.empty()) {
      absl::StatusOr<std::string> espeak_data_dir =
          kokoro::UnpackEspeakDataFromLitertLm(*lm_resources, cache_dir);
      if (espeak_data_dir.ok()) {
        config.espeak_data_dir = *std::move(espeak_data_dir);
      } else if (!absl::IsNotFound(espeak_data_dir.status())) {
        return espeak_data_dir.status();
      }
    }

    // Text normalization rule tables and CJK lexicon blobs ride along as data
    // so that language-specific tables stay out of the binary.
    for (const std::string& name : lm_resources->GetGenericBinaryDataNames()) {
      absl::string_view language(name);
      if (absl::ConsumeSuffix(&language, kTextNormSectionSuffix) &&
          !language.empty()) {
        ABSL_ASSIGN_OR_RETURN(absl::string_view rules,
                              lm_resources->GetGenericBinaryDataBuffer(name));
        config.text_norm_rules[NormalizeLanguageCode(language)] =
            std::string(rules);
        continue;
      }
      language = name;
      if (absl::ConsumeSuffix(&language, kLexiconSectionSuffix) &&
          !language.empty()) {
        ABSL_ASSIGN_OR_RETURN(absl::string_view blob,
                              lm_resources->GetGenericBinaryDataBuffer(name));
        config.cjk_lexicons[NormalizeLanguageCode(language)] = blob;
      }
    }
  } else {
    LITERT_ASSIGN_OR_RETURN(
        auto acoustic_compiled,
        CreateCompiledModel(env, acoustic_options, config.acoustic_file));
    LITERT_ASSIGN_OR_RETURN(
        auto vocoder_compiled,
        CreateCompiledModel(env, vocoder_options, config.vocoder_file));
    acoustic_opt.emplace(std::move(acoustic_compiled));
    vocoder_opt.emplace(std::move(vocoder_compiled));
  }
  CompiledModel acoustic = std::move(*acoustic_opt);
  CompiledModel vocoder = std::move(*vocoder_opt);

  // Verify that acoustic and vocoder models agree on frame capacity.
  auto acoustic_type = acoustic.GetOutputTensorType("acoustic_features");
  auto vocoder_type = vocoder.GetInputTensorType("acoustic_features");
  if (acoustic_type && vocoder_type) {
    auto acoustic_dims = acoustic_type->Layout().Dimensions();
    auto vocoder_dims = vocoder_type->Layout().Dimensions();
    if (!acoustic_dims.empty() && !vocoder_dims.empty() &&
        acoustic_dims.back() != vocoder_dims.back()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Kokoro acoustic model frame capacity (%d) does not match vocoder "
          "frame capacity (%d). Both models must be exported with the same "
          "frame capacity.",
          acoustic_dims.back(), vocoder_dims.back()));
    }
  }

  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "kokoro_acoustic", std::make_shared<CompiledModel>(std::move(acoustic))));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "kokoro_vocoder", std::make_shared<CompiledModel>(std::move(vocoder))));

  return absl::OkStatus();
}

TextChunkConfig ReviseTextChunkConfigForKokoro(
    const KokoroModelConfig& config, TextChunkConfig text_chunk_config) {
  if (config.target_bucket > 0 && text_chunk_config.max_buffer_size == 0) {
    text_chunk_config.max_buffer_size =
        std::min(120, static_cast<int>(config.target_bucket * 0.9));
  }
  return text_chunk_config;
}

absl::StatusOr<TtsSession::Components> CreateKokoroComponents(
    const KokoroModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<StreamTextSource> absl_nonnull text_source,
    std::shared_ptr<ModelResources> resources) {
  TtsSession::Components components;
  components.text_source = std::move(text_source);

  // Stage 1: Text frontend, phonemization, and unified acoustic prediction.
  ABSL_ASSIGN_OR_RETURN(auto acoustic, KokoroAcousticStage::Create(
                                           components.text_source.get(), config,
                                           model_folder, resources));
  // Stage 2: Neural vocoder and iSTFT audio synthesis.
  ABSL_ASSIGN_OR_RETURN(auto vocoder,
                        KokoroVocoderStage::Create(acoustic.get(), resources));

  if (acoustic->frame_capacity() != vocoder->frame_capacity()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Kokoro acoustic stage frame capacity (%d) does not match vocoder "
        "stage frame capacity (%d). Both models must be exported with the "
        "same frame capacity.",
        acoustic->frame_capacity(), vocoder->frame_capacity()));
  }

  components.intermediate_stages.push_back(std::move(acoustic));
  components.vocoder = std::move(vocoder);

  return components;
}

std::vector<std::string> GetAvailableKokoroVoices(
    absl::string_view model_folder, const lm::ModelResources* lm_resources) {
  std::vector<std::string> voices;
  if (lm_resources != nullptr) {
    for (const auto& name : lm_resources->GetGenericBinaryDataNames()) {
      // Not every GenericBinaryData section is a voice pack.
      if (name == kokoro::kEspeakNgSectionName ||
          absl::EndsWith(name, kTextNormSectionSuffix) ||
          absl::EndsWith(name, kLexiconSectionSuffix)) {
        continue;
      }
      std::string voice = kokoro::VoiceNameFromIdentifier(name);
      if (!voice.empty()) {
        voices.push_back(std::move(voice));
      }
    }
  }

  if (!model_folder.empty()) {
    std::filesystem::path base_path = std::string(model_folder);
    std::error_code ec;
    if (std::filesystem::is_regular_file(base_path, ec) &&
        !base_path.parent_path().empty()) {
      base_path = base_path.parent_path();
    }
    std::filesystem::path voices_dir = base_path / "voices";
    if (std::filesystem::is_directory(voices_dir, ec)) {
      for (const auto& entry :
           std::filesystem::directory_iterator(voices_dir, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".bin") {
          voices.push_back(entry.path().stem().string());
        }
      }
    } else if (std::filesystem::is_directory(base_path, ec)) {
      for (const auto& entry :
           std::filesystem::directory_iterator(base_path, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".bin") {
          voices.push_back(entry.path().stem().string());
        }
      }
    }
  }
  std::sort(voices.begin(), voices.end());
  voices.erase(std::unique(voices.begin(), voices.end()), voices.end());
  return voices;
}

std::string GetKokoroVoiceLanguage(absl::string_view voice_name) {
  return LanguageForVoiceName(voice_name);
}

std::string GetDefaultKokoroVoice(absl::string_view language_code) {
  std::string normalized = NormalizeLanguageCode(language_code);
  if (normalized == "es") return "ef_dora";
  if (normalized == "fr-fr") return "ff_siwis";
  if (normalized == "hi") return "hf_alpha";
  if (normalized == "it") return "if_sara";
  if (normalized == "pt-br") return "pf_dora";
  if (normalized == "en-gb") return "bf_alice";
  if (normalized == "ja") return "jf_alpha";
  if (normalized == "cmn") return "zf_xiaobei";
  return "af_heart";
}

std::string ToKokoroLanguageCode(absl::string_view language) {
  if (language.empty()) return "";
  std::string normalized = NormalizeLanguageCode(language);
  if (normalized == "en-us" || normalized == "en-gb" || normalized == "es" ||
      normalized == "fr-fr" || normalized == "hi" || normalized == "it" ||
      normalized == "pt-br" || normalized == "ja" || normalized == "cmn") {
    return normalized;
  }
  return "";
}

std::string KokoroCodeToBcp47(absl::string_view kokoro_code) {
  if (kokoro_code == "en-us") return "en-US";
  if (kokoro_code == "en-gb") return "en-GB";
  if (kokoro_code == "es") return "es";
  if (kokoro_code == "fr-fr") return "fr";
  if (kokoro_code == "hi") return "hi";
  if (kokoro_code == "it") return "it";
  if (kokoro_code == "pt-br") return "pt-BR";
  if (kokoro_code == "ja") return "ja";
  if (kokoro_code == "cmn") return "zh-CN";
  return "";
}

}  // namespace litert::omni::tts
