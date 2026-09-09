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

#include "omni/asr/model_metadata.h"

#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "omni/asr/asr_engine.h"
#include "omni/asr/log_mel_spectrogram_processor.h"
#include "omni/asr/model_metadata_embedded.h"

namespace litert::omni::asr {

absl::string_view GetEmbeddedModelMetadataJson() {
  return kEmbeddedModelMetadataJson;
}

absl::Status PopulateConfigFromMetadataJson(absl::string_view model_name,
                                            absl::string_view json_str,
                                            AsrEngineConfig& config) {
  absl::string_view effective_json =
      json_str.empty() ? kEmbeddedModelMetadataJson : json_str;

  auto j = nlohmann::json::parse(effective_json, nullptr,
                                 /*allow_exceptions=*/false);
  if (j.is_discarded()) {
    return absl::InvalidArgumentError("Failed to parse model metadata JSON");
  }
  std::string key(model_name);
  if (!j.contains(key)) {
    return absl::NotFoundError(
        absl::StrCat("Unsupported or unknown ASR model: ", model_name));
  }
  const auto& m = j[key];
  config.model_name = std::string(model_name);

  if (m.contains("modelRemoteUrl")) {
    config.model_url = m["modelRemoteUrl"].get<std::string>();
  }
  if (m.contains("tokenizerUrl")) {
    config.tokenizer_url = m["tokenizerUrl"].get<std::string>();
  }
  if (m.contains("inputMilliseconds")) {
    config.input_milliseconds = m["inputMilliseconds"].get<int>();
  }
  if (m.contains("decodeStartTokenId")) {
    config.decode_start_token_id = m["decodeStartTokenId"].get<int>();
  }
  if (m.contains("decodeStopTokenId")) {
    config.decode_stop_token_id = m["decodeStopTokenId"].get<int>();
  }
  if (m.contains("decodeSkipUntilTokenId")) {
    config.decode_skip_until_token_id = m["decodeSkipUntilTokenId"].get<int>();
  }
  if (m.contains("stateBufferNamePatterns")) {
    config.state_buffer_name_patterns =
        m["stateBufferNamePatterns"].get<std::vector<std::string>>();
  }

  if (m.contains("textMergerType")) {
    std::string merger = m["textMergerType"].get<std::string>();
    if (absl::EqualsIgnoreCase(merger, "levenshtein")) {
      config.text_merger_type = AsrEngineConfig::TextMergerType::kLevenshtein;
    } else {
      config.text_merger_type = AsrEngineConfig::TextMergerType::kTimestamp;
    }
  } else if (model_name == "tinygemma-asr") {
    config.text_merger_type = AsrEngineConfig::TextMergerType::kLevenshtein;
  } else {
    config.text_merger_type = AsrEngineConfig::TextMergerType::kTimestamp;
  }

  std::string model_ext;
  if (!config.model_path.empty()) {
    model_ext = std::filesystem::path(config.model_path).extension().string();
  } else if (!config.model_url.empty()) {
    model_ext = std::filesystem::path(config.model_url).extension().string();
  }

  if (model_ext == ".litertlm" || model_name == "tinygemma-asr" ||
      model_name == "qwen3-asr-0.6b") {
    config.decoder_type = AsrEngineConfig::DecoderType::kLm;
  } else if (absl::StrContains(model_name, "tdt")) {
    config.decoder_type = AsrEngineConfig::DecoderType::kTdt;
  } else if (absl::StrContains(model_name, "ctc")) {
    config.decoder_type = AsrEngineConfig::DecoderType::kCtc;
  } else {
    config.decoder_type = AsrEngineConfig::DecoderType::kStateless;
  }

  if (m.contains("logMelSpectro")) {
    config.has_log_mel_config = true;
    const auto& l = m["logMelSpectro"];
    if (l.contains("nFFT")) config.log_mel_config.n_fft = l["nFFT"].get<int>();
    if (l.contains("nMels")) {
      config.log_mel_config.n_mels = l["nMels"].get<int>();
    }
    if (l.contains("nFrames")) {
      config.log_mel_config.n_frames = l["nFrames"].get<int>();
    }
    if (l.contains("transpose")) {
      config.log_mel_config.transpose = l["transpose"].get<bool>();
    }
    if (l.contains("preemphasis")) {
      config.log_mel_config.preemphasis = l["preemphasis"].get<float>();
    }
    if (l.contains("fftLength")) {
      config.log_mel_config.fft_length = l["fftLength"].get<int>();
    }
    if (l.contains("inputScale")) {
      config.log_mel_config.input_scale = l["inputScale"].get<float>();
    }
    if (l.contains("melLowHz")) {
      config.log_mel_config.mel_low_hz = l["melLowHz"].get<float>();
    }
    if (l.contains("melHighHz")) {
      config.log_mel_config.mel_high_hz = l["melHighHz"].get<float>();
    }
    if (l.contains("melFloor")) {
      config.log_mel_config.mel_floor = l["melFloor"].get<float>();
    }
    if (l.contains("normalizeMel")) {
      config.log_mel_config.normalize_mel = l["normalizeMel"].get<bool>();
    }
    if (l.contains("addFloorToMelBeforeLog")) {
      config.log_mel_config.add_floor_to_mel_before_log =
          l["addFloorToMelBeforeLog"].get<bool>();
    }
    if (l.contains("normType")) {
      std::string norm = l["normType"].get<std::string>();
      if (norm == "whisper") {
        config.log_mel_config.norm_type =
            LogMelSpectrogramProcessor::NormType::kWhisper;
      }
    }
  } else {
    config.has_log_mel_config = false;
  }

  return absl::OkStatus();
}

absl::StatusOr<AsrEngineConfig> GetConfigFromMetadataJson(
    absl::string_view model_name, absl::string_view json_str) {
  AsrEngineConfig config;
  auto status = PopulateConfigFromMetadataJson(model_name, json_str, config);
  if (!status.ok()) {
    return status;
  }
  return config;
}

}  // namespace litert::omni::asr
