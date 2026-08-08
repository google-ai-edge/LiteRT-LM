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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "omni/asr/asr_engine.h"
#include "omni/asr/file_audio_source.h"
#include "omni/asr/log_mel_spectrogram_processor.h"

ABSL_FLAG(std::string, model_name, "parakeet-tdt-0.6b-v3",
          "ASR model name as defined in metadata JSON.");
ABSL_FLAG(std::string, metadata_path,
          "omni/asr/model_metadata.json",
          "Path to model_metadata.json file.");
ABSL_FLAG(std::string, audio_path, "", "Path to input audio file (WAV).");
ABSL_FLAG(std::string, cache_dir, "/tmp/asr_models",
          "Cache directory to download and store models and tokenizers.");
ABSL_FLAG(std::string, backend, "cpu",
          "Hardware backend accelerator: cpu, gpu, or npu.");
ABSL_FLAG(int, num_threads, 4, "Number of CPU threads for LiteRT inference.");
ABSL_FLAG(float, overlap_ratio, 0.4f,
          "Overlap ratio between consecutive audio chunks from audio source "
          "between 0.0 and 1.0.");
ABSL_FLAG(std::string, text_merger_type, "timestamp",
          "Text merger implementation to use: timestamp or levenshtein.");

namespace {

using json = nlohmann::json;

absl::Status DownloadFileWithCurl(absl::string_view url,
                                  absl::string_view target_path) {
  ABSL_LOG(INFO) << "Downloading " << url << " to " << target_path;
  std::string cmd =
      absl::StrCat("mkdir -p $(dirname \"", target_path,
                   "\") && curl -L -s -o \"", target_path, "\" \"", url, "\"");
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to download from %s to %s", url, target_path));
  }
  return absl::OkStatus();
}

absl::StatusOr<litert::omni::asr::AsrEngineConfig> LoadConfigFromJsonFile(
    absl::string_view json_path, absl::string_view model_name,
    absl::string_view cache_dir, absl::string_view backend_flag,
    int num_threads, float overlap_ratio, absl::string_view text_merger_flag) {
  std::ifstream f(std::string(json_path).c_str());
  if (!f.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Could not open JSON config file: ", json_path));
  }
  std::stringstream buffer;
  buffer << f.rdbuf();

  auto j = json::parse(buffer.str(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) {
    return absl::InvalidArgumentError("Failed to parse metadata JSON");
  }
  if (!j.contains(std::string(model_name))) {
    return absl::NotFoundError(
        absl::StrCat("Model ", model_name, " not found in configuration JSON"));
  }
  const auto& m = j[std::string(model_name)];
  litert::omni::asr::AsrEngineConfig config;
  config.model_name = std::string(model_name);
  config.cache_dir = std::string(cache_dir);
  config.num_threads = num_threads;
  config.overlap_ratio = overlap_ratio;

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

  std::string merger_str;
  if (m.contains("textMergerType")) {
    merger_str = m["textMergerType"].get<std::string>();
  }
  if (!text_merger_flag.empty()) {
    merger_str = std::string(text_merger_flag);
  }
  if (absl::EqualsIgnoreCase(merger_str, "levenshtein")) {
    config.text_merger_type =
        litert::omni::asr::AsrEngineConfig::TextMergerType::kLevenshtein;
  } else {
    config.text_merger_type =
        litert::omni::asr::AsrEngineConfig::TextMergerType::kTimestamp;
  }

  if (absl::StrContains(config.model_name, "tdt")) {
    config.decoder_type = litert::omni::asr::AsrEngineConfig::DecoderType::kTdt;
  } else if (absl::StrContains(config.model_name, "ctc")) {
    config.decoder_type = litert::omni::asr::AsrEngineConfig::DecoderType::kCtc;
  } else {
    config.decoder_type =
        litert::omni::asr::AsrEngineConfig::DecoderType::kStateless;
  }

  if (backend_flag == "gpu") {
    config.backend = litert::omni::asr::AsrEngineConfig::Backend::kGpu;
  } else if (backend_flag == "npu") {
    config.backend = litert::omni::asr::AsrEngineConfig::Backend::kNpu;
  } else {
    config.backend = litert::omni::asr::AsrEngineConfig::Backend::kCpu;
  }

  if (m.contains("logMelSpectro")) {
    config.has_log_mel_config = true;
    const auto& l = m["logMelSpectro"];
    if (l.contains("nFFT")) config.log_mel_config.n_fft = l["nFFT"].get<int>();
    if (l.contains("nMels"))
      config.log_mel_config.n_mels = l["nMels"].get<int>();
    if (l.contains("nFrames"))
      config.log_mel_config.n_frames = l["nFrames"].get<int>();
    if (l.contains("transpose"))
      config.log_mel_config.transpose = l["transpose"].get<bool>();
    if (l.contains("preemphasis"))
      config.log_mel_config.preemphasis = l["preemphasis"].get<float>();
    if (l.contains("normType")) {
      std::string norm = l["normType"].get<std::string>();
      if (norm == "whisper") {
        config.log_mel_config.norm_type =
            litert::omni::asr::LogMelSpectrogramProcessor::NormType::kWhisper;
      }
    }
  } else {
    config.has_log_mel_config = false;
  }

  std::string model_filename = absl::StrCat(model_name, ".tflite");
  std::string tokenizer_filename = absl::StrCat(model_name, "_tokenizer.json");
  config.model_path = absl::StrCat(config.cache_dir, "/", model_filename);
  config.tokenizer_path =
      absl::StrCat(config.cache_dir, "/", tokenizer_filename);

  return config;
}

absl::Status RunAsrRunner(absl::string_view model_name,
                          absl::string_view metadata_path,
                          absl::string_view audio_path,
                          absl::string_view cache_dir,
                          absl::string_view backend_flag, int num_threads,
                          float overlap_ratio,
                          absl::string_view text_merger_flag) {
  if (audio_path.empty()) {
    return absl::InvalidArgumentError("--audio_path flag is required.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto config,
      LoadConfigFromJsonFile(metadata_path, model_name, cache_dir, backend_flag,
                             num_threads, overlap_ratio, text_merger_flag));
  absl::Duration interval = absl::Milliseconds(config.input_milliseconds);
  absl::Duration overlap = interval * overlap_ratio;
  ABSL_ASSIGN_OR_RETURN(
      auto engine, litert::omni::asr::AsrEngine::Create(std::move(config),
                                                        DownloadFileWithCurl));
  ABSL_ASSIGN_OR_RETURN(
      auto audio_source,
      litert::omni::asr::FileAudioSource::Create(
          audio_path, interval, overlap, engine->config().sample_rate_hz));
  ABSL_ASSIGN_OR_RETURN(auto session,
                        engine->CreateSession(std::move(audio_source)));

  ABSL_LOG(INFO) << "Starting speech recognition on " << audio_path << "...";
  while (true) {
    auto result = session->ProcessNextChunk();
    if (!result.ok()) {
      if (absl::IsOutOfRange(result.status())) {
        auto flush_result = session->Flush();
        if (flush_result.ok() && !flush_result->confirmed_text.empty()) {
          std::cout << flush_result->confirmed_text;
        }
        std::cout << std::endl;
        break;
      }
      return result.status();
    }
    if (!result->confirmed_text.empty()) {
      std::cout << result->confirmed_text << " " << std::flush;
    }
  }
  ABSL_LOG(INFO) << "Finished speech recognition.";
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = RunAsrRunner(
      absl::GetFlag(FLAGS_model_name), absl::GetFlag(FLAGS_metadata_path),
      absl::GetFlag(FLAGS_audio_path), absl::GetFlag(FLAGS_cache_dir),
      absl::GetFlag(FLAGS_backend), absl::GetFlag(FLAGS_num_threads),
      absl::GetFlag(FLAGS_overlap_ratio),
      absl::GetFlag(FLAGS_text_merger_type));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "ASR Runner failed: " << status;
    return 1;
  }
  return 0;
}
