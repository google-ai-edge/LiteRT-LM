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
#include <filesystem>  // NOLINT: Required for path manipulation.
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
#include "omni/asr/asr_engine.h"
#include "omni/asr/file_audio_source.h"
#include "omni/asr/model_metadata.h"

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
ABSL_FLAG(std::string, model_path, "",
          "Direct path to model file (.litertlm or .tflite), bypassing curl "
          "download.");

namespace {

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
    int num_threads, float overlap_ratio, absl::string_view text_merger_flag,
    absl::string_view model_path_flag) {
  std::string json_content;
  if (!json_path.empty() && std::filesystem::exists(std::string(json_path))) {
    std::ifstream f{std::string(json_path)};
    if (!f.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Could not open JSON config file: ", json_path));
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    json_content = buffer.str();
  }

  litert::omni::asr::AsrEngineConfig config;
  config.cache_dir = std::string(cache_dir);
  config.num_threads = num_threads;
  config.overlap_ratio = overlap_ratio;
  if (!model_path_flag.empty()) {
    config.model_path = std::string(model_path_flag);
  }

  ABSL_RETURN_IF_ERROR(litert::omni::asr::PopulateConfigFromMetadataJson(
      model_name, json_content, config));

  if (!text_merger_flag.empty()) {
    if (absl::EqualsIgnoreCase(text_merger_flag, "levenshtein")) {
      config.text_merger_type =
          litert::omni::asr::AsrEngineConfig::TextMergerType::kLevenshtein;
    } else {
      config.text_merger_type =
          litert::omni::asr::AsrEngineConfig::TextMergerType::kTimestamp;
    }
  }

  if (backend_flag == "gpu") {
    config.backend = litert::omni::asr::AsrEngineConfig::Backend::kGpu;
  } else if (backend_flag == "npu") {
    config.backend = litert::omni::asr::AsrEngineConfig::Backend::kNpu;
  } else {
    config.backend = litert::omni::asr::AsrEngineConfig::Backend::kCpu;
  }

  std::string model_ext =
      std::filesystem::path(config.model_url).extension().string();
  if (model_ext.empty()) {
    model_ext = ".tflite";
  }

  std::filesystem::path cache_path(config.cache_dir);
  std::string model_filename = absl::StrCat(model_name, model_ext);
  std::string tokenizer_filename = absl::StrCat(model_name, "_tokenizer.json");
  if (config.model_path.empty()) {
    config.model_path = (cache_path / model_filename).string();
  } else {
    config.model_url.clear();
  }
  if (config.tokenizer_path.empty()) {
    config.tokenizer_path = (cache_path / tokenizer_filename).string();
  }

  return config;
}

absl::Status RunAsrRunner(
    absl::string_view model_name, absl::string_view metadata_path,
    absl::string_view audio_path, absl::string_view cache_dir,
    absl::string_view backend_flag, int num_threads, float overlap_ratio,
    absl::string_view text_merger_flag, absl::string_view model_path_flag) {
  if (audio_path.empty()) {
    return absl::InvalidArgumentError("--audio_path flag is required.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto config,
      LoadConfigFromJsonFile(metadata_path, model_name, cache_dir, backend_flag,
                             num_threads, overlap_ratio, text_merger_flag,
                             model_path_flag));
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
      absl::GetFlag(FLAGS_overlap_ratio), absl::GetFlag(FLAGS_text_merger_type),
      absl::GetFlag(FLAGS_model_path));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "ASR Runner failed: " << status;
    return 1;
  }
  return 0;
}
