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

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <iostream>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/file_audio_source.h"
#include "runtime/proto/speech2text_session_config.pb.h"

ABSL_FLAG(std::string, model_name, "parakeet-tdt-0.6b-v3",
          "ASR model name as defined in model metadata presets.");
ABSL_FLAG(std::string, audio_path, "", "Path to input audio file (WAV).");
ABSL_FLAG(std::string, cache_dir, "/tmp/omni_models",
          "Cache directory to download and store models and tokenizers.");
ABSL_FLAG(std::string, backend, "cpu",
          "Hardware backend accelerator: cpu, gpu, or npu.");
ABSL_FLAG(bool, force_fp32_on_gpu, false,
          "If true, forces FP32 precision on GPU instead of FP16.");
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

class FileOmniInputSource : public litert::omni::OmniSession::InputSource {
 public:
  explicit FileOmniInputSource(
      std::unique_ptr<litert::omni::asr::FileAudioSource> source)
      : source_(std::move(source)) {}

 protected:
  void ResetInternal() override { source_->Reset(); }
  bool NeedScheduleInternal() const override { return source_->NeedSchedule(); }
  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    ABSL_RETURN_IF_ERROR(source_->Schedule());
    ABSL_ASSIGN_OR_RETURN(std::vector<float> chunk, source_->GetOutput());
    PushOutput(
        litert::omni::OmniSession::AudioInput{.pcm_samples = std::move(chunk)});
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<litert::omni::asr::FileAudioSource> source_;
};

absl::Status RunAsrRunner(absl::string_view model_name,
                          absl::string_view audio_path,
                          absl::string_view cache_dir,
                          absl::string_view backend_flag, int num_threads,
                          float overlap_ratio,
                          absl::string_view text_merger_flag,
                          absl::string_view model_path_flag) {
  if (audio_path.empty()) {
    return absl::InvalidArgumentError("--audio_path flag is required.");
  }
  if (!cache_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(std::string(cache_dir), ec);
    if (ec) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to create --cache_dir '", cache_dir, "': ", ec.message()));
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      litert::omni::OmniRegistry::RegisteredModel reg_model,
      litert::omni::OmniRegistry::Get().GetRegisteredModel(model_name));
  if (reg_model.metadata_proto_type_id !=
      "litert.lm.proto.Speech2TextMetadata") {
    return absl::NotFoundError(
        absl::StrCat("Unsupported or unknown ASR model: ", model_name));
  }

  litert::omni::OmniEngine::Settings engine_settings;
  engine_settings.cache_dir = std::string(cache_dir);
  engine_settings.num_threads = num_threads;
  engine_settings.force_fp32_on_gpu = absl::GetFlag(FLAGS_force_fp32_on_gpu);
  if (!model_path_flag.empty()) {
    engine_settings.model_path = std::string(model_path_flag);
  } else {
    engine_settings.model_remote_url = reg_model.model_remote_url;
    std::string model_ext =
        std::filesystem::path(engine_settings.model_remote_url)
            .extension()
            .string();
    if (model_ext.empty()) {
      model_ext = ".litertlm";
    }
    engine_settings.model_path =
        (std::filesystem::path(engine_settings.cache_dir) /
         absl::StrCat(model_name, model_ext))
            .string();
    if (!engine_settings.model_remote_url.empty() &&
        !std::filesystem::exists(engine_settings.model_path)) {
      ABSL_RETURN_IF_ERROR(litert::omni::DownloadFileWithCurl(
          engine_settings.model_remote_url, engine_settings.model_path));
    }
  }

  ABSL_ASSIGN_OR_RETURN(engine_settings.backend,
                        litert::omni::ParseOmniBackend(backend_flag));

  litert::lm::proto::Speech2TextSessionConfig session_config;
  session_config.set_overlap_ratio(overlap_ratio);
  if (!text_merger_flag.empty()) {
    if (absl::EqualsIgnoreCase(text_merger_flag, "levenshtein")) {
      session_config.set_text_merger_type(
          litert::lm::proto::Speech2TextSessionConfig::
              TEXT_MERGER_TYPE_LEVENSHTEIN);
    } else {
      session_config.set_text_merger_type(
          litert::lm::proto::Speech2TextSessionConfig::
              TEXT_MERGER_TYPE_TIMESTAMP);
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      auto engine, litert::omni::OmniEngine::Create(model_name, engine_settings,
                                                    &session_config));
  ABSL_ASSIGN_OR_RETURN(
      auto file_source,
      litert::omni::asr::FileAudioSource::Create(
          audio_path, /*interval=*/absl::Seconds(1),
          /*overlap=*/absl::ZeroDuration(), /*sample_rate_hz=*/16000));
  ABSL_ASSIGN_OR_RETURN(
      auto session, engine->CreateSession(std::make_unique<FileOmniInputSource>(
                        std::move(file_source))));

  ABSL_LOG(INFO) << "Starting speech recognition on " << audio_path << "...";
  while (true) {
    auto result = session->ProcessNext();
    if (!result.ok()) {
      if (absl::IsOutOfRange(result.status())) {
        auto flush_result = session->Flush();
        if (flush_result.ok()) {
          const auto* text_out =
              std::get_if<litert::omni::OmniSession::TextOutput>(
                  &*flush_result);
          if (text_out != nullptr && !text_out->confirmed_text.empty()) {
            std::cout << text_out->confirmed_text;
          }
        }
        std::cout << std::endl;
        break;
      }
      return result.status();
    }
    const auto* text_out =
        std::get_if<litert::omni::OmniSession::TextOutput>(&*result);
    if (text_out != nullptr && !text_out->confirmed_text.empty()) {
      std::cout << text_out->confirmed_text << " " << std::flush;
    }
  }
  ABSL_LOG(INFO) << "Finished speech recognition.";
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = RunAsrRunner(
      absl::GetFlag(FLAGS_model_name), absl::GetFlag(FLAGS_audio_path),
      absl::GetFlag(FLAGS_cache_dir), absl::GetFlag(FLAGS_backend),
      absl::GetFlag(FLAGS_num_threads), absl::GetFlag(FLAGS_overlap_ratio),
      absl::GetFlag(FLAGS_text_merger_type), absl::GetFlag(FLAGS_model_path));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "ASR Runner failed: " << status;
    return 1;
  }
  return 0;
}
