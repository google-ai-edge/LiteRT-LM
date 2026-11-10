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

// Example usage:
// bazel run -c opt \
//   //omni/any2any:any2any_runner -- \
//   --model_path="$HOME/models/minicpm_o/MiniCPM-o-4_5.litertlm" \
//   --model_type="minicpm-o-4.5" \
//   --prompt="Describe what you see and hear." \
//   --image_path="/tmp/sample.jpg" \
//   --output_wav="/tmp/response.wav" \
//   --num_threads=4

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/initialize.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/file_audio_source.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "miniaudio.h"  // from @miniaudio

ABSL_FLAG(std::string, model_path, "",
          "Path to the Any2Any .litertlm model file.");
ABSL_FLAG(std::string, model_type, "minicpm-o-4.5",
          "Any2Any model architecture type (e.g. 'minicpm-o-4.5').");
ABSL_FLAG(std::string, prompt, "Hello! How can you help me today?",
          "Text prompt for the multimodal assistant.");
ABSL_FLAG(std::string, image_path, "",
          "Optional path to an input image file (PNG/JPEG).");
ABSL_FLAG(std::string, audio_path, "",
          "Optional path to an input WAV audio file.");
ABSL_FLAG(std::string, output_wav, "",
          "Optional path to save the generated 24 kHz WAV audio response.");
ABSL_FLAG(std::optional<bool>, generate_audio, std::nullopt,
          "Whether to synthesize speech audio output in addition to text.");
ABSL_FLAG(int, max_new_tokens, 512,
          "Maximum number of LLM text tokens to generate.");
ABSL_FLAG(float, temperature, 0.7f, "Sampling temperature (0.0 for greedy).");
ABSL_FLAG(std::string, language, "en-US", "Target output language.");
ABSL_FLAG(std::string, voice, "default", "Target output voice identifier.");
ABSL_FLAG(int, num_threads, 4,
          "Number of CPU threads for LiteRT model execution.");
ABSL_FLAG(std::string, backend, "cpu",
          "Backend to use for model execution: 'cpu' or 'gpu'.");
ABSL_FLAG(std::string, cache_dir, "",
          "Directory to store/load backend compilation cache.");

namespace {

absl::Status WriteWavFile(absl::string_view filename,
                          const std::vector<float>& samples, int sample_rate) {
  std::vector<int16_t> pcm16(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    const float s = std::clamp(samples[i], -1.0f, 1.0f);
    pcm16[i] = static_cast<int16_t>(s * 32767.0f);
  }

  ma_encoder_config config =
      ma_encoder_config_init(ma_encoding_format_wav, ma_format_s16,
                             /*channels=*/1, sample_rate);
  ma_encoder encoder;
  ma_result result =
      ma_encoder_init_file(std::string(filename).c_str(), &config, &encoder);
  if (result != MA_SUCCESS) {
    return absl::InternalError(
        absl::StrFormat("Failed to initialize miniaudio encoder for file: %s "
                        "(error code: %d)",
                        filename, result));
  }
  absl::Cleanup cleanup_encoder = [&] { ma_encoder_uninit(&encoder); };
  if (ma_encoder_write_pcm_frames(&encoder, pcm16.data(), pcm16.size(),
                                  /*framesWritten=*/nullptr) != MA_SUCCESS) {
    return absl::InternalError("Failed to write PCM frames");
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string model_type_str = absl::GetFlag(FLAGS_model_type);

  litert::omni::OmniEngine::Settings engine_settings;
  engine_settings.cache_dir = absl::GetFlag(FLAGS_cache_dir);
  engine_settings.num_threads = absl::GetFlag(FLAGS_num_threads);
  ABSL_ASSIGN_OR_RETURN(
      engine_settings.backend,
      litert::omni::ParseOmniBackend(absl::GetFlag(FLAGS_backend)));

  ABSL_ASSIGN_OR_RETURN(
      litert::omni::OmniRegistry::RegisteredModel reg_model,
      litert::omni::OmniRegistry::Get().GetRegisteredModel(model_type_str));
  const std::string model_name = reg_model.canonical_model_name;

  const std::string effective_cache_dir =
      (engine_settings.cache_dir.empty() ||
       engine_settings.cache_dir == litert::omni::kNoCacheDir)
          ? "/tmp/any2any_models"
          : engine_settings.cache_dir;
  if (!model_path.empty()) {
    engine_settings.model_path = model_path;
  } else {
    engine_settings.model_remote_url = reg_model.model_remote_url;
    std::string model_ext =
        std::filesystem::path(engine_settings.model_remote_url)
            .extension()
            .string();
    if (model_ext.empty()) {
      model_ext = ".litertlm";
    }
    engine_settings.model_path = (std::filesystem::path(effective_cache_dir) /
                                  absl::StrCat(model_name, model_ext))
                                     .string();
    if (!engine_settings.model_remote_url.empty() &&
        !std::filesystem::exists(engine_settings.model_path)) {
      ABSL_RETURN_IF_ERROR(litert::omni::DownloadFileWithCurl(
          engine_settings.model_remote_url, engine_settings.model_path));
    }
  }

  const std::string output_wav = absl::GetFlag(FLAGS_output_wav);
  const std::optional<bool> generate_audio =
      absl::GetFlag(FLAGS_generate_audio);

  litert::lm::proto::Any2AnySessionConfig session_config;
  if (generate_audio.has_value()) {
    session_config.set_generate_audio(*generate_audio);
  } else if (!output_wav.empty()) {
    session_config.set_generate_audio(true);
  }
  session_config.set_max_new_tokens(absl::GetFlag(FLAGS_max_new_tokens));
  session_config.set_temperature(absl::GetFlag(FLAGS_temperature));
  session_config.set_language(absl::GetFlag(FLAGS_language));
  session_config.set_voice(absl::GetFlag(FLAGS_voice));

  ABSL_ASSIGN_OR_RETURN(
      auto engine, litert::omni::OmniEngine::Create(model_name, engine_settings,
                                                    &session_config,
                                                    /*metadata=*/nullptr));

  auto input_source = std::make_unique<litert::omni::PushInputSource>();
  litert::omni::PushInputSource* raw_input = input_source.get();
  ABSL_ASSIGN_OR_RETURN(auto session,
                        engine->CreateSession(std::move(input_source)));

  const std::string image_path = absl::GetFlag(FLAGS_image_path);
  if (!image_path.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::string img_bytes,
                          litert::omni::LoadFile(image_path));
    litert::omni::OmniSession::ImageInput img_in;
    img_in.image_bytes.assign(img_bytes.begin(), img_bytes.end());
    ABSL_RETURN_IF_ERROR(raw_input->PushInput(std::move(img_in)));
  }

  const std::string audio_path = absl::GetFlag(FLAGS_audio_path);
  if (!audio_path.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        auto wav_source,
        litert::omni::asr::FileAudioSource::Create(
            audio_path, /*interval=*/absl::Seconds(10),
            /*overlap=*/absl::ZeroDuration(), /*sample_rate_hz=*/16000));
    ABSL_RETURN_IF_ERROR(
        raw_input->PushInput(litert::omni::OmniSession::AudioInputMetadata{
            .sample_rate_hz = 16000}));
    while (true) {
      absl::Status s = wav_source->Schedule();
      if (absl::IsOutOfRange(s)) break;
      ABSL_RETURN_IF_ERROR(s);
      while (wav_source->HasOutput()) {
        ABSL_ASSIGN_OR_RETURN(std::vector<float> chunk,
                              wav_source->GetOutput());
        ABSL_RETURN_IF_ERROR(
            raw_input->PushInput(litert::omni::OmniSession::AudioInput{
                .pcm_samples = std::move(chunk)}));
      }
    }
  }

  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  if (!prompt.empty()) {
    ABSL_RETURN_IF_ERROR(raw_input->PushInput(
        litert::omni::OmniSession::TextInput{.text = prompt}));
  }
  raw_input->Finish();

  const absl::Time start_time = absl::Now();
  std::string generated_text;
  std::vector<float> generated_pcm;
  int output_sample_rate = 24000;

  auto handle_output = [&](const litert::omni::Output& out) {
    if (const auto* txt = std::get_if<litert::omni::TextOutput>(&out)) {
      generated_text += txt->confirmed_text;
    } else if (const auto* aud = std::get_if<litert::omni::AudioOutput>(&out)) {
      if (aud->sample_rate_hz > 0) output_sample_rate = aud->sample_rate_hz;
      generated_pcm.insert(generated_pcm.end(), aud->pcm_samples.begin(),
                           aud->pcm_samples.end());
    }
  };

  while (true) {
    absl::StatusOr<litert::omni::Output> out = session->ProcessNext();
    if (!out.ok()) {
      if (absl::IsOutOfRange(out.status()) || absl::IsNotFound(out.status())) {
        ABSL_ASSIGN_OR_RETURN(litert::omni::Output flushed, session->Flush());
        handle_output(flushed);
        break;
      }
      return out.status();
    }
    handle_output(*out);
  }

  const double wall_sec = absl::ToDoubleSeconds(absl::Now() - start_time);
  std::cout << "Assistant: " << generated_text << "\n";
  std::cout << absl::StrFormat("Elapsed: %.3f s\n", wall_sec);

  if (!output_wav.empty() && !generated_pcm.empty()) {
    ABSL_RETURN_IF_ERROR(
        WriteWavFile(output_wav, generated_pcm, output_sample_rate));
    std::cout << "Saved audio response to: " << output_wav << "\n";
  }

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  absl::Status status = Run();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Any2Any runner failed: " << status;
    std::cerr << "ERROR: " << status << "\n";
    return 1;
  }
  return 0;
}
