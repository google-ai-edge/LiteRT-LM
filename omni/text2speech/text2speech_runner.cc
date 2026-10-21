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
//   //omni/text2speech:text2speech_runner -- \
//   --model_path="$HOME/work/qwen3_tts_tflite" \
//   --model_type="qwen3" \
//   --text="Hello from Lite-RT TTS Engine, running fully on device." \
//   --output_wav="/tmp/hello_from_litert.wav" \
//   --num_threads=4 --v=2 --alsologtostderr

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/initialize.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2speech/qwen3_tts/common.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_model_type.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"
#include "miniaudio.h"  // from @miniaudio

ABSL_FLAG(std::string, model_path, "",
          "Path to the TTS model (.litertlm file or directory).");
ABSL_FLAG(std::string, model_type, "kokoro",
          "TTS model architecture type: 'kokoro' or 'qwen3'.");
ABSL_FLAG(std::string, text, "Hello from LiteRT TTS Engine.",
          "Text string to synthesize into speech.");
ABSL_FLAG(std::string, output_wav, "/tmp/output.wav",
          "Path to save the generated 24 kHz mono 16-bit PCM WAV file.");
ABSL_FLAG(std::string, language, "en-US",
          "Target language in BCP-47 format (e.g., en-US, es, hi). Defaults "
          "to 'en-US'.");
ABSL_FLAG(int, num_threads, 4,
          "Number of CPU threads for LiteRT model execution.");
ABSL_FLAG(std::string, backend, "cpu",
          "Backend to use for model execution: 'cpu' or 'gpu'.");
ABSL_FLAG(bool, force_fp32_on_gpu, false,
          "If true, forces FP32 precision on GPU instead of FP16.");
ABSL_FLAG(std::string, vocoder_backend, "",
          "Optional backend override for Kokoro vocoder ('cpu' or 'gpu'). "
          "If empty, inherits the engine backend.");

ABSL_FLAG(std::string, cache_dir, "",
          "Directory to store/load XNNPack weight cache.");
ABSL_FLAG(int, max_frames, 512,
          "Maximum number of audio codebook frames to generate.");
ABSL_FLAG(int, target_bucket, 0,
          "Target sequence length bucket (0 = load all buckets).");
ABSL_FLAG(std::string, voice, "af_heart",
          "Voice profile name or file for Kokoro (e.g. 'af_heart', 'af_bella', "
          "'af_sarah', 'am_michael', 'bf_emma', etc.).");

namespace {

absl::Status WriteWavFile(absl::string_view filename,
                          const std::vector<float>& samples, int sample_rate) {
  std::vector<int16_t> pcm16(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    float s = std::clamp(samples[i], -1.0f, 1.0f);
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
  ABSL_RETURN_IF_ERROR(
      ma_encoder_write_pcm_frames(&encoder, pcm16.data(), pcm16.size(),
                                  /*framesWritten=*/nullptr) == MA_SUCCESS
          ? absl::OkStatus()
          : absl::InternalError("Failed to write PCM frames"));
  ma_encoder_uninit(&encoder);
  return absl::OkStatus();
}

absl::Status Run() {
  const std::string model_path = absl::GetFlag(FLAGS_model_path);

  std::string input_text = absl::GetFlag(FLAGS_text);
  if (input_text.empty()) {
    return absl::InvalidArgumentError("--text flag must not be empty.");
  }

  std::string model_type_str = absl::GetFlag(FLAGS_model_type);
  litert::omni::OmniEngine::Settings engine_settings;
  engine_settings.cache_dir = absl::GetFlag(FLAGS_cache_dir);
  engine_settings.num_threads = absl::GetFlag(FLAGS_num_threads);
  engine_settings.force_fp32_on_gpu = absl::GetFlag(FLAGS_force_fp32_on_gpu);

  ABSL_ASSIGN_OR_RETURN(
      engine_settings.backend,
      litert::omni::ParseOmniBackend(absl::GetFlag(FLAGS_backend)));

  litert::lm::proto::Text2SpeechMetadata metadata;
  std::string model_name;
  if (model_type_str == "kokoro" || model_type_str == "kokoro_tts" ||
      model_type_str == "kokoro-82m") {
    model_name = "kokoro";
    auto* kokoro = metadata.mutable_tts_model_type()->mutable_kokoro();
    kokoro->set_voice_name(absl::GetFlag(FLAGS_voice));
    kokoro->set_target_bucket(absl::GetFlag(FLAGS_target_bucket));
    kokoro->set_language(absl::GetFlag(FLAGS_language));
    const std::string voc_backend_str = absl::GetFlag(FLAGS_vocoder_backend);
    if (!voc_backend_str.empty()) {
      kokoro->set_vocoder_backend(voc_backend_str);
    }
  } else if (model_type_str == "qwen3" || model_type_str == "qwen3_tts" ||
             model_type_str == "qwen3-tts") {
    model_name = "qwen3-tts";
    auto* qwen3 = metadata.mutable_tts_model_type()->mutable_qwen3_tts();
    const std::string requested_lang = absl::GetFlag(FLAGS_language);
    if (!requested_lang.empty()) {
      qwen3->set_language(requested_lang);
    }
    qwen3->set_max_frames(absl::GetFlag(FLAGS_max_frames));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown model_type: ", model_type_str));
  }

  const std::string effective_cache_dir = engine_settings.cache_dir.empty()
                                              ? "/tmp/tts_models"
                                              : engine_settings.cache_dir;
  if (!model_path.empty()) {
    engine_settings.model_path = model_path;
  } else {
    ABSL_ASSIGN_OR_RETURN(
        litert::omni::OmniRegistry::RegisteredModel reg_model,
        litert::omni::OmniRegistry::Get().GetRegisteredModel(model_name));
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

  litert::lm::proto::Text2SpeechSessionConfig session_config;
  const std::string requested_lang_str = absl::GetFlag(FLAGS_language);
  if (!requested_lang_str.empty()) {
    session_config.set_language(requested_lang_str);
  }
  const std::string requested_voice_str = absl::GetFlag(FLAGS_voice);
  if (!requested_voice_str.empty()) {
    session_config.set_voice(requested_voice_str);
  }

  ABSL_ASSIGN_OR_RETURN(auto engine, litert::omni::OmniEngine::Create(
                                         model_name, engine_settings,
                                         &session_config, &metadata));

  auto input_source = std::make_unique<litert::omni::PushInputSource>();
  litert::omni::PushInputSource* raw_input_source = input_source.get();
  ABSL_ASSIGN_OR_RETURN(auto session,
                        engine->CreateSession(std::move(input_source)));

  std::cout << "Synthesizing text: \"" << input_text << "\"\n";
  absl::Time start_time = absl::Now();

  ABSL_RETURN_IF_ERROR(raw_input_source->PushInput(
      litert::omni::OmniSession::TextInput{.text = input_text}));
  raw_input_source->Finish();
  ABSL_ASSIGN_OR_RETURN(auto output, session->Flush());
  const auto& audio = std::get<litert::omni::AudioOutput>(output);

  absl::Time end_time = absl::Now();
  double wall_time_sec = absl::ToDoubleSeconds(end_time - start_time);

  int sample_rate = audio.sample_rate_hz > 0
                        ? audio.sample_rate_hz
                        : litert::omni::tts::qwen3_tts::kAudioSampleRateHz;
  double duration_sec =
      static_cast<double>(audio.pcm_samples.size()) / sample_rate;
  double rtf = wall_time_sec / std::max(0.001, duration_sec);

  std::cout << "\n============================================================="
               "===================\n";
  std::cout << "                       TtsEngine Execution Summary             "
               "                 \n";
  std::cout << "==============================================================="
               "=================\n";
  std::cout << absl::StrFormat("  Synthesis Wall-Clock Latency : %8.3f s\n",
                               wall_time_sec);
  std::cout << absl::StrFormat(
      "  Generated Audio Duration     : %8.3f s (%zu samples)\n", duration_sec,
      audio.pcm_samples.size());
  std::cout << absl::StrFormat(
      "  Real-Time Factor (RTF)       : %8.3f x (lower is faster)\n", rtf);
  std::cout << "==============================================================="
               "=================\n\n";

  std::string output_wav = absl::GetFlag(FLAGS_output_wav);
  if (!output_wav.empty()) {
    ABSL_RETURN_IF_ERROR(
        WriteWavFile(output_wav, audio.pcm_samples, sample_rate));
    std::cout << "Saved audio to: " << output_wav << "\n";
  }

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  absl::Status status = Run();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "TTS execution failed: " << status;
    std::cerr << "ERROR: " << status << "\n";
    return 1;
  }

  return 0;
}
