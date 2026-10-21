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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
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
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"
#include "miniaudio.h"  // from @miniaudio
#include "stb_image_write.h"  // from @stb

// Common engine flags.
ABSL_FLAG(std::string, model_name, "",
          "Model name (e.g., parakeet-tdt-0.6b-v3, kokoro, bonsai-flux2). If "
          "empty, inferred from --model_path.");
ABSL_FLAG(std::string, model_path, "",
          "Path to a .litertlm model package or model directory.");
ABSL_FLAG(std::string, backend, "cpu",
          "Hardware backend accelerator: cpu, gpu, or npu.");
ABSL_FLAG(bool, force_fp32_on_gpu, false,
          "If true, forces FP32 precision on GPU instead of FP16.");
ABSL_FLAG(int, num_threads, 4, "Number of CPU threads for LiteRT inference.");
ABSL_FLAG(std::string, cache_dir, "/tmp/omni_cache",
          "Cache directory for downloaded models and compiled backend caches.");

// Input & output flags.
ABSL_FLAG(std::string, audio_path, "",
          "Path to input WAV audio file (for ASR transcription).");
ABSL_FLAG(std::string, text, "",
          "Input text prompt (for TTS synthesis or Text-to-Image generation).");
ABSL_FLAG(std::string, output_path, "",
          "Output file path (.wav for TTS audio, .png for Text-to-Image, or "
          "optional .txt file for ASR transcript).");

// ASR session flags.
ABSL_FLAG(float, overlap_ratio, 0.4f,
          "ASR chunk overlap ratio between 0.0 and 1.0.");
ABSL_FLAG(std::string, text_merger_type, "timestamp",
          "ASR text merger strategy: timestamp or levenshtein.");

// TTS session flags.
ABSL_FLAG(std::string, language, "en-US",
          "TTS language code (BCP-47, e.g., en-US, es, zh-CN).");
ABSL_FLAG(std::string, voice, "",
          "Optional TTS voice ID (e.g., af_heart). If empty, uses default.");

// Text-to-Image session flags.
ABSL_FLAG(int, img_size, 0,
          "Target output image width and height in pixels (e.g., 256 or 512). "
          "If 0, uses the model's default size.");
ABSL_FLAG(int, steps, 4, "Number of denoising steps for Text-to-Image.");
ABSL_FLAG(uint64_t, seed, 42, "Random seed for Text-to-Image generation.");

namespace {

using ::litert::omni::AudioOutput;
using ::litert::omni::ImageOutput;
using ::litert::omni::MessageLite;
using ::litert::omni::OmniEngine;
using ::litert::omni::OmniRegistry;
using ::litert::omni::OmniSession;
using ::litert::omni::ParseOmniBackend;
using ::litert::omni::PushInputSource;
using ::litert::omni::TextOutput;

absl::Status ReadWavFile(absl::string_view audio_path, int sample_rate_hz,
                         std::vector<float>& pcm_samples) {
  ma_decoder_config config =
      ma_decoder_config_init(ma_format_f32, /*channels=*/1, sample_rate_hz);
  ma_decoder decoder;
  if (ma_decoder_init_file(std::string(audio_path).c_str(), &config,
                           &decoder) != MA_SUCCESS) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open or read WAV file: ", audio_path));
  }
  absl::Cleanup uninit_decoder = [&decoder] { ma_decoder_uninit(&decoder); };

  constexpr ma_uint64 kChunkFrames = 16000;
  std::vector<float> chunk(kChunkFrames);
  while (true) {
    ma_uint64 frames_read = 0;
    ma_result res = ma_decoder_read_pcm_frames(&decoder, chunk.data(),
                                               kChunkFrames, &frames_read);
    if (frames_read > 0) {
      pcm_samples.insert(pcm_samples.end(), chunk.begin(),
                         chunk.begin() + frames_read);
    }
    if (res == MA_AT_END || frames_read == 0) {
      break;
    }
    if (res != MA_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("Failed while reading audio frames from: ", audio_path));
    }
  }
  return absl::OkStatus();
}

absl::Status WriteWavFile(absl::string_view filename,
                          const std::vector<float>& samples,
                          int sample_rate_hz) {
  if (sample_rate_hz <= 0) {
    return absl::InvalidArgumentError("Invalid audio sample rate.");
  }
  ma_encoder_config config = ma_encoder_config_init(
      ma_encoding_format_wav, ma_format_s16, /*channels=*/1, sample_rate_hz);
  ma_encoder encoder;
  if (ma_encoder_init_file(std::string(filename).c_str(), &config, &encoder) !=
      MA_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Failed to open WAV file for writing: ", filename));
  }
  absl::Cleanup uninit_encoder = [&encoder] { ma_encoder_uninit(&encoder); };

  std::vector<int16_t> pcm16(samples.size());
  ma_pcm_f32_to_s16(pcm16.data(), samples.data(), samples.size(),
                    ma_dither_mode_none);
  ma_uint64 frames_written = 0;
  if (ma_encoder_write_pcm_frames(&encoder, pcm16.data(), pcm16.size(),
                                  &frames_written) != MA_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("Failed to write PCM frames to: ", filename));
  }
  return absl::OkStatus();
}

absl::Status WritePngFile(absl::string_view filename,
                          const ImageOutput& image) {
  if (image.width <= 0 || image.height <= 0 || image.channels <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid image dimensions: %dx%dx%d", image.width,
                        image.height, image.channels));
  }
  const int stride_in_bytes = image.width * image.channels;
  if (stbi_write_png(std::string(filename).c_str(), image.width, image.height,
                     image.channels, image.rgb_data.data(),
                     stride_in_bytes) == 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to write PNG file: %s", filename));
  }
  return absl::OkStatus();
}

struct OutputCollector {
  std::string output_path;
  std::vector<float> accumulated_audio;
  int audio_sample_rate_hz = 24000;
  std::string accumulated_transcript;
  bool printed_text = false;
};

absl::Status HandleOutput(const OmniSession::Output& out,
                          OutputCollector& collector) {
  if (const auto* text_out = std::get_if<TextOutput>(&out)) {
    if (!text_out->confirmed_text.empty()) {
      if (collector.printed_text) {
        std::cout << " ";
        collector.accumulated_transcript.append(" ");
      }
      std::cout << text_out->confirmed_text << std::flush;
      collector.accumulated_transcript.append(text_out->confirmed_text);
      collector.printed_text = true;
    }
  } else if (const auto* audio_out = std::get_if<AudioOutput>(&out)) {
    if (audio_out->sample_rate_hz > 0) {
      collector.audio_sample_rate_hz = audio_out->sample_rate_hz;
    }
    collector.accumulated_audio.insert(collector.accumulated_audio.end(),
                                       audio_out->pcm_samples.begin(),
                                       audio_out->pcm_samples.end());
  } else if (const auto* image_out = std::get_if<ImageOutput>(&out)) {
    const std::string png_path = collector.output_path.empty()
                                     ? "/tmp/omni_output.png"
                                     : collector.output_path;
    ABSL_RETURN_IF_ERROR(WritePngFile(png_path, *image_out));
    std::cout << absl::StrFormat("Saved %dx%d image to %s\n", image_out->width,
                                 image_out->height, png_path);
  }
  return absl::OkStatus();
}

absl::Status RunOmniRunner() {
  const std::string audio_path = absl::GetFlag(FLAGS_audio_path);
  const std::string text = absl::GetFlag(FLAGS_text);
  if (audio_path.empty() == text.empty()) {
    return absl::InvalidArgumentError(
        "Exactly one of --audio_path (for ASR) or --text (for TTS / "
        "Text-to-Image) must be provided.");
  }

  const std::string cache_dir = absl::GetFlag(FLAGS_cache_dir);
  if (!cache_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (ec) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to create --cache_dir '", cache_dir, "': ", ec.message()));
    }
  }

  OmniEngine::Settings settings;
  settings.cache_dir = cache_dir;
  settings.num_threads = absl::GetFlag(FLAGS_num_threads);
  settings.force_fp32_on_gpu = absl::GetFlag(FLAGS_force_fp32_on_gpu);
  ABSL_ASSIGN_OR_RETURN(settings.backend,
                        ParseOmniBackend(absl::GetFlag(FLAGS_backend)));

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (!model_path.empty()) {
    settings.model_path = model_path;
  }

  std::string model_name = absl::GetFlag(FLAGS_model_name);
  if (model_name.empty()) {
    if (model_path.empty()) {
      return absl::InvalidArgumentError(
          "Either --model_path or --model_name must be provided.");
    }
    std::filesystem::path p(model_path);
    model_name = p.stem().string();
    if (model_name.empty()) {
      model_name = p.parent_path().filename().string();
    }
    // Published files carry variant suffixes (e.g. `kokoro_int8_gpu`), so fall
    // back to the registered name contained in the file name.
    if (!OmniRegistry::Get().IsModelRegistered(model_name)) {
      if (auto inferred = OmniRegistry::Get().FindModelNameForPath(model_path);
          inferred.ok()) {
        model_name = *std::move(inferred);
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(const auto reg,
                        OmniRegistry::Get().GetRegisteredModel(model_name));

  // If no local --model_path was given and the model is registered with a
  // remote URL, download it into --cache_dir on first run.
  if (settings.model_path.empty() && !reg.model_remote_url.empty()) {
    std::string ext =
        std::filesystem::path(reg.model_remote_url).extension().string();
    if (ext.empty()) {
      ext = ".litertlm";
    }
    settings.model_path =
        (std::filesystem::path(cache_dir) / absl::StrCat(model_name, ext))
            .string();
    if (!std::filesystem::exists(settings.model_path)) {
      ABSL_RETURN_IF_ERROR(litert::omni::DownloadFileWithCurl(
          reg.model_remote_url, settings.model_path));
    }
  }

  litert::lm::proto::Speech2TextSessionConfig asr_config;
  litert::lm::proto::Text2SpeechSessionConfig tts_config;
  litert::lm::proto::Text2ImageSessionConfig t2i_config;
  const MessageLite* session_config = nullptr;

  if (reg.metadata_proto_type_id == "litert.lm.proto.Speech2TextMetadata") {
    if (audio_path.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("ASR model '", model_name, "' requires --audio_path."));
    }
    asr_config.set_overlap_ratio(absl::GetFlag(FLAGS_overlap_ratio));
    if (absl::EqualsIgnoreCase(absl::GetFlag(FLAGS_text_merger_type),
                               "levenshtein")) {
      asr_config.set_text_merger_type(
          litert::lm::proto::Speech2TextSessionConfig::
              TEXT_MERGER_TYPE_LEVENSHTEIN);
    } else {
      asr_config.set_text_merger_type(
          litert::lm::proto::Speech2TextSessionConfig::
              TEXT_MERGER_TYPE_TIMESTAMP);
    }
    session_config = &asr_config;
  } else if (reg.metadata_proto_type_id ==
             "litert.lm.proto.Text2ImageMetadata") {
    if (text.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Text-to-Image model '", model_name, "' requires --text."));
    }
    const int img_size = absl::GetFlag(FLAGS_img_size);
    if (img_size > 0) {
      t2i_config.set_width(img_size);
      t2i_config.set_height(img_size);
    }
    t2i_config.set_num_inference_steps(absl::GetFlag(FLAGS_steps));
    t2i_config.set_seed(absl::GetFlag(FLAGS_seed));
    session_config = &t2i_config;
  } else if (reg.metadata_proto_type_id ==
             "litert.lm.proto.Text2SpeechMetadata") {
    if (text.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("TTS model '", model_name, "' requires --text."));
    }
    const std::string language = absl::GetFlag(FLAGS_language);
    const std::string voice = absl::GetFlag(FLAGS_voice);
    if (!language.empty()) {
      tts_config.set_language(language);
    }
    if (!voice.empty()) {
      tts_config.set_voice(voice);
    }
    session_config = &tts_config;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported metadata proto type '", reg.metadata_proto_type_id,
        "' for model '", model_name, "'."));
  }

  const absl::Time load_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      auto engine, OmniEngine::Create(model_name, settings, session_config));
  const double load_sec = absl::ToDoubleSeconds(absl::Now() - load_start);
  ABSL_LOG(INFO) << absl::StrFormat("Loaded OmniEngine '%s' in %.3f s.",
                                    engine->model_name(), load_sec);

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  ABSL_ASSIGN_OR_RETURN(auto session,
                        engine->CreateSession(std::move(input_source)));

  if (!audio_path.empty()) {
    constexpr int kDefaultAsrSampleRateHz = 16000;
    std::vector<float> pcm_samples;
    ABSL_RETURN_IF_ERROR(
        ReadWavFile(audio_path, kDefaultAsrSampleRateHz, pcm_samples));
    ABSL_RETURN_IF_ERROR(
        raw_input_source->PushInput(OmniSession::AudioInputMetadata{
            .sample_rate_hz = kDefaultAsrSampleRateHz, .num_channels = 1}));
    constexpr size_t kPushChunkSamples = 16000;  // 1-second chunks at 16 kHz.
    for (size_t offset = 0; offset < pcm_samples.size();
         offset += kPushChunkSamples) {
      const size_t end =
          std::min(offset + kPushChunkSamples, pcm_samples.size());
      ABSL_RETURN_IF_ERROR(raw_input_source->PushInput(OmniSession::AudioInput{
          .pcm_samples = std::vector<float>(pcm_samples.begin() + offset,
                                            pcm_samples.begin() + end)}));
    }
  } else {
    ABSL_RETURN_IF_ERROR(
        raw_input_source->PushInput(OmniSession::TextInput{.text = text}));
  }
  raw_input_source->Finish();

  OutputCollector collector{
      .output_path = absl::GetFlag(FLAGS_output_path),
  };

  const absl::Time run_start = absl::Now();
  while (true) {
    auto next = session->ProcessNext();
    if (!next.ok()) {
      if (absl::IsOutOfRange(next.status()) ||
          absl::IsNotFound(next.status())) {
        ABSL_ASSIGN_OR_RETURN(auto flushed, session->Flush());
        ABSL_RETURN_IF_ERROR(HandleOutput(flushed, collector));
        break;
      }
      return next.status();
    }
    ABSL_RETURN_IF_ERROR(HandleOutput(*next, collector));
  }
  const double run_sec = absl::ToDoubleSeconds(absl::Now() - run_start);

  if (collector.printed_text) {
    std::cout << std::endl;
  }
  if (!audio_path.empty() && !collector.output_path.empty()) {
    std::ofstream out_file(collector.output_path);
    if (!out_file.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open transcript file for writing: ",
                       collector.output_path));
    }
    out_file << collector.accumulated_transcript << "\n";
  }
  if (!collector.accumulated_audio.empty()) {
    const std::string wav_path = collector.output_path.empty()
                                     ? "/tmp/omni_output.wav"
                                     : collector.output_path;
    ABSL_RETURN_IF_ERROR(WriteWavFile(wav_path, collector.accumulated_audio,
                                      collector.audio_sample_rate_hz));
    const double audio_duration_sec =
        static_cast<double>(collector.accumulated_audio.size()) /
        collector.audio_sample_rate_hz;
    std::cout << absl::StrFormat(
        "Saved %.2f s of audio (%zu samples @ %d Hz) to %s (inference: %.3f "
        "s)\n",
        audio_duration_sec, collector.accumulated_audio.size(),
        collector.audio_sample_rate_hz, wav_path, run_sec);
  } else {
    ABSL_LOG(INFO) << absl::StrFormat("Completed in %.3f s.", run_sec);
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = RunOmniRunner();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Omni runner failed: " << status;
    std::cerr << "ERROR: " << status << "\n";
    return 1;
  }
  return 0;
}
