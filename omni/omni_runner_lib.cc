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

#include "omni/omni_runner_lib.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
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
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"
#include "miniaudio.h"  // from @miniaudio
#include "stb_image_write.h"  // from @stb

namespace litert::omni {
namespace {

// Decodes a WAV file at `audio_path` into mono float32 PCM samples resampled to
// `sample_rate_hz`.
absl::StatusOr<std::vector<float>> ReadWavFile(absl::string_view audio_path,
                                               int sample_rate_hz) {
  if (sample_rate_hz <= 0) {
    return absl::InvalidArgumentError("Invalid audio sample rate.");
  }
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
  std::vector<float> pcm_samples;
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
  return pcm_samples;
}

// Encodes mono float32 PCM `samples` at `sample_rate_hz` into a 16-bit PCM WAV
// file at `filename`.
absl::Status WriteWavFile(absl::string_view filename,
                          absl::Span<const float> samples, int sample_rate_hz) {
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

  if (!samples.empty()) {
    std::vector<int16_t> pcm16(samples.size());
    ma_pcm_f32_to_s16(pcm16.data(), samples.data(), samples.size(),
                      ma_dither_mode_none);
    ma_uint64 frames_written = 0;
    if (ma_encoder_write_pcm_frames(&encoder, pcm16.data(), pcm16.size(),
                                    &frames_written) != MA_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("Failed to write PCM frames to: ", filename));
    }
  }
  return absl::OkStatus();
}

// Validates `image` dimensions/buffer size and writes it as a PNG to
// `filename`.
absl::Status WritePngFile(absl::string_view filename,
                          const ImageOutput& image) {
  if (image.width <= 0 || image.height <= 0 || image.channels <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid image dimensions: %dx%dx%d", image.width,
                        image.height, image.channels));
  }
  const size_t expected_bytes = static_cast<size_t>(image.width) *
                                static_cast<size_t>(image.height) *
                                static_cast<size_t>(image.channels);
  if (image.rgb_data.size() < expected_bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Image buffer too small: got %zu bytes, expected at least %zu bytes "
        "for %dx%dx%d",
        image.rgb_data.size(), expected_bytes, image.width, image.height,
        image.channels));
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

// Accumulates streamed session outputs across `ProcessNext()` and `Flush()`.
struct OutputCollector {
  std::string output_path;
  bool separate_text_chunks_with_space = false;
  std::vector<float> accumulated_audio;
  int audio_sample_rate_hz = 24000;
  std::string accumulated_transcript;
  bool printed_text = false;
  bool wrote_image = false;
};

// Dispatches one `OmniSession::Output` chunk to `out_stream`, `collector`, or
// an image file on disk.
absl::Status HandleOutput(const OmniSession::Output& out,
                          OutputCollector& collector,
                          std::ostream& out_stream) {
  if (const auto* text_out = std::get_if<TextOutput>(&out)) {
    if (!text_out->confirmed_text.empty()) {
      if (collector.printed_text && collector.separate_text_chunks_with_space) {
        out_stream << " ";
        collector.accumulated_transcript.append(" ");
      }
      out_stream << text_out->confirmed_text << std::flush;
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
    collector.wrote_image = true;
    out_stream << absl::StrFormat("Saved %dx%d image to %s\n", image_out->width,
                                  image_out->height, png_path);
  }
  return absl::OkStatus();
}

// Creates `cache_dir` when caching is enabled and returns the directory used
// for staging or downloading model files.
absl::StatusOr<std::string> PrepareDownloadDir(absl::string_view cache_dir) {
  if (!cache_dir.empty() && cache_dir != kNoCacheDir) {
    std::string dir(cache_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to create --cache_dir '", cache_dir, "': ", ec.message()));
    }
    return dir;
  }
  return std::string("/tmp/omni_models");
}

// Builds `OmniEngine::Settings` from `options`, staging any remote internal
// `model_path` into `download_dir`.
absl::StatusOr<OmniEngine::Settings> BuildEngineSettings(
    const OmniRunnerOptions& options, OmniEngine::Settings::Backend backend,
    [[maybe_unused]] absl::string_view download_dir) {
  OmniEngine::Settings settings;
  settings.cache_dir = options.cache_dir;
  settings.num_threads = options.num_threads;
  settings.force_fp32_on_gpu = options.force_fp32_on_gpu;
  settings.litert_dispatch_lib_dir = options.litert_dispatch_lib_dir;
  settings.backend = backend;

  std::string model_path = options.model_path;
  if (!model_path.empty()) {
    settings.model_path = std::move(model_path);
  }
  return settings;
}

// Returns `options.model_name` if specified, otherwise infers a registered
// model name from the un-staged `model_path`.
absl::StatusOr<std::string> ResolveModelName(const OmniRunnerOptions& options,
                                             absl::string_view model_path) {
  if (!options.model_name.empty()) {
    return options.model_name;
  }
  if (model_path.empty()) {
    return absl::InvalidArgumentError(
        "Either --model_path or --model_name must be provided.");
  }
  if (auto inferred = OmniRegistry::Get().FindModelNameForPath(model_path);
      inferred.ok()) {
    return *std::move(inferred);
  }
  std::filesystem::path p{std::string(model_path)};
  std::string fallback = p.stem().string();
  if (fallback.empty()) {
    fallback = p.parent_path().filename().string();
  }
  return fallback.empty() ? std::string(model_path) : fallback;
}

// Resolves the cached path for a remote model registered in `reg`, downloading
// it into `download_dir` on first run if not already present.
absl::StatusOr<std::string> EnsureRemoteModelDownloaded(
    const OmniRegistry::RegisteredModel& reg, absl::string_view model_name,
    absl::string_view download_dir) {
  std::string ext =
      std::filesystem::path(reg.model_remote_url).extension().string();
  if (ext.empty()) {
    ext = ".litertlm";
  }
  const std::filesystem::path dir_path{std::string(download_dir)};
  const std::filesystem::path local_file_path =
      dir_path / absl::StrCat(model_name, ext);
  std::string local_path = local_file_path.string();
  std::error_code ec;
  const bool file_exists = std::filesystem::exists(local_file_path, ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to check cached model path '", local_path,
                     "': ", ec.message()));
  }
  if (!file_exists) {
    std::filesystem::create_directories(dir_path, ec);
    if (ec) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to create download directory '", download_dir,
                       "': ", ec.message()));
    }
    ABSL_RETURN_IF_ERROR(
        DownloadFileWithCurl(reg.model_remote_url, local_path));
  }
  return local_path;
}

// Builds an `Any2AnySessionConfig` proto from `options`.
std::unique_ptr<MessageLite> BuildAny2AnySessionConfig(
    const OmniRunnerOptions& options) {
  auto config = std::make_unique<litert::lm::proto::Any2AnySessionConfig>();
  config->set_generate_audio(options.generate_audio);
  if (options.generate_image) {
    config->set_generate_image(true);
    config->set_seed(options.seed);
  }
  if (options.system_prompt.has_value()) {
    config->set_system_prompt(*options.system_prompt);
  }
  if (options.max_new_tokens.has_value()) {
    config->set_max_new_tokens(*options.max_new_tokens);
  }
  if (options.max_tts_tokens.has_value()) {
    config->set_max_tts_tokens(*options.max_tts_tokens);
  }
  if (options.temperature.has_value()) {
    config->set_temperature(*options.temperature);
  }
  if (options.tts_temperature.has_value()) {
    config->set_tts_temperature(*options.tts_temperature);
  }
  if (options.tts_top_k.has_value()) {
    config->set_tts_top_k(*options.tts_top_k);
  }
  if (options.cfg_weight.has_value()) {
    config->set_cfg_weight(*options.cfg_weight);
  }
  if (options.image_temperature.has_value()) {
    config->set_image_temperature(*options.image_temperature);
  }
  if (!options.voice.empty()) {
    config->set_voice(options.voice);
  }
  if (!options.language.empty()) {
    config->set_language(options.language);
  }
  return config;
}

// Validates ASR inputs and builds a `Speech2TextSessionConfig` proto.
absl::StatusOr<std::unique_ptr<MessageLite>> BuildSpeech2TextSessionConfig(
    const OmniRunnerOptions& options, absl::string_view model_name) {
  if (options.audio_path.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("ASR model '", model_name, "' requires --audio_path."));
  }
  if (!options.image_path.empty() || !options.text.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ASR model '", model_name,
        "' only accepts --audio_path (--image_path and --text are not "
        "supported)."));
  }
  auto config = std::make_unique<litert::lm::proto::Speech2TextSessionConfig>();
  config->set_overlap_ratio(options.overlap_ratio);
  if (absl::EqualsIgnoreCase(options.text_merger_type, "levenshtein")) {
    config->set_text_merger_type(litert::lm::proto::Speech2TextSessionConfig::
                                     TEXT_MERGER_TYPE_LEVENSHTEIN);
  } else {
    config->set_text_merger_type(litert::lm::proto::Speech2TextSessionConfig::
                                     TEXT_MERGER_TYPE_TIMESTAMP);
  }
  return config;
}

// Validates Text-to-Image inputs and builds a `Text2ImageSessionConfig` proto.
absl::StatusOr<std::unique_ptr<MessageLite>> BuildText2ImageSessionConfig(
    const OmniRunnerOptions& options, absl::string_view model_name) {
  if (options.text.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Text-to-Image model '", model_name, "' requires --text."));
  }
  if (!options.audio_path.empty() || !options.image_path.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Text-to-Image model '", model_name,
        "' only accepts --text (--audio_path and --image_path are not "
        "supported)."));
  }
  auto config = std::make_unique<litert::lm::proto::Text2ImageSessionConfig>();
  if (options.img_size > 0) {
    config->set_width(options.img_size);
    config->set_height(options.img_size);
  }
  config->set_num_inference_steps(options.steps);
  config->set_seed(options.seed);
  return config;
}

// Validates TTS inputs and builds a `Text2SpeechSessionConfig` proto.
absl::StatusOr<std::unique_ptr<MessageLite>> BuildText2SpeechSessionConfig(
    const OmniRunnerOptions& options, absl::string_view model_name) {
  if (options.text.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TTS model '", model_name, "' requires --text."));
  }
  if (!options.audio_path.empty() || !options.image_path.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TTS model '", model_name,
        "' only accepts --text (--audio_path and --image_path are not "
        "supported)."));
  }
  auto config = std::make_unique<litert::lm::proto::Text2SpeechSessionConfig>();
  config->set_language(options.language.empty() ? "en-US" : options.language);
  if (!options.voice.empty()) {
    config->set_voice(options.voice);
  }
  return config;
}

// Validates modality-specific inputs and constructs the session config proto
// corresponding to `metadata_proto_type_id`.
absl::StatusOr<std::unique_ptr<MessageLite>> BuildSessionConfig(
    const OmniRunnerOptions& options, absl::string_view model_name,
    absl::string_view metadata_proto_type_id) {
  if (metadata_proto_type_id == "litert.lm.proto.Any2AnyMetadata") {
    return BuildAny2AnySessionConfig(options);
  }
  if (metadata_proto_type_id == "litert.lm.proto.Speech2TextMetadata") {
    return BuildSpeech2TextSessionConfig(options, model_name);
  }
  if (metadata_proto_type_id == "litert.lm.proto.Text2ImageMetadata") {
    return BuildText2ImageSessionConfig(options, model_name);
  }
  if (metadata_proto_type_id == "litert.lm.proto.Text2SpeechMetadata") {
    return BuildText2SpeechSessionConfig(options, model_name);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported metadata proto type '", metadata_proto_type_id,
                   "' for model '", model_name, "'."));
}

// Loads any image, audio, and text inputs from `options` into `input_source`
// and signals end-of-input. When `is_streaming` is true, audio is pushed in
// 1-second PCM chunks; otherwise the entire audio clip is pushed as a single
// `AudioInput`.
absl::Status PushInputs(const OmniRunnerOptions& options, bool is_streaming,
                        PushInputSource& input_source) {
  if (!options.image_path.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::string bytes, LoadFile(options.image_path));
    OmniSession::ImageInput image_in;
    image_in.image_bytes.assign(bytes.begin(), bytes.end());
    ABSL_RETURN_IF_ERROR(input_source.PushInput(std::move(image_in)));
  }
  if (!options.audio_path.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        std::vector<float> pcm_samples,
        ReadWavFile(options.audio_path, kDefaultInputSampleRateHz));
    ABSL_RETURN_IF_ERROR(input_source.PushInput(OmniSession::AudioInputMetadata{
        .sample_rate_hz = kDefaultInputSampleRateHz, .num_channels = 1}));
    if (is_streaming) {
      constexpr size_t kPushChunkSamples =
          static_cast<size_t>(kDefaultInputSampleRateHz);  // 1-second chunks.
      for (size_t offset = 0; offset < pcm_samples.size();
           offset += kPushChunkSamples) {
        const size_t end =
            std::min(offset + kPushChunkSamples, pcm_samples.size());
        ABSL_RETURN_IF_ERROR(input_source.PushInput(OmniSession::AudioInput{
            .pcm_samples = std::vector<float>(pcm_samples.begin() + offset,
                                              pcm_samples.begin() + end)}));
      }
    } else if (!pcm_samples.empty()) {
      ABSL_RETURN_IF_ERROR(input_source.PushInput(
          OmniSession::AudioInput{.pcm_samples = std::move(pcm_samples)}));
    }
  }
  if (!options.text.empty()) {
    ABSL_RETURN_IF_ERROR(
        input_source.PushInput(OmniSession::TextInput{.text = options.text}));
  }
  input_source.Finish();
  return absl::OkStatus();
}

// Drains `session` until end-of-stream, flushes buffered outputs, and routes
// all emitted chunks through `HandleOutput`.
absl::Status RunSessionLoop(OmniSession& session, OutputCollector& collector,
                            std::ostream& out) {
  while (true) {
    auto next = session.ProcessNext();
    if (!next.ok()) {
      if (absl::IsOutOfRange(next.status()) ||
          absl::IsNotFound(next.status())) {
        ABSL_ASSIGN_OR_RETURN(auto flushed, session.Flush());
        return HandleOutput(flushed, collector, out);
      }
      return next.status();
    }
    ABSL_RETURN_IF_ERROR(HandleOutput(*next, collector, out));
  }
}

// Writes final transcript/audio output files and prints completion summaries.
absl::Status FinalizeOutputs(const OmniRunnerOptions& options,
                             absl::string_view metadata_proto_type_id,
                             const OutputCollector& collector, double run_sec,
                             std::ostream& out) {
  if (collector.printed_text) {
    out << std::endl;
  }
  const bool expects_text_output_file =
      (metadata_proto_type_id == "litert.lm.proto.Speech2TextMetadata") ||
      (metadata_proto_type_id == "litert.lm.proto.Any2AnyMetadata" &&
       !options.generate_audio && !options.generate_image &&
       !collector.wrote_image && collector.accumulated_audio.empty());
  if (expects_text_output_file && !collector.output_path.empty()) {
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
    out << absl::StrFormat(
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

absl::Status RunOmniRunner(const OmniRunnerOptions& options,
                           std::ostream& out) {
  if (options.audio_path.empty() && options.image_path.empty() &&
      options.text.empty()) {
    return absl::InvalidArgumentError(
        "At least one of --audio_path, --image_path, or --text must be "
        "provided.");
  }

  ABSL_ASSIGN_OR_RETURN(const std::string download_dir,
                        PrepareDownloadDir(options.cache_dir));
  ABSL_ASSIGN_OR_RETURN(const OmniEngine::Settings::Backend backend,
                        ParseOmniBackend(options.backend));
  // Resolve the model name and validate session options against the original
  // `options.model_path` before staging any remote model file so that
  // parent-directory inference works and unknown/invalid inputs fail fast.
  ABSL_ASSIGN_OR_RETURN(const std::string model_name,
                        ResolveModelName(options, options.model_path));
  ABSL_ASSIGN_OR_RETURN(const auto reg,
                        OmniRegistry::Get().GetRegisteredModel(model_name));
  ABSL_ASSIGN_OR_RETURN(
      const std::unique_ptr<MessageLite> session_config,
      BuildSessionConfig(options, model_name, reg.metadata_proto_type_id));
  ABSL_ASSIGN_OR_RETURN(OmniEngine::Settings settings,
                        BuildEngineSettings(options, backend, download_dir));
  if (settings.model_path.empty() && !reg.model_remote_url.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        settings.model_path,
        EnsureRemoteModelDownloaded(reg, model_name, download_dir));
  }

  const absl::Time load_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(auto engine, OmniEngine::Create(model_name, settings,
                                                        session_config.get()));
  const double load_sec = absl::ToDoubleSeconds(absl::Now() - load_start);
  ABSL_LOG(INFO) << absl::StrFormat("Loaded OmniEngine '%s' in %.3f s.",
                                    engine->model_name(), load_sec);

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  ABSL_ASSIGN_OR_RETURN(auto session,
                        engine->CreateSession(std::move(input_source)));
  // Speech2Text models coalesce streamed 1-second PCM chunks, whereas Any2Any
  // audio stages treat each `AudioInput` as a self-contained clip (independent
  // mel extraction/padding, rejection of clips shorter than one 160-sample hop,
  // and per-clip audio framing tokens in Qwen2.5-Omni), so only stream chunks
  // for Speech2Text models.
  const bool is_streaming =
      (reg.metadata_proto_type_id == "litert.lm.proto.Speech2TextMetadata");
  ABSL_RETURN_IF_ERROR(PushInputs(options, is_streaming, *raw_input_source));

  OutputCollector collector{
      .output_path = options.output_path,
      .separate_text_chunks_with_space =
          (reg.metadata_proto_type_id == "litert.lm.proto.Speech2TextMetadata"),
  };
  const absl::Time run_start = absl::Now();
  ABSL_RETURN_IF_ERROR(RunSessionLoop(*session, collector, out));
  const double run_sec = absl::ToDoubleSeconds(absl::Now() - run_start);

  return FinalizeOutputs(options, reg.metadata_proto_type_id, collector,
                         run_sec, out);
}

}  // namespace litert::omni
