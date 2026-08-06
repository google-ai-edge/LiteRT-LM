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

#include "omni/asr/asr_engine.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/ctc_decoder.h"
#include "omni/asr/dummy_preprocessor.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/log_mel_spectrogram_processor.h"
#include "omni/asr/stateless_decoder.h"
#include "omni/asr/tdt_decoder.h"
#include "omni/asr/text_merger.h"
#include "omni/asr/timestamp_text_merger.h"
#include "omni/asr/tokenizer_detokenizer.h"
#include "omni/base/litert_runner.h"
#include "support/tokenizer/huggingface_tokenizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

bool FileExists(absl::string_view path) {
  std::ifstream f(std::string(path).c_str());
  return f.good();
}

}  // namespace

absl::Status AsrEngine::EnsureFilesDownloaded(
    AsrEngineConfig& config, const FileDownloader& downloader) {
  if (!config.model_url.empty() && !FileExists(config.model_path)) {
    if (!downloader) {
      return absl::NotFoundError(absl::StrCat("Model file missing at ",
                                              config.model_path,
                                              " and no downloader provided."));
    }
    ABSL_RETURN_IF_ERROR(downloader(config.model_url, config.model_path));
  }
  if (!config.tokenizer_url.empty() && !FileExists(config.tokenizer_path)) {
    if (!downloader) {
      return absl::NotFoundError(absl::StrCat("Tokenizer file missing at ",
                                              config.tokenizer_path,
                                              " and no downloader provided."));
    }
    ABSL_RETURN_IF_ERROR(
        downloader(config.tokenizer_url, config.tokenizer_path));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AsrEngine>> AsrEngine::Create(
    AsrEngineConfig config, FileDownloader downloader) {
  ABSL_RETURN_IF_ERROR(EnsureFilesDownloaded(config, downloader));

  ABSL_ASSIGN_OR_RETURN(auto tokenizer,
                        ::litert::support::HuggingFaceTokenizer::CreateFromFile(
                            config.tokenizer_path));
  if (config.vocab_size == 0) {
    config.vocab_size = tokenizer->GetVocabSize();
  }

  LITERT_ASSIGN_OR_RETURN(auto environment, ::litert::Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  uint32_t accelerators = static_cast<uint32_t>(::litert::HwAccelerators::kCpu);
  if (config.num_threads > 0) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    cpu_options.SetNumThreads(config.num_threads);
  }
  if (config.backend == AsrEngineConfig::Backend::kGpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kGpu);
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    gpu_options.SetPrecision(::litert::GpuOptions::Precision::kFp32);
    gpu_options.EnableConstantTensorSharing(true);
  } else if (config.backend == AsrEngineConfig::Backend::kNpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kNpu);
    LITERT_ASSIGN_OR_RETURN(auto& qnn_options, options.GetQualcommOptions());
    qnn_options.SetHtpPerformanceMode(::litert::qualcomm::QualcommOptions::
                                          HtpPerformanceMode::kHighPerformance);
  }
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(
      static_cast<::litert::HwAccelerators>(accelerators)));

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      ::litert::CompiledModel::Create(environment, config.model_path, options));

  return std::unique_ptr<AsrEngine>(new AsrEngine(
      std::move(config), std::move(tokenizer),
      std::make_unique<::litert::Environment>(std::move(environment)),
      std::make_unique<::litert::CompiledModel>(std::move(compiled_model))));
}

AsrEngine::AsrEngine(AsrEngineConfig config,
                     std::unique_ptr<::litert::support::Tokenizer> tokenizer,
                     std::unique_ptr<::litert::Environment> environment,
                     std::unique_ptr<::litert::CompiledModel> compiled_model)
    : config_(std::move(config)),
      tokenizer_(std::move(tokenizer)),
      environment_(std::move(environment)),
      compiled_model_(std::move(compiled_model)) {}

absl::StatusOr<std::unique_ptr<AsrSession>> AsrEngine::CreateSession(
    std::unique_ptr<AudioSource> audio_source) {
  auto runner = std::make_unique<LiteRtRunnerImpl>(compiled_model_.get());

  AudioSource* raw_audio_source = audio_source.get();
  std::unique_ptr<AudioPreprocessor> preprocessor;
  if (config_.has_log_mel_config) {
    ABSL_ASSIGN_OR_RETURN(
        preprocessor,
        LogMelSpectrogramProcessor::Create(
            config_.sample_rate_hz, config_.log_mel_config, raw_audio_source));
  } else {
    preprocessor = std::make_unique<DummyPreprocessor>(raw_audio_source);
  }

  std::unique_ptr<LiteRtSpeechRecognizer::Decoder> decoder;
  switch (config_.decoder_type) {
    case AsrEngineConfig::DecoderType::kTdt: {
      ABSL_ASSIGN_OR_RETURN(
          decoder,
          TdtDecoder::Create(runner.get(), config_.decode_start_token_id,
                             config_.decode_statefully_after));
      break;
    }
    case AsrEngineConfig::DecoderType::kCtc: {
      ABSL_ASSIGN_OR_RETURN(decoder, CtcDecoder::Create(config_.vocab_size));
      break;
    }
    case AsrEngineConfig::DecoderType::kStateless: {
      ABSL_ASSIGN_OR_RETURN(
          decoder,
          StatelessDecoder::Create(runner.get(), config_.decode_start_token_id,
                                   config_.decode_stop_token_id,
                                   config_.decode_skip_until_token_id));
      break;
    }
  }

  AudioPreprocessor* raw_preprocessor = preprocessor.get();
  ABSL_ASSIGN_OR_RETURN(
      auto speech_recognizer,
      LiteRtSpeechRecognizer::Create(std::move(runner), raw_preprocessor,
                                     std::move(decoder)));

  LiteRtSpeechRecognizer* raw_speech_recognizer = speech_recognizer.get();
  auto detokenizer = std::make_unique<TokenizerDetokenizer>(
      raw_speech_recognizer, tokenizer_.get());

  TokenizerDetokenizer* raw_detokenizer = detokenizer.get();
  std::unique_ptr<TextMerger> text_merger;
  switch (config_.text_merger_type) {
    case AsrEngineConfig::TextMergerType::kLevenshtein:
      text_merger = std::make_unique<LevenshteinTextMerger>(raw_detokenizer);
      break;
    case AsrEngineConfig::TextMergerType::kTimestamp:
      text_merger = std::make_unique<TimestampTextMerger>(
          raw_detokenizer, config_.overlap_ratio);
      break;
  }

  AsrSession::Components components;
  components.audio_source = std::move(audio_source);
  components.preprocessor = std::move(preprocessor);
  components.speech_recognizer = std::move(speech_recognizer);
  components.detokenizer = std::move(detokenizer);
  components.text_merger = std::move(text_merger);

  return AsrSession::Create(std::move(components));
}

}  // namespace litert::omni::asr
