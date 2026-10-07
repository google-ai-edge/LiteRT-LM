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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/ctc_decoder.h"
#include "omni/asr/dummy_preprocessor.h"
#include "omni/asr/levenshtein_text_merger.h"
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/lm_decoder.h"
#include "omni/asr/lm_engine_decoder.h"
#include "omni/asr/log_mel_spectrogram_processor.h"
#include "omni/asr/stateless_decoder.h"
#include "omni/asr/tdt_decoder.h"
#include "omni/asr/text_merger.h"
#include "omni/asr/timestamp_text_merger.h"
#include "omni/asr/tokenizer_detokenizer.h"
#include "omni/base/litert_lm_engine_runner.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/asr_model_type.pb.h"
#include "support/tokenizer/huggingface_tokenizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kDefaultPrompt =
    "Transcribe the following speech segment: ";
constexpr int kDefaultMaxOutputTokens = 128;

bool FileExists(absl::string_view path) {
  std::ifstream f(std::string(path).c_str());
  return f.good();
}

lm::Backend ToLmBackend(OmniEngine::Options::Backend backend) {
  switch (backend) {
    case OmniEngine::Options::Backend::kGpu:
      return lm::Backend::GPU;
    case OmniEngine::Options::Backend::kNpu:
      return lm::Backend::NPU;
    case OmniEngine::Options::Backend::kCpu:
    default:
      return lm::Backend::CPU;
  }
}

void ResolveDecoderTypeFromLitertLm(lm::ModelResources& lm_resources,
                                    AsrEngineConfig& config) {
  auto asr_metadata = lm_resources.GetAsrMetadata();
  if (asr_metadata.ok() && *asr_metadata != nullptr) {
    switch ((*asr_metadata)->asr_model_type().model_type_case()) {
      case lm::proto::AsrModelType::kWhisper:
      case lm::proto::AsrModelType::kMoonshine:
        config.decoder_type = AsrEngineConfig::DecoderType::kStateless;
        return;
      case lm::proto::AsrModelType::kParakeet:
        if (lm_resources
                .GetTFLiteModelBuffer(
                    lm::proto::AsrMetadata::TF_LITE_ENCODER_DECODER)
                .ok()) {
          config.decoder_type = AsrEngineConfig::DecoderType::kTdt;
        } else if (lm_resources
                       .GetTFLiteModelBuffer(
                           lm::proto::AsrMetadata::TF_LITE_AUDIO_ENCODER)
                       .ok()) {
          config.decoder_type = AsrEngineConfig::DecoderType::kCtc;
        }
        return;
      case lm::proto::AsrModelType::kQwen3Asr:
        config.decoder_type = AsrEngineConfig::DecoderType::kLm;
        return;
      case lm::proto::AsrModelType::MODEL_TYPE_NOT_SET:
        break;
    }
  }
  // Legacy LLM-based ASR .litertlm packages (e.g., tinygemma-asr,
  // qwen3-asr-0.6b) may only contain LlmMetadata without an AsrMetadata
  // section, so check for LlmMetadata as a fallback to identify LM decoders.
  auto llm_metadata = lm_resources.GetLlmMetadata();
  if (llm_metadata.ok() && *llm_metadata != nullptr) {
    config.decoder_type = AsrEngineConfig::DecoderType::kLm;
  }
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

  LITERT_ASSIGN_OR_RETURN(auto environment, ::litert::Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(auto options, ::litert::Options::Create());
  uint32_t accelerators = static_cast<uint32_t>(::litert::HwAccelerators::kCpu);
  if (config.num_threads > 0) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                            options.GetOptions<::litert::CpuOptions>());
    cpu_options.SetNumThreads(config.num_threads);
  }
  if (config.backend == OmniEngine::Options::Backend::kGpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kGpu);
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                            options.GetOptions<::litert::GpuOptions>());
    gpu_options.SetPrecision(::litert::GpuOptions::Precision::kFp32);
    gpu_options.EnableConstantTensorSharing(true);
    for (const auto& pattern : config.state_buffer_name_patterns) {
      gpu_options.AddExternalTensorPattern(pattern.c_str());
      gpu_options.AddBufferStorageTensorPattern(pattern.c_str());
    }
  } else if (config.backend == OmniEngine::Options::Backend::kNpu) {
    accelerators |= static_cast<uint32_t>(::litert::HwAccelerators::kNpu);
    LITERT_ASSIGN_OR_RETURN(
        auto& qnn_options,
        options.GetOptions<::litert::qualcomm::QualcommOptions>());
    qnn_options.SetHtpPerformanceMode(::litert::qualcomm::QualcommOptions::
                                          HtpPerformanceMode::kHighPerformance);
  }
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(
      static_cast<::litert::HwAccelerators>(accelerators)));

  auto thread_pool = std::make_unique<::litert::lm::ThreadPool>(
      "asr_engine_pool", config.num_threads);

  std::shared_ptr<lm::ModelResources> lm_resources;
  if (absl::EndsWith(config.model_path, ".litertlm")) {
    ABSL_ASSIGN_OR_RETURN(lm_resources,
                          CreateLmModelResources(config.model_path));
    ResolveDecoderTypeFromLitertLm(*lm_resources, config);
  }
  if (config.decoder_type == AsrEngineConfig::DecoderType::kUnspecified) {
    return absl::InvalidArgumentError(
        "ASR decoder_type is unspecified and could not be resolved from model "
        "metadata.");
  }

  if (config.decoder_type != AsrEngineConfig::DecoderType::kLm) {
    std::unique_ptr<::litert::support::Tokenizer> tokenizer;
    std::unique_ptr<::litert::CompiledModel> compiled_model;

    if (lm_resources != nullptr) {
      ABSL_ASSIGN_OR_RETURN(tokenizer, lm_resources->GetTokenizer());
      auto model_buffer = lm_resources->GetTFLiteModelBuffer(
          lm::proto::AsrMetadata::TF_LITE_ENCODER_DECODER);
      if (!model_buffer.ok()) {
        model_buffer = lm_resources->GetTFLiteModelBuffer(
            lm::proto::AsrMetadata::TF_LITE_AUDIO_ENCODER);
      }
      if (!model_buffer.ok()) {
        return model_buffer.status();
      }
      LITERT_ASSIGN_OR_RETURN(
          auto comp_model,
          ::litert::CompiledModel::Create(
              environment,
              ::litert::BufferRef<uint8_t>(
                  reinterpret_cast<const uint8_t*>(model_buffer->data()),
                  model_buffer->size()),
              options));
      compiled_model =
          std::make_unique<::litert::CompiledModel>(std::move(comp_model));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          tokenizer, ::litert::support::HuggingFaceTokenizer::CreateFromFile(
                         config.tokenizer_path));
      LITERT_ASSIGN_OR_RETURN(auto comp_model,
                              ::litert::CompiledModel::Create(
                                  environment, config.model_path, options));
      compiled_model =
          std::make_unique<::litert::CompiledModel>(std::move(comp_model));
    }

    if (config.vocab_size == 0) {
      config.vocab_size = tokenizer->GetVocabSize();
    }

    return std::unique_ptr<AsrEngine>(new AsrEngine(
        std::move(config), std::move(lm_resources), std::move(tokenizer),
        std::make_unique<::litert::Environment>(std::move(environment)),
        std::move(compiled_model), std::move(thread_pool)));
  }

  ModelOptions lm_options;
  lm_options.backend = ToLmBackend(config.backend);
  lm_options.num_threads = config.num_threads;
  lm_options.cache_dir = config.cache_dir;

  // Models requiring mel normalization (e.g. multimodal speech-language
  // models) use the end-to-end LiteRT-LM Engine which integrates the audio
  // frontend and LLM, whereas other models use the modular executor-based
  // runner.
  bool use_engine = config.log_mel_config.normalize_mel;

  std::unique_ptr<LiteRtLmRunner> lm_runner;
  std::unique_ptr<LiteRtLmEngineRunner> lm_engine_runner;
  std::unique_ptr<::litert::CompiledModel> compiled_model;

  if (use_engine) {
    ABSL_ASSIGN_OR_RETURN(
        lm_engine_runner,
        CreateLmEngineRunner(environment, lm_options, config.model_path));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        lm_runner, CreateLmRunner(environment, lm_options, config.model_path));

    auto* model_resources = lm_runner->mutable_model_resources();
    if (model_resources == nullptr) {
      return absl::InternalError("ModelResources not available in LmRunner.");
    }

    ABSL_ASSIGN_OR_RETURN(auto audio_model_flatbuffer,
                          model_resources->GetTFLiteModelBuffer(
                              lm::ModelType::kTfLiteAudioEncoderHw));
    LITERT_ASSIGN_OR_RETURN(
        auto comp_model,
        ::litert::CompiledModel::Create(
            environment,
            ::litert::BufferRef<uint8_t>(
                reinterpret_cast<const uint8_t*>(audio_model_flatbuffer.data()),
                audio_model_flatbuffer.size()),
            options));
    compiled_model =
        std::make_unique<::litert::CompiledModel>(std::move(comp_model));
  }

  lm::ModelResources* model_resources = nullptr;
  if (lm_engine_runner != nullptr) {
    model_resources = lm_engine_runner->mutable_model_resources();
  } else if (lm_runner != nullptr) {
    model_resources = lm_runner->mutable_model_resources();
  }
  if (model_resources == nullptr) {
    return absl::InternalError("ModelResources not available in runner.");
  }
  ABSL_ASSIGN_OR_RETURN(auto tokenizer, model_resources->GetTokenizer());
  if (config.vocab_size == 0) {
    config.vocab_size = tokenizer->GetVocabSize();
  }

  return std::unique_ptr<AsrEngine>(new AsrEngine(
      std::move(config), std::move(lm_resources), std::move(tokenizer),
      std::make_unique<::litert::Environment>(std::move(environment)),
      std::move(compiled_model), std::move(thread_pool), std::move(lm_runner),
      std::move(lm_engine_runner)));
}

AsrEngine::AsrEngine(
    AsrEngineConfig config,
    std::shared_ptr<::litert::lm::ModelResources> absl_nullable model_resources,
    std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<::litert::Environment> absl_nonnull environment,
    std::unique_ptr<::litert::CompiledModel> absl_nullable compiled_model,
    std::unique_ptr<::litert::lm::ThreadPool> absl_nonnull thread_pool,
    std::unique_ptr<LiteRtLmRunner> lm_runner,
    std::unique_ptr<LiteRtLmEngineRunner> lm_engine_runner)
    : config_(std::move(config)),
      model_resources_(std::move(model_resources)),
      tokenizer_(std::move(tokenizer)),
      environment_(std::move(environment)),
      compiled_model_(std::move(compiled_model)),
      thread_pool_(std::move(thread_pool)),
      lm_runner_(std::move(lm_runner)),
      lm_engine_runner_(std::move(lm_engine_runner)) {}

absl::StatusOr<std::unique_ptr<MultiStagedSession>> AsrEngine::CreateSession(
    std::unique_ptr<AudioSource> absl_nonnull audio_source) {
  std::unique_ptr<LiteRtRunner> runner;
  if (compiled_model_ != nullptr) {
    runner = std::make_unique<LiteRtRunnerImpl>(compiled_model_.get());
  } else {
    const size_t buffer_size_bytes = config_.log_mel_config.n_frames *
                                     config_.log_mel_config.n_mels *
                                     sizeof(float);
    runner = std::make_unique<PassthroughRunner>(
        std::vector<size_t>{buffer_size_bytes});
  }

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
          StatelessDecoder::Create(
              runner.get(), config_.decode_start_token_id,
              config_.decode_stop_token_id,
              config_.decode_skip_until_token_id));
      break;
    }
    case AsrEngineConfig::DecoderType::kLm: {
      if (lm_engine_runner_ != nullptr) {
        ABSL_ASSIGN_OR_RETURN(
            decoder, LmEngineDecoder::Create(
                         lm_engine_runner_.get(), std::string(kDefaultPrompt),
                         kDefaultMaxOutputTokens, config_.decode_start_token_id,
                         config_.decode_stop_token_id,
                         config_.decode_skip_until_token_id));
      } else if (lm_runner_ != nullptr) {
        ABSL_ASSIGN_OR_RETURN(
            decoder,
            LmDecoder::Create(lm_runner_.get(), config_.decode_start_token_id,
                              config_.decode_stop_token_id,
                              config_.decode_skip_until_token_id));
      } else {
        return absl::InternalError(
            "LmDecoder requested but neither lm_runner_ nor "
            "lm_engine_runner_ is available.");
      }
      break;
    }
    case AsrEngineConfig::DecoderType::kUnspecified:
      return absl::InvalidArgumentError(
          "ASR decoder_type must be specified before creating a session.");
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

  TextMerger* raw_text_merger = text_merger.get();
  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.reserve(5);
  stages.push_back(std::move(audio_source));
  stages.push_back(std::move(preprocessor));
  stages.push_back(std::move(speech_recognizer));
  stages.push_back(std::move(detokenizer));
  stages.push_back(std::move(text_merger));

  return MultiStagedSession::Create(std::move(stages), raw_text_merger,
                                    thread_pool_.get());
}

}  // namespace litert::omni::asr
