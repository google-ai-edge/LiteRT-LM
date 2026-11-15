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

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_api_types.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_custom_op.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_utils.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/speech2text/asr_session.h"
#include "omni/speech2text/audio_source.h"
#include "omni/speech2text/litert_speech_recognizer.h"
#include "omni/speech2text/speech_recognizer.h"
#include "omni/speech2text/tdt_decoder.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/speech2text_model_type.pb.h"
#include "support/tokenizer/tokenizer.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert

namespace litert::omni::asr {
namespace {

constexpr absl::string_view kModelNames[] = {
    "parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v3-int8",
    "parakeet-tdt",
    "parakeet-tdt-int8",
};

// Whether to run the Parakeet TDT decoder (`decode` / `decode_1`) on CPU when
// the backend is GPU.
constexpr bool kDecodeOnCpu = true;

// Restricts `options` to compile only the specified model signatures (e.g.
// only `"encode"` for the GPU encoder or `"decode"` / `"decode_1"` for the CPU
// decoder) so unused signatures are not delegated or compiled.
absl::Status SetSelectedSignatures(
    ::litert::Options& options,
    absl::Span<const absl::string_view> selected_signatures) {
  LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                          options.GetOptions<::litert::RuntimeOptions>());
  std::vector<::litert::StringView> keys;
  keys.reserve(selected_signatures.size());
  for (absl::string_view sig : selected_signatures) {
    keys.emplace_back(sig.data(), sig.size());
  }
  LITERT_RETURN_IF_ERROR(
      runtime_options.SetSelectedSignatures(absl::MakeConstSpan(keys)));
  return absl::OkStatus();
}

// Returns true when `status` indicates that `signature_name` is not present in
// the model (`kNotFound` or `CompiledModel`'s `"Unknown selected signature
// key"` `kInvalidArgument`).
bool IsMissingSignatureError(const absl::Status& status,
                             absl::string_view signature_name) {
  return absl::IsNotFound(status) ||
         (absl::IsInvalidArgument(status) &&
          absl::StrContains(status.message(), signature_name));
}

// Compiles a dedicated CPU `CompiledModel` for the TDT decoder signatures
// (`selected_signatures`) with its own `<model_name>_decoder` cache key and
// custom bidirectional LSTM op registration.
absl::StatusOr<std::unique_ptr<::litert::CompiledModel>>
CreateCpuDecoderCompiledModel(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    ::litert::Environment& env, lm::ModelResources* absl_nullable lm_resources,
    absl::Span<const absl::string_view> selected_signatures) {
  LITERT_ASSIGN_OR_RETURN(auto decoder_options, ::litert::Options::Create());
  LITERT_RETURN_IF_ERROR(
      decoder_options.SetHardwareAccelerators(::litert::HwAccelerators::kCpu));
  const std::string decoder_cache_name = absl::StrCat(model_name, "_decoder");
  ABSL_ASSIGN_OR_RETURN(
      const ModelCacheConfig cache,
      ResolveModelCache(decoder_cache_name, settings.model_path,
                        settings.cache_dir));
  LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                          decoder_options.GetOptions<::litert::CpuOptions>());
  if (settings.num_threads > 0) {
    LITERT_RETURN_IF_ERROR(cpu_options.SetNumThreads(settings.num_threads));
  }
  auto set_flags = cpu_options.GetXNNPackFlags();
  uint32_t xnn_flags = (set_flags.HasValue() && *set_flags != 0)
                           ? *set_flags
                           : TfLiteXNNPackDelegateOptionsDefault().flags;
  xnn_flags |= TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED |
               TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  LITERT_RETURN_IF_ERROR(cpu_options.SetXNNPackFlags(xnn_flags));
  if (!cache.key.empty()) {
    ABSL_RETURN_IF_ERROR(
        ConfigureCpuModelCache(decoder_cache_name, cache, cpu_options));
  }
  LITERT_RETURN_IF_ERROR(
      litert::custom_ops::RegisterBidirectionalLstmCustomOp(decoder_options));
  if (!selected_signatures.empty()) {
    ABSL_RETURN_IF_ERROR(
        SetSelectedSignatures(decoder_options, selected_signatures));
  }
  OmniEngine::Settings cpu_settings = settings;
  cpu_settings.backend = OmniEngine::Settings::Backend::kCpu;
  return CreateCompiledModel(
      cpu_settings, env, decoder_options, lm_resources,
      lm::proto::Speech2TextMetadata::TF_LITE_ENCODER_DECODER);
}

// Adapter that co-owns `decoder_runner_` alongside `decoder_` so `decoder_`
// (which holds a reference to `decoder_runner_`) is destroyed before
// `decoder_runner_`.
class DecoderWithOwnedRunner : public LiteRtSpeechRecognizer::Decoder {
 public:
  DecoderWithOwnedRunner(
      std::unique_ptr<LiteRtRunner> absl_nonnull decoder_runner,
      std::unique_ptr<LiteRtSpeechRecognizer::Decoder> absl_nonnull decoder)
      : decoder_runner_(std::move(decoder_runner)),
        decoder_(std::move(decoder)) {}

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs,
      std::optional<DecodeRange> decode_range) override {
    return decoder_->Decode(encoder_outputs, decode_range);
  }

 private:
  std::unique_ptr<LiteRtRunner> absl_nonnull decoder_runner_;
  std::unique_ptr<LiteRtSpeechRecognizer::Decoder> absl_nonnull decoder_;
};

// Resolves configs and instantiates encoder, decoder, and sessions on demand.
class ParakeetTdtSessionFactory : public AsrSessionFactory {
 public:
  static lm::proto::Speech2TextMetadata ResolveMetadata(
      const MessageLite* absl_nullable metadata) {
    lm::proto::Speech2TextMetadata resolved;
    if (metadata != nullptr) {
      resolved.CheckTypeAndMergeFrom(*metadata);
    }
    if (!resolved.has_asr_model_type()) {
      resolved.mutable_asr_model_type()->mutable_parakeet();
    }
    if (resolved.decode_start_token_ids_size() == 0) {
      resolved.add_decode_start_token_ids(8192);
    }
    auto* mel = resolved.mutable_log_mel_spectrogram_config();
    if (!mel->has_n_fft()) mel->set_n_fft(512);
    if (!mel->has_n_mels()) mel->set_n_mels(128);
    if (!mel->has_n_frames()) mel->set_n_frames(500);
    if (!mel->has_preemphasis()) mel->set_preemphasis(0.97f);
    if (!mel->has_slaney_mel()) mel->set_slaney_mel(true);
    if (!mel->has_power_spectrum()) mel->set_power_spectrum(true);
    if (!mel->has_mel_floor()) mel->set_mel_floor(5.960464477539063e-08f);
    return resolved;
  }

  ParakeetTdtSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const MessageLite* absl_nullable metadata, ::litert::Environment env,
      std::shared_ptr<lm::ModelResources> absl_nullable lm_resources,
      std::unique_ptr<::litert::support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model,
      std::unique_ptr<::litert::CompiledModel> absl_nullable
      decoder_compiled_model = nullptr)
      : AsrSessionFactory(settings, session_config, metadata, std::move(env),
                          std::move(lm_resources), std::move(tokenizer)),
        compiled_model_(std::move(compiled_model)),
        decoder_compiled_model_(std::move(decoder_compiled_model)) {}

 protected:
  bool ShouldEnableWarmStart() const override { return true; }

  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithAudioSource(
      std::unique_ptr<AudioSource> absl_nonnull audio_source) override {
    auto encoder = std::make_unique<LiteRtRunnerImpl>(compiled_model_.get());
    std::unique_ptr<LiteRtRunner> decoder_runner;
    LiteRtRunner* decoder_runner_ptr = encoder.get();
    if (decoder_compiled_model_ != nullptr) {
      decoder_runner =
          std::make_unique<LiteRtRunnerImpl>(decoder_compiled_model_.get());
      decoder_runner_ptr = decoder_runner.get();
    }
    std::vector<int> start_token_ids(
        metadata().decode_start_token_ids().begin(),
        metadata().decode_start_token_ids().end());
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<LiteRtSpeechRecognizer::Decoder> decoder,
        TdtDecoder::Create(decoder_runner_ptr, std::move(start_token_ids),
                           metadata().decode_statefully_after()));
    if (decoder_runner != nullptr) {
      decoder = std::make_unique<DecoderWithOwnedRunner>(
          std::move(decoder_runner), std::move(decoder));
    }
    return CreateMultiStagedSession(std::move(audio_source), std::move(encoder),
                                    std::move(decoder));
  }

 private:
  std::unique_ptr<::litert::CompiledModel> absl_nonnull compiled_model_;
  std::unique_ptr<::litert::CompiledModel> absl_nullable
  decoder_compiled_model_;
};

// Instantiates common objects shared among sessions, then returns the session
// factory.
absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
GetParakeetTdtSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata,
    ::litert::Environment& env, ::litert::Options& options,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "lm_resources must not be null (model_path: ", settings.model_path,
        ")."));
  }
  std::unique_ptr<::litert::CompiledModel> decoder_compiled_model;
  if (settings.backend == OmniEngine::Settings::Backend::kGpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                            options.GetOptions<::litert::GpuOptions>());
    if (kDecodeOnCpu) {
      // The TDT decoder runs small autoregressive single-frame LSTM steps
      // (`decode` and `decode_1`) many times per chunk, where per-step GPU
      // dispatch and synchronization overhead is much higher than CPU
      // execution. Restrict the GPU `CompiledModel` to `"encode"` only (so the
      // GPU delegate does not compile or allocate buffers for `"decode"` /
      // `"decode_1"`) and compile the decoder signatures on CPU.
      ABSL_RETURN_IF_ERROR(SetSelectedSignatures(options, {"encode"}));
      gpu_options.AddExternalTensorPattern("encode_output_0_output");
      gpu_options.AddBufferStorageTensorPattern("encode_output_0_output");
      auto decoder_model = CreateCpuDecoderCompiledModel(
          model_name, settings, env, lm_resources.get(),
          {"decode", "decode_1"});
      if (!decoder_model.ok()) {
        if (IsMissingSignatureError(decoder_model.status(), "decode_1")) {
          decoder_model = CreateCpuDecoderCompiledModel(
              model_name, settings, env, lm_resources.get(), {"decode"});
        } else {
          ABSL_LOG(WARNING)
              << "Failed to compile CPU decoder for " << model_name
              << " with stateful signature: " << decoder_model.status();
        }
      }
      ABSL_ASSIGN_OR_RETURN(decoder_compiled_model, std::move(decoder_model));
    } else {
      constexpr absl::string_view kStateBufferPatterns[] = {
          "encode_output_0_output",   "decode_args_0",
          "decode_1_args_0",          "decode_1_args_2",
          "decode_1_args_3",          "decode_1_output_1_output",
          "decode_1_output_2_output", "decode_output_1_output",
          "decode_output_2_output",
      };
      for (absl::string_view pattern : kStateBufferPatterns) {
        gpu_options.AddExternalTensorPattern(pattern.data());
        gpu_options.AddBufferStorageTensorPattern(pattern.data());
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      auto compiled_model,
      CreateCompiledModel(
          settings, env, options, lm_resources.get(),
          lm::proto::Speech2TextMetadata::TF_LITE_ENCODER_DECODER));
  ABSL_ASSIGN_OR_RETURN(
      auto tokenizer,
      CreateTokenizer(lm_resources.get(), settings.tokenizer_path));
  lm::proto::Speech2TextMetadata resolved_metadata =
      ParakeetTdtSessionFactory::ResolveMetadata(metadata);
  return std::make_unique<ParakeetTdtSessionFactory>(
      settings, session_config, &resolved_metadata, std::move(env),
      std::move(lm_resources), std::move(tokenizer), std::move(compiled_model),
      std::move(decoder_compiled_model));
}

[[maybe_unused]] const bool kParakeetTdtRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetParakeetTdtSessionFactory,
      .session_config_type_id = "Speech2Text",
      .model_remote_url =
          "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/"
          "resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::asr
