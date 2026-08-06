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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_ENGINE_H_

#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/asr/asr_session.h"
#include "omni/asr/audio_source.h"
#include "omni/asr/log_mel_spectrogram_processor.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {

struct AsrEngineConfig {
  enum class DecoderType {
    kCtc = 0,
    kTdt = 1,
    kStateless = 2,
  };

  enum class Backend {
    kCpu = 0,
    kGpu = 1,
    kNpu = 2,
  };

  enum class TextMergerType {
    kLevenshtein = 0,
    kTimestamp = 1,
  };

  std::string model_name;
  std::string model_path;
  std::string tokenizer_path;
  std::string model_url;
  std::string tokenizer_url;

  int sample_rate_hz = 16000;
  int input_milliseconds = 5000;

  DecoderType decoder_type = DecoderType::kTdt;
  Backend backend = Backend::kCpu;
  TextMergerType text_merger_type = TextMergerType::kTimestamp;
  int num_threads = 4;
  float overlap_ratio = 0.4f;

  bool has_log_mel_config = false;
  LogMelSpectrogramProcessor::LogMelSpectrogramConfig log_mel_config;

  int decode_start_token_id = -1;
  int decode_stop_token_id = -1;
  int decode_skip_until_token_id = -1;
  int decode_statefully_after = 4;
  int vocab_size = 0;
  int blank_token_id = 0;

  std::string cache_dir = "/tmp/asr_models";
};

// Engine that manages model loading, tokenizer instantiation, and ASR session
// creation.
class AsrEngine {
 public:
  using FileDownloader = absl::AnyInvocable<absl::Status(
      absl::string_view url, absl::string_view target_path) const>;

  // Creates an AsrEngine. Downloads missing model/tokenizer files using
  // downloader callback.
  static absl::StatusOr<std::unique_ptr<AsrEngine>> Create(
      AsrEngineConfig config, FileDownloader downloader = nullptr);

  ~AsrEngine() = default;

  // Instantiates components based on configuration and returns a new
  // AsrSession.
  absl::StatusOr<std::unique_ptr<AsrSession>> CreateSession(
      std::unique_ptr<AudioSource> audio_source);

  const AsrEngineConfig& config() const { return config_; }

 private:
  AsrEngine(AsrEngineConfig config,
            std::unique_ptr<::litert::support::Tokenizer> tokenizer,
            std::unique_ptr<::litert::Environment> environment,
            std::unique_ptr<::litert::CompiledModel> compiled_model);

  static absl::Status EnsureFilesDownloaded(AsrEngineConfig& config,
                                            const FileDownloader& downloader);

  AsrEngineConfig config_;
  std::unique_ptr<::litert::support::Tokenizer> tokenizer_;
  std::unique_ptr<::litert::Environment> environment_;
  std::unique_ptr<::litert::CompiledModel> compiled_model_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_ASR_ENGINE_H_
