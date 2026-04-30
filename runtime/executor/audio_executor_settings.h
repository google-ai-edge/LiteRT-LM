// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_AUDIO_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_AUDIO_EXECUTOR_SETTINGS_H_

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

class AudioExecutorSettings : public ExecutorSettingsBase {
 public:
  static absl::StatusOr<AudioExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, int max_sequence_length, Backend backend,
      bool bundled_with_main_model = true);

  // Suffix constants
  static constexpr absl::string_view kStaticEncoderName =
      ".static_audio_encoder";
  static constexpr absl::string_view kStreamingEncoderName =
      ".streaming_audio_encoder";
  static constexpr absl::string_view kAdapterName = ".audio_adapter";
  static constexpr absl::string_view kEncoderName = ".audio_encoder";

  // Getter for max_sequence_length.
  int GetMaxSequenceLength() const;
  // Setter for max_sequence_length.
  void SetMaxSequenceLength(int max_sequence_length);

  // Getter for bundled_with_main_model.
  bool GetBundledWithMainModel() const;
  // Setter for bundled_with_main_model.
  void SetBundledWithMainModel(bool bundled_with_main_model);

  absl::Status SetBackend(const Backend& backend) override;

  // Getter for num_threads for CPU backend.
  int GetNumThreads() const { return num_threads_; }
  // Setter for num_threads for CPU backend.
  void SetNumThreads(int num_threads) { num_threads_ = num_threads; }

  // Getter for scoped_encoder_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedEncoderCacheFile() const {
    return scoped_encoder_cache_file_;
  }

  // Setter for scoped_encoder_cache_file.
  void SetScopedEncoderCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_encoder_cache_file_ = std::move(cache_file);
  }

  // Getter for scoped_adapter_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedAdapterCacheFile() const {
    return scoped_adapter_cache_file_;
  }

  // Setter for scoped_adapter_cache_file.
  void SetScopedAdapterCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_adapter_cache_file_ = std::move(cache_file);
  }

  // Getter for scoped_encoder_program_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedEncoderProgramCacheFile()
      const {
    return scoped_encoder_program_cache_file_;
  }

  // Setter for scoped_encoder_program_cache_file.
  void SetScopedEncoderProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_encoder_program_cache_file_ = std::move(cache_file);
  }

  // Getter for scoped_adapter_program_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedAdapterProgramCacheFile()
      const {
    return scoped_adapter_program_cache_file_;
  }

  // Setter for scoped_adapter_program_cache_file.
  void SetScopedAdapterProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    scoped_adapter_program_cache_file_ = std::move(cache_file);
  }

  // Returns the weight cache file path for the audio encoder or adapter
  // model.
  // Note users should not use the ExecutorSettingsBase::GetWeightCacheFile()
  // method to get the weight cache file for the audio encoder or adapter
  // model, because the base class does not distinguish between the two
  // models.
  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetWeightCacheFile(absl::string_view suffix,
                     bool check_and_clean = false) const;

  absl::StatusOr<
      std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
  GetProgramCacheFile(absl::string_view suffix,
                      bool check_and_clean = false) const;

 private:
  explicit AudioExecutorSettings(const ModelAssets& model_assets,
                                 int max_sequence_length, int num_threads)
      : ExecutorSettingsBase(model_assets),
        max_sequence_length_(max_sequence_length),
        num_threads_(num_threads) {}

  int max_sequence_length_;
  bool bundled_with_main_model_;
  int num_threads_ = 4;

  // The cache file to use for the audio encoder model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_encoder_cache_file_;

  // The cache file to use for the audio adapter model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_adapter_cache_file_;

  // The program cache file to use for the audio encoder model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_encoder_program_cache_file_;

  // The program cache file to use for the audio adapter model.
  std::shared_ptr<litert::lm::ScopedFile> scoped_adapter_program_cache_file_;
};

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_AUDIO_EXECUTOR_SETTINGS_H
