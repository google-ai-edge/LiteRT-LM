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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_SETTINGS_H_

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Load-time configuration for exposing per-token text encoder outputs. When
// enabled and any active encoder signature lacks a requested output, the
// encoder FlatBuffer is rewritten in memory while keeping output 0
// (`encodings`) untouched and bit-identical.
struct PerTokenOutputsConfig {
  // Expose `token_embeddings` (`[batch, seq_len, embedding_dim]`) via the
  // pooling projection applied before masked mean pooling.
  bool token_embeddings = false;

  // 0-based layer indices exposed as `layer_<NN>` (`[batch, seq_len, dim]`),
  // tapped after each layer's residual ADD and skip-scale MUL.
  std::vector<int> hidden_state_layers;

  bool IsEnabled() const {
    return token_embeddings || !hidden_state_layers.empty();
  }

  bool operator==(const PerTokenOutputsConfig& other) const = default;
};

// Settings for configuring an embedding executor.
class EmbeddingExecutorSettings : public ExecutorSettingsBase {
 public:
  static constexpr absl::string_view kEncoderName =
      GetCacheModelName(CacheModel::kTextEncoder);

  // Creates a default configuration for the embedding executor.
  //
  // Args:
  //   model_assets: The model assets to be used by the executor.
  //   backend: The execution backend to target.
  //
  // Returns:
  //   An EmbeddingExecutorSettings instance with default settings for the
  //   specified backend, or an error status if the default settings could not
  //   be created.
  static absl::StatusOr<EmbeddingExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, Backend backend);

  // Constructs an EmbeddingExecutorSettings with the given model assets.
  //
  // Args:
  //   model_assets: The model assets to be used by the executor.
  explicit EmbeddingExecutorSettings(const ModelAssets& model_assets)
      : ExecutorSettingsBase(model_assets) {}

  // Getter for num_threads for CPU backend.
  int GetNumThreads() const { return num_threads_; }
  // Setter for num_threads for CPU backend.
  void SetNumThreads(int num_threads) { num_threads_ = num_threads; }

  // Getters for the NPU options used when compiling the model for the NPU
  // backend.
  const NpuOptions& GetNpuOptions() const { return npu_options_; }
  NpuOptions& GetMutableNpuOptions() { return npu_options_; }
  // Setter for the NPU options used when compiling the model for the NPU
  // backend. Ignored by non-NPU backends.
  void SetNpuOptions(NpuOptions npu_options) {
    npu_options_ = std::move(npu_options);
  }

  // Getter for scoped_encoder_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedEncoderCacheFile() const {
    return GetScopedCacheFile(CacheModel::kTextEncoder, CacheFileKind::kWeight);
  }

  // Setter for scoped_encoder_cache_file.
  void SetScopedEncoderCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    SetScopedCacheFile(CacheModel::kTextEncoder, CacheFileKind::kWeight,
                       std::move(cache_file));
  }

  // Getter for scoped_encoder_program_cache_file.
  std::shared_ptr<litert::lm::ScopedFile> GetScopedEncoderProgramCacheFile()
      const {
    return GetScopedCacheFile(CacheModel::kTextEncoder,
                              CacheFileKind::kProgram);
  }

  // Setter for scoped_encoder_program_cache_file.
  void SetScopedEncoderProgramCacheFile(
      std::shared_ptr<litert::lm::ScopedFile> cache_file) {
    SetScopedCacheFile(CacheModel::kTextEncoder, CacheFileKind::kProgram,
                       std::move(cache_file));
  }

  // Selected signatures APIs.
  const std::vector<std::string>& GetSelectedSignatures() const {
    return selected_signatures_;
  }
  void SetSelectedSignatures(std::vector<std::string> selected_signatures) {
    selected_signatures_ = std::move(selected_signatures);
  }

  // Optional load-time injection of per-token encoder outputs.
  const PerTokenOutputsConfig& GetPerTokenOutputs() const {
    return per_token_outputs_;
  }
  PerTokenOutputsConfig& GetMutablePerTokenOutputs() {
    return per_token_outputs_;
  }
  void SetPerTokenOutputs(PerTokenOutputsConfig per_token_outputs) {
    per_token_outputs_ = std::move(per_token_outputs);
  }

 private:
  int num_threads_ = 4;

  // The NPU options for the text encoder model.
  NpuOptions npu_options_;

  std::vector<std::string> selected_signatures_;

  PerTokenOutputsConfig per_token_outputs_;
};

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingExecutorSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_SETTINGS_H_
