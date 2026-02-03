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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "litert/cc/internal/scoped_file.h"  // from @litert

namespace litert::lm {

// Note for development conventions:
// 1. Any optional field should use std::optional.
// 2. All member variables should be private and have their corresponding
// getters and setters.
// 3. For basic types, e.g. int, float, bool, etc., the getters and setters
// should be Get*() and Set*().
// 4. For complex types, e.g. proto::BenchmarkParams, the getters and setters
// should be Get*() and GetMutable*().
// 5. For optional fields, the mutable getter should create a default instance
// if the field is not set. But the non-mutable getter should return a
// const reference to the std::optional<T> field.

// Settings used for initializing LiteRT LM Engine.
// This class encapsulates the model-specific settings that are used for
// initializing the LiteRT LM. These settings are typically fixed for a given
// model and are not expected to change during the inference process.
//
// This class is used to initialize the LiteRT LM Engine. The user should
// create an EngineSettings object and then call the MaybeUpdateAndValidate()
// method to validate the settings. If the validation fails, the user should
// not use the EngineSettings object.
//
// Example:
//
//   ASSIGN_OR_RETURN(ModelAssets model_assets,
//                    ModelAssets::Create(model_path));
//   ASSIGN_OR_RETURN(EngineSettings engine_settings,
//                    EngineSettings::CreateDefault(model_assets));
//    ...initialize the Engine...
//   ASSIGN_OR_RETURN(std::unique_ptr<Engine> engine,
//                    Engine::CreateEngine(engine_settings));
// TODO(b/397975034) Add overloading << operator for debugging.
class EngineSettings {
 public:
  // Creates a default EngineSettings with the given model assets and specified
  // backend.
  static absl::StatusOr<EngineSettings> CreateDefault(
      ModelAssets model_assets, Backend backend = Backend::CPU,
      std::optional<Backend> vision_backend = std::nullopt,
      std::optional<Backend> audio_backend = std::nullopt);

  // Updates the EngineSettings fields by loading the metadata from the model
  // assets. The function also validates to check if all of the required fields
  // are set correctly. Returns an error if the validation fails.
  absl::Status MaybeUpdateAndValidate(
      Tokenizer& tokenizer,
      const proto::LlmMetadata* absl_nullable metadata_from_file,
      absl::string_view input_prompt_as_hint = "",
      const std::optional<std::string>& text_backend_constraint = std::nullopt,
      const std::optional<std::string>& vision_backend_constraint =
          std::nullopt,
      const std::optional<std::string>& audio_backend_constraint =
          std::nullopt);

  // Returns the LlmExecutorSettings.
  const LlmExecutorSettings& GetMainExecutorSettings() const;
  // Returns the mutable LlmExecutorSettings.
  LlmExecutorSettings& GetMutableMainExecutorSettings();

  // Returns the VisionExecutorSettings for the vision model.
  const std::optional<VisionExecutorSettings>& GetVisionExecutorSettings()
      const;
  // Returns the mutable VisionExecutorSettings for the vision model.
  std::optional<VisionExecutorSettings>& GetMutableVisionExecutorSettings();

  // Returns the AudioExecutorSettings for the audio model.
  const std::optional<AudioExecutorSettings>& GetAudioExecutorSettings() const;
  // Returns the mutable AudioExecutorSettings for the audio model.
  std::optional<AudioExecutorSettings>& GetMutableAudioExecutorSettings();

  // Benchmark parameters:
  // Returns true if the benchmark is enabled.
  bool IsBenchmarkEnabled() const;
  // Returns the benchmark parameters.
  const std::optional<proto::BenchmarkParams>& GetBenchmarkParams() const;
  // Returns the mutable benchmark parameters.
  proto::BenchmarkParams& GetMutableBenchmarkParams();

  // Returns the LlmMetadata parameters.
  const std::optional<proto::LlmMetadata>& GetLlmMetadata() const;
  // Returns the mutable LlmMetadata parameters. Note that is the metadata_ is
  // not set (i.e. std::nullopt), then the default LlmMetadata will be
  // created and returned.
  proto::LlmMetadata& GetMutableLlmMetadata();

 private:
  explicit EngineSettings(
      LlmExecutorSettings executor_settings,
      std::optional<VisionExecutorSettings> vision_executor_settings,
      std::optional<AudioExecutorSettings> audio_executor_settings,
      std::optional<proto::BenchmarkParams> benchmark_params = std::nullopt);

  // Settings for the main executor.
  LlmExecutorSettings main_executor_settings_;

  // Settings for the vision executor.
  std::optional<VisionExecutorSettings> vision_executor_settings_;

  // Settings for the audio executor.
  std::optional<AudioExecutorSettings> audio_executor_settings_;

  // Parameters used to configure the benchmarking process.
  std::optional<proto::BenchmarkParams> benchmark_params_;

  // Default metadata for the model. This is loaded from the model assets (if
  // present).
  std::optional<proto::LlmMetadata> metadata_;
};
std::ostream& operator<<(std::ostream& os, const EngineSettings& settings);

// Configurations used for the session.
// This class encapsulates the session-specific configurations that are used for
// creating a LiteRT LM session.
class SessionConfig {
 public:
  // Creates a default SessionConfig.
  static SessionConfig CreateDefault();

  // Updates the SessionConfig fields from the EngineSettings when not set. The
  // function also validates to check if all of the required fields are set
  // correctly. Returns an error if the validation fails.
  absl::Status MaybeUpdateAndValidate(const EngineSettings& engine_settings);

  // Configures the audio modality in the session.
  bool AudioModalityEnabled() const { return audio_modality_enabled_; }
  void SetAudioModalityEnabled(bool enable_audio_modality) {
    audio_modality_enabled_ = enable_audio_modality;
  }

  // Configures the vision modality in the session.
  bool VisionModalityEnabled() const { return vision_modality_enabled_; }
  void SetVisionModalityEnabled(bool enable_vision_modality) {
    vision_modality_enabled_ = enable_vision_modality;
  }

  // Sampler parameters:
  // Getters for the sampler parameters.
  const proto::SamplerParameters& GetSamplerParams() const;
  proto::SamplerParameters& GetMutableSamplerParams();

  // Stop token ids:
  // Getters for the stop token ids.
  const std::vector<std::vector<int>>& GetStopTokenIds() const;
  std::vector<std::vector<int>>& GetMutableStopTokenIds();

  // Set the start token ids.
  int GetStartTokenId() const;
  void SetStartTokenId(int start_token_id);

  // Number of output candidates:
  // Getters for the number of output candidates.
  int GetNumOutputCandidates() const;
  void SetNumOutputCandidates(int num_output_candidates);

  // Sampler backend:
  // Getters for the backend of the sampler.
  Backend GetSamplerBackend() const;
  void SetSamplerBackend(Backend sampler_backend);

  // Prompt templates:
  // Getters for the prompt templates.

  [[deprecated("Use Jinja prompt template instead.")]]
  const proto::PromptTemplates& GetPromptTemplates() const;
  [[deprecated("Use Jinja prompt template instead.")]]
  proto::PromptTemplates& GetMutablePromptTemplates();

  // Llm model type:
  // Getters for the LLM model type.
  const proto::LlmModelType& GetLlmModelType() const;
  proto::LlmModelType& GetMutableLlmModelType();

  // Whether to apply the deprecated prompt templates in the session.
  // TODO - b/453312248: Remove this field once the prompt templates are
  // removed.
  bool GetApplyPromptTemplateInSession() const {
    return apply_prompt_template_in_session_;
  }
  void SetApplyPromptTemplateInSession(bool apply_prompt_template_in_session) {
    apply_prompt_template_in_session_ = apply_prompt_template_in_session;
  }

  // Whether to use external sampler.
  bool UseExternalSampler() const { return use_external_sampler_; }
  void SetUseExternalSampler(bool use_external_sampler) {
    use_external_sampler_ = use_external_sampler;
  }

  // LoRA ID:
  // The LoRA adapter ID to activate for this session. If nullopt, no LoRA
  // adapter is used (base model).
  std::optional<uint32_t> GetLoraId() const { return lora_id_; }
  void SetLoraId(std::optional<uint32_t> lora_id) { lora_id_ = lora_id; }

  // Scoped LoRA file:
  // Getters for the scoped LoRA file.
  std::shared_ptr<ScopedFile> GetScopedLoraFile() const;
  void SetScopedLoraFile(
      std::shared_ptr<ScopedFile> scoped_lora_file);

  // The maximum number of tokens to generate in a single request:
  // Getters for the max output tokens.
  int GetMaxOutputTokens() const { return max_output_tokens_; }
  void SetMaxOutputTokens(int max_output_tokens) {
    max_output_tokens_ = max_output_tokens;
  }

 private:
  // Private constructor for the SessionConfig. The user should use the
  // CreateDefault() method to create a SessionConfig.
  explicit SessionConfig(const proto::SamplerParameters& sampler_params);

  // Whether to enable audio modality in the session.
  bool audio_modality_enabled_ = false;

  // Whether to enable vision modality in the session.
  bool vision_modality_enabled_ = false;

  // Parameters used to configure the sampling process.
  proto::SamplerParameters sampler_params_;

  // Stop token ids for the session. Note that the stop token could be a
  // sequence of token ids (as opposed to a single token id). The first
  // dimension is the index of the stop token in the session, and the second
  // dimension is the sequence of token ids that constitutes the stop token.
  std::vector<std::vector<int>> stop_token_ids_;

  // Start token id for the session.
  int start_token_id_ = -1;

  // Prompt templates for the session. This is loaded from the model assets (if
  // present).
  proto::PromptTemplates prompt_templates_;

  // Llm model type for the session. This is loaded from the model assets (if
  // present).
  proto::LlmModelType llm_model_type_;

  // The number of output candidates to generate. Default value is 1 and setting
  // it to a value greater than 1 will require the model to support batching.
  int num_output_candidates_ = 1;

  // Backend to use for sampling.
  Backend sampler_backend_ = Backend::UNSPECIFIED;

  // Whether to apply the deprecated prompt templates in the session.
  // TODO - b/453312248: Remove this field once the prompt templates are
  // removed.
  bool apply_prompt_template_in_session_ = true;

  // Whether to use external sampler.
  // notice: this is only used in advanced engine.
  bool use_external_sampler_ = false;

  // The LoRA adapter ID to activate for this session.
  std::optional<uint32_t> lora_id_;

  // Scoped file for the LoRA weights.
  std::shared_ptr<ScopedFile> scoped_lora_file_;

  // The maximum number of tokens to generate in a single request. This limits
  // the number of decoding steps for a request, as opposed to
  // LlmExecutorSettings::GetMaxNumTokens(), which limits the total number of
  // tokens (input + output) stored in the KV cache over the lifetime of a
  // session.
  int max_output_tokens_ = std::numeric_limits<int>::max();
};

std::ostream& operator<<(std::ostream& os, const SessionConfig& config);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_SETTINGS_H_
