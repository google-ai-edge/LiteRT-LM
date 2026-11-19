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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_RESOURCES_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

// Adapter around `lm::ModelResources` that maps standard LLM executor section
// types (`TF_LITE_PREFILL_DECODE`, `TF_LITE_EMBEDDER`) to the corresponding
// Any2Any `.litertlm` bundle section types (`TF_LITE_LLM_BACKBONE`,
// `TF_LITE_TEXT_EMBEDDER`) while delegating all other lookups unchanged.
class BackboneLmModelResources : public lm::ModelResources {
 public:
  explicit BackboneLmModelResources(
      std::shared_ptr<lm::ModelResources> absl_nonnull inner);

  using lm::ModelResources::GetTFLiteModel;
  using lm::ModelResources::GetTFLiteModelBackendConstraint;
  using lm::ModelResources::GetTFLiteModelBuffer;
  using lm::ModelResources::GetTFLiteModelPreferActivationType;
  using lm::ModelResources::GetTFLiteModelSectionFileRegion;
  using lm::ModelResources::GetTokenizer;
  using lm::ModelResources::GetWeightInMemoryMap;
  using lm::ModelResources::GetWeightsSectionOffset;
  using lm::ModelResources::HasTFLiteModel;

  bool HasTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  bool HasTFLiteModel(absl::string_view model_type_str) override;

  absl::StatusOr<const Model*> GetTFLiteModel(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<const Model*> GetTFLiteModel(
      absl::string_view model_type_str) override;

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override;

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      absl::string_view model_type_str) override;

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      absl::string_view model_type_str) override;

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str) override;

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::proto::LlmMetadata::TfLiteModelType model_type) override;
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str) override;

  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(
      lm::proto::LlmMetadata::TfLiteModelType model_type) const override;
  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(absl::string_view model_type_str) const override;

  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override;
  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override;
  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override;
  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override;
  absl::StatusOr<const lm::proto::Any2AnyMetadata*> GetAny2AnyMetadata()
      override;

 private:
  static std::string MapModelTypeString(absl::string_view model_type_str);

  std::shared_ptr<lm::ModelResources> absl_nonnull inner_;
};

// Options for creating shared Any2Any `.litertlm` model resources.
struct Any2AnyResourceOptions {
  // Human-readable model name included in diagnostic error messages.
  std::string model_display_name = "Any2Any";
  // When true, the TTS projector, TTS decoder, and audio vocoder sections are
  // required and compiled from the `.litertlm` bundle. When false, those
  // sections are skipped even if present in the bundle.
  bool generate_audio = false;
  // Optional per-component backend overrides. Unset vision, audio encoder,
  // LLM, and vocoder backends fall back to the `backend` argument of
  // `CreateAny2AnyResources`; unset `tts_backend` defaults to
  // `lm::Backend::CPU`.
  std::optional<lm::Backend> vision_backend;
  std::optional<lm::Backend> audio_encoder_backend;
  std::optional<lm::Backend> llm_backend;
  std::optional<lm::Backend> tts_backend;
  std::optional<lm::Backend> vocoder_backend;
  // Whether to force float32 precision on GPU instead of float16.
  bool force_fp32_on_gpu = false;
};

// Creates a `LiteRtLmRunner` for the Any2Any LLM backbone section from
// `lm_resources` using `BackboneLmModelResources` section mapping.
absl::StatusOr<std::unique_ptr<LiteRtLmRunner> absl_nonnull>
CreateBackboneLmRunner(
    Environment& env, const ModelOptions& options,
    std::shared_ptr<lm::ModelResources> absl_nonnull lm_resources);

// Compiles `section_type` from `lm_resources` into a `CompiledModel`. If the
// section is not found (`absl::StatusCode::kNotFound`), returns `nullptr` when
// `required` is false, or the `kNotFound` error when `required` is true. All
// other errors (such as corrupt section or I/O failures) are propagated
// regardless of `required`. `model_name` is used in cache keys and diagnostic
// messages.
absl::StatusOr<std::shared_ptr<CompiledModel> absl_nullable> TryCompileSection(
    Environment& env, lm::ModelResources& lm_resources,
    lm::proto::Any2AnyMetadata::TfLiteModelType section_type,
    absl::string_view model_name, const ModelOptions& options, bool required);

// Compiles and returns the standard Any2Any sub-models (`text_embedder` on
// CPU, optional `vision_encoder`, optional `audio_encoder`, `llm_backbone`,
// and, when `options.generate_audio` is true, required `tts_projector`,
// `tts_decoder`, and `audio_vocoder`) from a `.litertlm` bundle in a new
// `ModelResources` that retains `env` so `env` outlives all compiled models.
// If `lm_resources` is non-null, it is reused directly and `model_path` is not
// reopened.
absl::StatusOr<ModelResources> CreateAny2AnyResources(
    const Any2AnyResourceOptions& options, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources = nullptr);

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_ANY2ANY_RESOURCES_H_
