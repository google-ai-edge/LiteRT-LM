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

#include "omni/text2image/flux2/flux2_factory.h"

#include <cstddef>
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
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_denoiser_stage.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/flux2/flux2_vae_decoder_stage.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text_encoder_stage.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/image_gen_metadata.pb.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image {
namespace {

ModelOptions MakeModelOptions(absl::string_view model_dir,
                              absl::string_view cache_dir, lm::Backend backend,
                              int num_threads) {
  ModelOptions options;
  options.model_dir = model_dir;
  options.cache_dir = cache_dir;
  options.backend = backend;
  options.num_threads = num_threads;
  return options;
}

absl::StatusOr<size_t> ResolveSignatureArgInputIndex(
    const CompiledModel& model, size_t arg_index,
    absl::string_view fallback_name = "") {
  const std::string exact_arg = absl::StrCat("args_", arg_index);
  const std::string suffixed_arg = absl::StrCat("_", exact_arg);
  const std::string prefixed_arg = absl::StrCat("args_", arg_index, ":");
  auto names_res = model.GetSignatureInputNames();
  if (names_res.HasValue()) {
    const auto& names = *names_res;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == exact_arg || absl::EndsWith(names[i], suffixed_arg) ||
          absl::StrContains(names[i], prefixed_arg)) {
        return i;
      }
    }
    if (!fallback_name.empty()) {
      for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == fallback_name) {
          return i;
        }
      }
    }
    if (arg_index < names.size()) {
      return arg_index;
    }
    return absl::NotFoundError(absl::StrCat(
        "Input argument ", exact_arg, " exceeds model signature input count (",
        names.size(), ")."));
  }
  return arg_index;
}

}  // namespace

absl::Status InitFlux2Resources(Flux2ModelConfig& config,
                                absl::string_view model_folder,
                                absl::string_view cache_dir,
                                lm::Backend backend, int num_threads,
                                ::litert::Environment& env,
                                ModelResources& resources) {
  if (!resources.HasLmModelResources()) {
    return absl::NotFoundError(
        absl::StrCat("No .litertlm model container found in: ", model_folder));
  }

  const lm::Backend textenc_backend = config.textenc_backend.value_or(backend);
  const lm::Backend dit_backend = config.dit_backend.value_or(backend);
  const lm::Backend vae_backend = config.vae_backend.value_or(backend);

  ModelOptions textenc_options =
      MakeModelOptions(model_folder, cache_dir, textenc_backend, num_threads);
  ModelOptions dit_options =
      MakeModelOptions(model_folder, cache_dir, dit_backend, num_threads);
  ModelOptions vae_options =
      MakeModelOptions(model_folder, cache_dir, vae_backend, num_threads);

  auto lm_resources = resources.GetLmModelResources();
  // TODO(b/568027544): Consolidate how model configs and proto metadata
  // overrides are defined.
  auto image_gen_metadata = lm_resources->GetImageGenMetadata();
  if (image_gen_metadata.ok() && *image_gen_metadata != nullptr) {
    const auto& model_type = (*image_gen_metadata)->image_gen_model_type();
    if (model_type.has_bonsai_flux2()) {
      PopulateFlux2ConfigFromProto(model_type.bonsai_flux2(), config);
    } else if (model_type.has_flux2_klein()) {
      PopulateFlux2ConfigFromProto(model_type.flux2_klein(), config);
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      absl::string_view textenc_buffer,
      lm_resources->GetTFLiteModelBuffer(
          lm::proto::ImageGenMetadata::TF_LITE_TEXT_ENCODER));
  LITERT_ASSIGN_OR_RETURN(
      auto textenc_compiled,
      CreateCompiledModelFromBuffer(env, textenc_options, textenc_buffer,
                                    "flux2_textenc"));

  LITERT_ASSIGN_OR_RETURN(
      absl::string_view dit_buffer,
      lm_resources->GetTFLiteModelBuffer(
          lm::proto::ImageGenMetadata::TF_LITE_IMAGE_DENOISER));
  LITERT_ASSIGN_OR_RETURN(
      auto dit_compiled,
      CreateCompiledModelFromBuffer(env, dit_options, dit_buffer, "flux2_dit"));

  LITERT_ASSIGN_OR_RETURN(
      absl::string_view vae_buffer,
      lm_resources->GetTFLiteModelBuffer(
          lm::proto::ImageGenMetadata::TF_LITE_IMAGE_DECODER));
  LITERT_ASSIGN_OR_RETURN(
      auto vae_compiled,
      CreateCompiledModelFromBuffer(env, vae_options, vae_buffer, "flux2_vae"));

  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "flux2_textenc",
      std::make_shared<CompiledModel>(std::move(textenc_compiled))));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "flux2_dit", std::make_shared<CompiledModel>(std::move(dit_compiled))));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      "flux2_vae", std::make_shared<CompiledModel>(std::move(vae_compiled))));
  return absl::OkStatus();
}

absl::Status CreateFlux2Components(
    const Flux2ModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages,
    Stage<Output>* absl_nullable* absl_nonnull output_stage) {
  if (!resources->HasLmModelResources()) {
    return absl::InvalidArgumentError(
        "FLUX.2 requires a .litertlm model container in ModelResources.");
  }
  auto lm_resources = resources->GetLmModelResources();
  auto tok = lm_resources->GetTokenizer();
  if (!tok.ok()) {
    tok = lm_resources->GetTokenizer(lm::ModelType::kTfLiteTextEncoder);
  }
  if (!tok.ok()) {
    return tok.status();
  }
  std::unique_ptr<support::Tokenizer> tokenizer = *std::move(tok);

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> textenc_model,
                        resources->GetCompiledModel("flux2_textenc"));
  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  ABSL_ASSIGN_OR_RETURN(
      textenc_config.input_indices.input_ids,
      ResolveSignatureArgInputIndex(*textenc_model, 0, "input_ids"));
  ABSL_ASSIGN_OR_RETURN(
      textenc_config.input_indices.attention_mask,
      ResolveSignatureArgInputIndex(*textenc_model, 1, "attention_mask"));

  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder,
      TextEncoderStage::Create(
          prompt_source.get(), std::move(tokenizer),
          std::make_unique<LiteRtRunnerImpl>(textenc_model.get()),
          textenc_config));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> dit_model,
                        resources->GetCompiledModel("flux2_dit"));
  Flux2DenoiserStage::InputIndices dit_indices;
  ABSL_ASSIGN_OR_RETURN(dit_indices.hidden,
                        ResolveSignatureArgInputIndex(*dit_model, 0, "hidden"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.enc,
                        ResolveSignatureArgInputIndex(*dit_model, 1, "enc"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.t,
                        ResolveSignatureArgInputIndex(*dit_model, 2, "t"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.img_ids, ResolveSignatureArgInputIndex(
                                                 *dit_model, 3, "img_ids"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.txt_ids, ResolveSignatureArgInputIndex(
                                                 *dit_model, 4, "txt_ids"));
  ABSL_ASSIGN_OR_RETURN(
      auto denoiser,
      Flux2DenoiserStage::Create(
          text_encoder.get(), config,
          std::make_unique<LiteRtRunnerImpl>(dit_model.get()), dit_indices));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vae_model,
                        resources->GetCompiledModel("flux2_vae"));
  ABSL_ASSIGN_OR_RETURN(
      auto vae_decoder,
      Flux2VaeDecoderStage::Create(
          denoiser.get(), config,
          std::make_unique<LiteRtRunnerImpl>(vae_model.get())));

  *output_stage = vae_decoder.get();
  // The first stage must be `PromptSource`.
  stages.push_back(std::move(prompt_source));
  stages.push_back(std::move(text_encoder));
  stages.push_back(std::move(denoiser));
  stages.push_back(std::move(vae_decoder));
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image
