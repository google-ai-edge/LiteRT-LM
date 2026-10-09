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

#include <array>
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
#include "omni/text2image/flux2/klein_denoiser_stage.h"
#include "omni/text2image/flux2/klein_text_encoder_stage.h"
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

constexpr std::array<lm::proto::ImageGenMetadata::TfLiteModelType, 3>
    kTextEncTypes = {
        lm::proto::ImageGenMetadata::TF_LITE_TEXT_ENCODER_0,
        lm::proto::ImageGenMetadata::TF_LITE_TEXT_ENCODER_1,
        lm::proto::ImageGenMetadata::TF_LITE_TEXT_ENCODER_2,
    };

constexpr std::array<lm::proto::ImageGenMetadata::TfLiteModelType, 2>
    kDoubleBlockTypes = {
        lm::proto::ImageGenMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_DOUBLE_BLOCK_0,
        lm::proto::ImageGenMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_DOUBLE_BLOCK_1,
    };

constexpr std::array<lm::proto::ImageGenMetadata::TfLiteModelType, 4>
    kSingleBlockTypes = {
        lm::proto::ImageGenMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_0,
        lm::proto::ImageGenMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_1,
        lm::proto::ImageGenMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_2,
        lm::proto::ImageGenMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_3,
    };

absl::Status RegisterCompiledModel(
    ::litert::Environment& env, lm::ModelResources& lm_resources,
    const ModelOptions& options,
    lm::proto::ImageGenMetadata::TfLiteModelType model_type,
    absl::string_view resource_name, ModelResources& resources) {
  LITERT_ASSIGN_OR_RETURN(absl::string_view buffer,
                          lm_resources.GetTFLiteModelBuffer(model_type));
  LITERT_ASSIGN_OR_RETURN(
      auto compiled,
      CreateCompiledModelFromBuffer(env, options, buffer, resource_name));
  return resources.AddCompiledModel(
      resource_name, std::make_shared<CompiledModel>(std::move(compiled)));
}

absl::Status CreateKleinComponents(
    const Flux2ModelConfig& config, lm::ModelResources& lm_resources,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
    ModelResources& resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages,
    Stage<Output>* absl_nullable* absl_nonnull output_stage) {
  ABSL_ASSIGN_OR_RETURN(
      absl::string_view embed_table_fp16,
      lm_resources.GetGenericBinaryDataBuffer(config.text_embed_table_key));

  KleinTextEncoderStage::Config textenc_config;
  textenc_config.framing.seq_len = config.seq_len;
  textenc_config.prompt_dim = config.prompt_dim;
  std::array<std::unique_ptr<LiteRtRunner>, 3> textenc_runners;
  for (size_t i = 0; i < 3; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> shard_model,
        resources.GetCompiledModel(absl::StrCat("flux2_textenc_", i)));
    ABSL_ASSIGN_OR_RETURN(
        textenc_config.shard_input_indices[i].hidden,
        ResolveSignatureArgInputIndex(*shard_model, 0, "hidden"));
    ABSL_ASSIGN_OR_RETURN(
        textenc_config.shard_input_indices[i].attention_mask,
        ResolveSignatureArgInputIndex(*shard_model, 1, "attention_mask"));
    ABSL_ASSIGN_OR_RETURN(
        textenc_config.shard_input_indices[i].cos,
        ResolveSignatureArgInputIndex(*shard_model, 2, "cos"));
    ABSL_ASSIGN_OR_RETURN(
        textenc_config.shard_input_indices[i].sin,
        ResolveSignatureArgInputIndex(*shard_model, 3, "sin"));
    textenc_runners[i] = std::make_unique<LiteRtRunnerImpl>(shard_model.get());
  }
  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder,
      KleinTextEncoderStage::Create(
          prompt_source.get(), std::move(tokenizer), std::move(textenc_runners),
          embed_table_fp16, textenc_config));

  KleinDenoiserStage::Runners dit_runners;
  KleinDenoiserStage::InputIndices dit_indices;

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> initial_model,
                        resources.GetCompiledModel("flux2_dit_initial"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.initial.hidden,
      ResolveSignatureArgInputIndex(*initial_model, 0, "hidden"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.initial.enc,
      ResolveSignatureArgInputIndex(*initial_model, 1, "enc"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.initial.t,
                        ResolveSignatureArgInputIndex(*initial_model, 2, "t"));
  dit_runners.initial =
      std::make_unique<LiteRtRunnerImpl>(initial_model.get());

  for (size_t i = 0; i < 2; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> dbl_model,
        resources.GetCompiledModel(absl::StrCat("flux2_dit_double_", i)));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.double_blocks[i].image,
        ResolveSignatureArgInputIndex(*dbl_model, 0, "image"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.double_blocks[i].text,
                          ResolveSignatureArgInputIndex(*dbl_model, 1, "text"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.double_blocks[i].cos,
                          ResolveSignatureArgInputIndex(*dbl_model, 2, "cos"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.double_blocks[i].sin,
                          ResolveSignatureArgInputIndex(*dbl_model, 3, "sin"));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.double_blocks[i].mod_img,
        ResolveSignatureArgInputIndex(*dbl_model, 4, "mod_img"));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.double_blocks[i].mod_txt,
        ResolveSignatureArgInputIndex(*dbl_model, 5, "mod_txt"));
    dit_runners.double_blocks[i] =
        std::make_unique<LiteRtRunnerImpl>(dbl_model.get());
  }

  for (size_t i = 0; i < 4; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> sgl_model,
        resources.GetCompiledModel(absl::StrCat("flux2_dit_single_", i)));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.single_blocks[i].joint,
        ResolveSignatureArgInputIndex(*sgl_model, 0, "joint"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.single_blocks[i].cos,
                          ResolveSignatureArgInputIndex(*sgl_model, 1, "cos"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.single_blocks[i].sin,
                          ResolveSignatureArgInputIndex(*sgl_model, 2, "sin"));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.single_blocks[i].mod_single,
        ResolveSignatureArgInputIndex(*sgl_model, 3, "mod_single"));
    dit_runners.single_blocks[i] =
        std::make_unique<LiteRtRunnerImpl>(sgl_model.get());
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> final_model,
                        resources.GetCompiledModel("flux2_dit_final"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.final_stage.joint,
      ResolveSignatureArgInputIndex(*final_model, 0, "joint"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.final_stage.temb,
                        ResolveSignatureArgInputIndex(*final_model, 1, "temb"));
  dit_runners.final_stage =
      std::make_unique<LiteRtRunnerImpl>(final_model.get());

  ABSL_ASSIGN_OR_RETURN(
      auto denoiser,
      KleinDenoiserStage::Create(text_encoder.get(), config,
                                 std::move(dit_runners), dit_indices));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vae_model,
                        resources.GetCompiledModel("flux2_vae"));
  ABSL_ASSIGN_OR_RETURN(
      auto vae_decoder,
      Flux2VaeDecoderStage::Create(
          denoiser.get(), config,
          std::make_unique<LiteRtRunnerImpl>(vae_model.get())));

  *output_stage = vae_decoder.get();
  stages.push_back(std::move(prompt_source));
  stages.push_back(std::move(text_encoder));
  stages.push_back(std::move(denoiser));
  stages.push_back(std::move(vae_decoder));
  return absl::OkStatus();
}

absl::Status CreateBonsaiComponents(
    const Flux2ModelConfig& config,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
    ModelResources& resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages,
    Stage<Output>* absl_nullable* absl_nonnull output_stage) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> textenc_model,
                        resources.GetCompiledModel("flux2_textenc"));
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
                        resources.GetCompiledModel("flux2_dit"));
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
                        resources.GetCompiledModel("flux2_vae"));
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
  if (!config.is_klein &&
      lm_resources
          ->GetTFLiteModelBuffer(lm::proto::ImageGenMetadata::
                                     TF_LITE_DIFFUSION_TRANSFORMER_INITIAL)
          .ok()) {
    config.is_klein = true;
  }

  if (config.is_klein) {
    ABSL_RETURN_IF_ERROR(
        lm_resources->GetGenericBinaryDataBuffer(config.text_embed_table_key)
            .status());
    for (size_t i = 0; i < kTextEncTypes.size(); ++i) {
      ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
          env, *lm_resources, textenc_options, kTextEncTypes[i],
          absl::StrCat("flux2_textenc_", i), resources));
    }

    ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
        env, *lm_resources, dit_options,
        lm::proto::ImageGenMetadata::TF_LITE_DIFFUSION_TRANSFORMER_INITIAL,
        "flux2_dit_initial", resources));

    for (size_t i = 0; i < kDoubleBlockTypes.size(); ++i) {
      ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
          env, *lm_resources, dit_options, kDoubleBlockTypes[i],
          absl::StrCat("flux2_dit_double_", i), resources));
    }

    for (size_t i = 0; i < kSingleBlockTypes.size(); ++i) {
      ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
          env, *lm_resources, dit_options, kSingleBlockTypes[i],
          absl::StrCat("flux2_dit_single_", i), resources));
    }

    ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
        env, *lm_resources, dit_options,
        lm::proto::ImageGenMetadata::TF_LITE_DIFFUSION_TRANSFORMER_FINAL,
        "flux2_dit_final", resources));
  } else {
    ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
        env, *lm_resources, textenc_options,
        lm::proto::ImageGenMetadata::TF_LITE_TEXT_ENCODER, "flux2_textenc",
        resources));
    ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
        env, *lm_resources, dit_options,
        lm::proto::ImageGenMetadata::TF_LITE_IMAGE_DENOISER, "flux2_dit",
        resources));
  }

  ABSL_RETURN_IF_ERROR(RegisterCompiledModel(
       env, *lm_resources, vae_options,
       lm::proto::ImageGenMetadata::TF_LITE_IMAGE_DECODER, "flux2_vae",
       resources));
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

  const bool is_klein =
      config.is_klein || resources->GetCompiledModel("flux2_dit_initial").ok();
  if (is_klein) {
    return CreateKleinComponents(config, *lm_resources,
                                 std::move(prompt_source), std::move(tokenizer),
                                 *resources, stages, output_stage);
  }
  return CreateBonsaiComponents(config, std::move(prompt_source),
                                std::move(tokenizer), *resources, stages,
                                output_stage);
}

}  // namespace litert::omni::text2image
