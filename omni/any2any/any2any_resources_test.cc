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

#include "omni/any2any/any2any_resources.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::Pair;

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|endoftext|>": 0,
      "Hello": 1
    },
    "unk_token": "<|endoftext|>"
  }
})json";

class FakeLmModelResources : public lm::ModelResources {
 public:
  using lm::ModelResources::GetTFLiteModel;
  using lm::ModelResources::GetTFLiteModelBackendConstraint;
  using lm::ModelResources::GetTFLiteModelBuffer;
  using lm::ModelResources::GetTFLiteModelPreferActivationType;
  using lm::ModelResources::GetTFLiteModelSectionFileRegion;
  using lm::ModelResources::GetWeightInMemoryMap;
  using lm::ModelResources::GetWeightsSectionOffset;
  using lm::ModelResources::HasTFLiteModel;

  bool HasTFLiteModel(absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    return buffers_.contains(model_type_str) ||
           models_.contains(model_type_str);
  }

  absl::StatusOr<const Model*> GetTFLiteModel(
      absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    auto it = models_.find(model_type_str);
    if (it == models_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Missing model: ", model_type_str));
    }
    return &it->second;
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    if (auto err_it = buffer_errors_.find(model_type_str);
        err_it != buffer_errors_.end()) {
      return err_it->second;
    }
    auto it = buffers_.find(model_type_str);
    if (it == buffers_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Missing buffer: ", model_type_str));
    }
    return absl::string_view(it->second);
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    auto it = weights_offsets_.find(model_type_str);
    if (it == weights_offsets_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Missing weights offset: ", model_type_str));
    }
    return it->second;
  }

  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    auto it = file_regions_.find(model_type_str);
    if (it == file_regions_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Missing file region: ", model_type_str));
    }
    return it->second;
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    auto it = backend_constraints_.find(model_type_str);
    if (it == backend_constraints_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str) override {
    last_queried_type_ = std::string(model_type_str);
    auto it = prefer_activation_types_.find(model_type_str);
    if (it == prefer_activation_types_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(absl::string_view model_type_str) const override {
    last_queried_type_ = std::string(model_type_str);
    auto it = weight_maps_.find(model_type_str);
    if (it == weight_maps_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::FailedPreconditionError("fake scoped file error");
  }

  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("fake tokenizer error");
  }

  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return &llm_metadata_;
  }

  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return &executor_metadata_;
  }

  absl::StatusOr<const lm::proto::Any2AnyMetadata*> GetAny2AnyMetadata()
      override {
    return &any2any_metadata_;
  }

  mutable std::string last_queried_type_;
  absl::flat_hash_map<std::string, absl::Status> buffer_errors_;
  absl::flat_hash_map<std::string, std::string> buffers_;
  absl::flat_hash_map<std::string, Model> models_;
  absl::flat_hash_map<std::string, std::pair<size_t, size_t>> weights_offsets_;
  absl::flat_hash_map<std::string, lm::FileRegion> file_regions_;
  absl::flat_hash_map<std::string, std::string> backend_constraints_;
  absl::flat_hash_map<std::string, std::string> prefer_activation_types_;
  absl::flat_hash_map<
      std::string,
      absl::flat_hash_map<std::string, absl::Span<const std::byte>>>
      weight_maps_;
  lm::proto::LlmMetadata llm_metadata_;
  lm::proto::ExecutorMetadata executor_metadata_;
  lm::proto::Any2AnyMetadata any2any_metadata_;
};

TEST(Any2AnyResourcesTest, BackboneLmModelResourcesMapsSectionsAndDelegates) {
  auto fake = std::make_shared<FakeLmModelResources>();
  fake->buffers_["tf_lite_llm_backbone"] = BuildMinimalTfliteModelBuffer(
      {{"input", {1, 4}, tflite::TensorType_FLOAT32}},
      {{"output", {1, 4}, tflite::TensorType_FLOAT32}});
  fake->buffers_["tf_lite_text_embedder"] = BuildMinimalTfliteModelBuffer(
      {{"token_ids", {1, 1}, tflite::TensorType_INT32}},
      {{"embeddings", {1, 1, 4}, tflite::TensorType_FLOAT32}});
  fake->buffers_["tf_lite_vision_encoder"] = "vision_bytes";

  ASSERT_OK_AND_ASSIGN(
      Model backbone_model,
      lm::CreateLiteRtModelFromBuffer(litert::BufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(
              fake->buffers_["tf_lite_llm_backbone"].data()),
          fake->buffers_["tf_lite_llm_backbone"].size())));
  ASSERT_OK_AND_ASSIGN(
      Model embedder_model,
      lm::CreateLiteRtModelFromBuffer(litert::BufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(
              fake->buffers_["tf_lite_text_embedder"].data()),
          fake->buffers_["tf_lite_text_embedder"].size())));
  fake->models_.emplace("tf_lite_llm_backbone", std::move(backbone_model));
  fake->models_.emplace("tf_lite_text_embedder", std::move(embedder_model));

  fake->weights_offsets_["tf_lite_llm_backbone"] = {100, 200};
  fake->weights_offsets_["tf_lite_text_embedder"] = {300, 400};
  fake->file_regions_["tf_lite_llm_backbone"] = {.offset = 10, .size = 20};
  fake->file_regions_["tf_lite_text_embedder"] = {.offset = 30, .size = 40};
  fake->backend_constraints_["tf_lite_llm_backbone"] = "gpu";
  fake->backend_constraints_["tf_lite_text_embedder"] = "cpu";
  fake->prefer_activation_types_["tf_lite_llm_backbone"] = "fp16";
  fake->prefer_activation_types_["tf_lite_text_embedder"] = "fp32";
  fake->weight_maps_["tf_lite_llm_backbone"] = {};
  fake->weight_maps_["tf_lite_text_embedder"] = {};
  fake->any2any_metadata_.set_input_sample_rate_hz(24000);

  BackboneLmModelResources adapter(fake);

  using lm::proto::LlmMetadata;
  // Enum and mixed-case string lookups for PREFILL_DECODE -> llm_backbone.
  EXPECT_TRUE(adapter.HasTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_EQ(fake->last_queried_type_, "tf_lite_llm_backbone");
  EXPECT_TRUE(adapter.HasTFLiteModel("TF_LITE_PREFILL_DECODE"));
  EXPECT_EQ(fake->last_queried_type_, "tf_lite_llm_backbone");
  EXPECT_THAT(adapter.GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE),
              IsOkAndHolds(&fake->models_.at("tf_lite_llm_backbone")));
  EXPECT_THAT(adapter.GetTFLiteModelBuffer(LlmMetadata::TF_LITE_PREFILL_DECODE),
              IsOkAndHolds(fake->buffers_.at("tf_lite_llm_backbone")));
  EXPECT_THAT(
      adapter.GetWeightsSectionOffset(LlmMetadata::TF_LITE_PREFILL_DECODE),
      IsOkAndHolds(Pair(100u, 200u)));
  ASSERT_OK_AND_ASSIGN(lm::FileRegion backbone_region,
                       adapter.GetTFLiteModelSectionFileRegion(
                           LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_EQ(backbone_region.offset, 10u);
  EXPECT_EQ(backbone_region.size, 20u);
  EXPECT_THAT(adapter.GetTFLiteModelBackendConstraint(
                  LlmMetadata::TF_LITE_PREFILL_DECODE),
              Optional(std::string("gpu")));
  EXPECT_THAT(adapter.GetTFLiteModelPreferActivationType(
                  LlmMetadata::TF_LITE_PREFILL_DECODE),
              Optional(std::string("fp16")));
  EXPECT_EQ(adapter.GetWeightInMemoryMap(LlmMetadata::TF_LITE_PREFILL_DECODE),
            &fake->weight_maps_.at("tf_lite_llm_backbone"));

  // Enum and mixed-case string lookups for EMBEDDER -> text_embedder.
  EXPECT_TRUE(adapter.HasTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_EQ(fake->last_queried_type_, "tf_lite_text_embedder");
  EXPECT_TRUE(adapter.HasTFLiteModel("TF_LITE_EMBEDDER"));
  EXPECT_EQ(fake->last_queried_type_, "tf_lite_text_embedder");
  EXPECT_THAT(adapter.GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
              IsOkAndHolds(&fake->models_.at("tf_lite_text_embedder")));
  EXPECT_THAT(adapter.GetTFLiteModelBuffer(LlmMetadata::TF_LITE_EMBEDDER),
              IsOkAndHolds(fake->buffers_.at("tf_lite_text_embedder")));
  EXPECT_THAT(adapter.GetWeightsSectionOffset(LlmMetadata::TF_LITE_EMBEDDER),
              IsOkAndHolds(Pair(300u, 400u)));
  ASSERT_OK_AND_ASSIGN(
      lm::FileRegion embedder_region,
      adapter.GetTFLiteModelSectionFileRegion(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_EQ(embedder_region.offset, 30u);
  EXPECT_EQ(embedder_region.size, 40u);
  EXPECT_THAT(
      adapter.GetTFLiteModelBackendConstraint(LlmMetadata::TF_LITE_EMBEDDER),
      Optional(std::string("cpu")));
  EXPECT_THAT(
      adapter.GetTFLiteModelPreferActivationType(LlmMetadata::TF_LITE_EMBEDDER),
      Optional(std::string("fp32")));
  EXPECT_EQ(adapter.GetWeightInMemoryMap(LlmMetadata::TF_LITE_EMBEDDER),
            &fake->weight_maps_.at("tf_lite_text_embedder"));

  // Unmapped model types pass through unchanged.
  EXPECT_TRUE(adapter.HasTFLiteModel("tf_lite_vision_encoder"));
  EXPECT_EQ(fake->last_queried_type_, "tf_lite_vision_encoder");
  EXPECT_THAT(adapter.GetTFLiteModelBuffer("tf_lite_vision_encoder"),
              IsOkAndHolds("vision_bytes"));
  EXPECT_EQ(adapter.GetWeightInMemoryMap("tf_lite_vision_encoder"), nullptr);

  // Metadata and resource accessors delegate directly.
  EXPECT_THAT(adapter.GetLlmMetadata(), IsOkAndHolds(&fake->llm_metadata_));
  EXPECT_THAT(adapter.GetExecutorMetadata(),
              IsOkAndHolds(&fake->executor_metadata_));
  EXPECT_THAT(adapter.GetAny2AnyMetadata(),
              IsOkAndHolds(&fake->any2any_metadata_));
  EXPECT_THAT(adapter.GetScopedFile(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(adapter.GetTokenizer(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(Any2AnyResourcesTest, CreateBackboneLmRunnerRejectsNullModelResources) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  std::shared_ptr<lm::ModelResources> null_resources;
  EXPECT_THAT(CreateBackboneLmRunner(*env, ModelOptions{}, null_resources),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Any2AnyResourcesTest,
     TryCompileSectionHandlesRequiredOptionalAndPresentSections) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  FakeLmModelResources fake_lm;
  fake_lm.buffers_["tf_lite_vision_encoder"] = BuildMinimalTfliteModelBuffer(
      {{"pixel_values", {1, 3, 28, 28}, tflite::TensorType_FLOAT32}},
      {{"vision_embeds", {1, 1, 4}, tflite::TensorType_FLOAT32}});

  const ModelOptions opts{.backend = lm::Backend::CPU, .num_threads = 1};

  // Missing optional section returns nullptr.
  EXPECT_THAT(
      TryCompileSection(*env, fake_lm,
                        lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_ENCODER,
                        "audio_encoder", opts, /*required=*/false),
      IsOkAndHolds(IsNull()));

  // Missing required section fails.
  EXPECT_THAT(
      TryCompileSection(*env, fake_lm,
                        lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_ENCODER,
                        "audio_encoder", opts, /*required=*/true),
      StatusIs(absl::StatusCode::kNotFound));

  // Present required and optional sections compile and return non-null models.
  EXPECT_THAT(
      TryCompileSection(*env, fake_lm,
                        lm::proto::Any2AnyMetadata::TF_LITE_VISION_ENCODER,
                        "vision_encoder", opts, /*required=*/true),
      IsOkAndHolds(NotNull()));
  EXPECT_THAT(
      TryCompileSection(*env, fake_lm,
                        lm::proto::Any2AnyMetadata::TF_LITE_VISION_ENCODER,
                        "vision_encoder", opts, /*required=*/false),
      IsOkAndHolds(NotNull()));

  // Non-NotFound errors propagate even when required is false.
  fake_lm.buffer_errors_["tf_lite_audio_vocoder"] =
      absl::DataLossError("corrupt section");
  EXPECT_THAT(
      TryCompileSection(*env, fake_lm,
                        lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_VOCODER,
                        "audio_vocoder", opts, /*required=*/false),
      StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(
      TryCompileSection(*env, fake_lm,
                        lm::proto::Any2AnyMetadata::TF_LITE_AUDIO_VOCODER,
                        "audio_vocoder", opts, /*required=*/true),
      StatusIs(absl::StatusCode::kDataLoss));
}

TEST(Any2AnyResourcesTest, CreateAny2AnyResourcesRejectsNullEnvironment) {
  std::shared_ptr<Environment> null_env;
  EXPECT_THAT(
      CreateAny2AnyResources(Any2AnyResourceOptions{}, "/tmp/model.litertlm",
                             "", lm::Backend::CPU, /*num_threads=*/1, null_env),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Environment must not be null")));
}

TEST(Any2AnyResourcesTest, CreateAny2AnyResourcesRejectsNonLitertLmPath) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  Any2AnyResourceOptions custom_opts{.model_display_name = "TestOmni"};
  EXPECT_THAT(
      CreateAny2AnyResources(custom_opts, "/tmp/not_a_bundle_dir", "",
                             lm::Backend::CPU, /*num_threads=*/1, shared_env),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("TestOmni requires a .litertlm model file")));

  Any2AnyResourceOptions empty_name_opts{.model_display_name = ""};
  EXPECT_THAT(
      CreateAny2AnyResources(empty_name_opts, "/tmp/not_a_bundle_dir", "",
                             lm::Backend::CPU, /*num_threads=*/1, shared_env),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Any2Any requires a .litertlm model file")));
}

TEST(Any2AnyResourcesTest,
     CreateAny2AnyResourcesCompilesOptionalAndAudioRequiredSections) {
  Any2AnyBundleOptions bundle_opts{
      .tokenizer_json = std::string(kMinimalTokenizerJson),
      .extra_sections = {BundleSectionPayload{
          .data_type = lm::schema::AnySectionDataType_TFLiteModel,
          .model_type = "tf_lite_vision_encoder",
          .bytes = BuildMinimalTfliteModelBuffer(
              {{"pixel_values", {1, 3, 28, 28}, tflite::TensorType_FLOAT32}},
              {{"vision_embeds", {1, 1, 4}, tflite::TensorType_FLOAT32}}),
      }},
  };
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteAny2AnyTestBundle("any2any_resources_test.litertlm",
                                              std::move(bundle_opts)));

  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  // Without generate_audio, missing audio_encoder and TTS/vocoder sections are
  // tolerated while text_embedder, vision_encoder, and llm_backbone are loaded.
  {
    Any2AnyResourceOptions opts{
        .model_display_name = "TestOmni",
        .generate_audio = false,
    };
    ASSERT_OK_AND_ASSIGN(
        ModelResources resources,
        CreateAny2AnyResources(opts, bundle_path, ":nocache", lm::Backend::CPU,
                               /*num_threads=*/1, shared_env));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_TRUE(resources.HasLmModelResources());
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_TRUE(resources.HasCompiledModel("vision_encoder"));
    EXPECT_FALSE(resources.HasCompiledModel("audio_encoder"));
    EXPECT_FALSE(resources.HasCompiledModel("tts_projector"));
    EXPECT_FALSE(resources.HasCompiledModel("tts_decoder"));
    EXPECT_FALSE(resources.HasCompiledModel("audio_vocoder"));
    EXPECT_OK(resources.GetLmRunner("llm_backbone"));
  }

  // With generate_audio=true, missing TTS/vocoder sections fail creation.
  {
    Any2AnyResourceOptions opts{
        .model_display_name = "TestOmni",
        .generate_audio = true,
    };
    EXPECT_THAT(
        CreateAny2AnyResources(opts, bundle_path, ":nocache", lm::Backend::CPU,
                               /*num_threads=*/1, shared_env),
        StatusIs(absl::StatusCode::kNotFound));
  }

  // Pre-populated LmModelResources is reused even if model_path has no
  // `.litertlm` extension.
  {
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<lm::ModelResources> lm_resources,
                         CreateLmModelResources(bundle_path));
    Any2AnyResourceOptions opts{.generate_audio = false};
    ASSERT_OK_AND_ASSIGN(
        ModelResources resources,
        CreateAny2AnyResources(opts, /*model_path=*/"", ":nocache",
                               lm::Backend::CPU, /*num_threads=*/1, shared_env,
                               lm_resources));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_EQ(resources.GetLmModelResources(), lm_resources);
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_OK(resources.GetLmRunner("llm_backbone"));
  }
}

TEST(Any2AnyResourcesTest,
     CreateAny2AnyResourcesSkipsTtsAndVocoderWhenGenerateAudioDisabled) {
  Any2AnyBundleOptions bundle_opts{
      .tokenizer_json = std::string(kMinimalTokenizerJson),
      .extra_sections =
          {
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_tts_projector",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"llm_hidden", {1, 4}, tflite::TensorType_FLOAT32}},
                      {{"cond_emb", {1, 4}, tflite::TensorType_FLOAT32}}),
              },
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_tts_decoder",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"cond_emb", {1, 4}, tflite::TensorType_FLOAT32}},
                      {{"code_logits", {1, 8}, tflite::TensorType_FLOAT32}}),
              },
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_audio_vocoder",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"codes", {1, 4}, tflite::TensorType_INT32}},
                      {{"waveform", {1, 8}, tflite::TensorType_FLOAT32}}),
              },
          },
  };
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteAny2AnyTestBundle("any2any_resources_audio_gate_test.litertlm",
                             std::move(bundle_opts)));

  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  // When generate_audio is false, TTS and vocoder sections present in the
  // bundle must NOT be compiled.
  {
    Any2AnyResourceOptions opts{.generate_audio = false};
    ASSERT_OK_AND_ASSIGN(
        ModelResources resources,
        CreateAny2AnyResources(opts, bundle_path, ":nocache", lm::Backend::CPU,
                               /*num_threads=*/1, shared_env));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_OK(resources.GetLmRunner("llm_backbone"));
    EXPECT_FALSE(resources.HasCompiledModel("tts_projector"));
    EXPECT_FALSE(resources.HasCompiledModel("tts_decoder"));
    EXPECT_FALSE(resources.HasCompiledModel("audio_vocoder"));
  }

  // When generate_audio is true, all three TTS/vocoder sections ARE compiled.
  {
    Any2AnyResourceOptions opts{.generate_audio = true};
    ASSERT_OK_AND_ASSIGN(
        ModelResources resources,
        CreateAny2AnyResources(opts, bundle_path, ":nocache", lm::Backend::CPU,
                               /*num_threads=*/1, shared_env));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_OK(resources.GetLmRunner("llm_backbone"));
    EXPECT_TRUE(resources.HasCompiledModel("tts_projector"));
    EXPECT_TRUE(resources.HasCompiledModel("tts_decoder"));
    EXPECT_TRUE(resources.HasCompiledModel("audio_vocoder"));
  }
}

}  // namespace
}  // namespace litert::omni::any2any
