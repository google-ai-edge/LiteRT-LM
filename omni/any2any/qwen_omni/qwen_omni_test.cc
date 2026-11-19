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
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/any2any/any2any_session.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/any2any/qwen_omni/qwen_omni_factory.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Optional;

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 6, "content": "<|audio_bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 7, "content": "<|audio_eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 8, "content": "<|vision_bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 9, "content": "<|vision_eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 10, "content": "<|AUDIO|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 11, "content": "<|IMAGE|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 12, "content": "<|VIDEO|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "Hello": 3,
      "world": 4,
      "\n": 5,
      "<|audio_bos|>": 6,
      "<|audio_eos|>": 7,
      "<|vision_bos|>": 8,
      "<|vision_eos|>": 9,
      "<|AUDIO|>": 10,
      "<|IMAGE|>": 11,
      "<|VIDEO|>": 12,
      "assistant": 13,
      "0": 14, "t15": 15, "t16": 16, "t17": 17, "t18": 18,
      "t19": 19, "t20": 20, "t21": 21, "t22": 22, "t23": 23, "t24": 24,
      "t25": 25, "t26": 26, "t27": 27, "t28": 28, "t29": 29, "t30": 30,
      "t31": 31
    },
    "unk_token": "<|endoftext|>"
  }
})json";

struct TestBundleOptions {
  bool include_vision_encoder = true;
  bool include_audio_encoder = true;
  bool include_tts_and_vocoder = true;
  bool is_moe = false;
  int num_codebooks = 1;
};

absl::StatusOr<std::string> WriteTestQwenOmniLitertLmFile(
    absl::string_view filename, const TestBundleOptions& opts = {}) {
  lm::proto::Any2AnyMetadata any2any_meta;
  any2any_meta.add_supported_languages("en-US");
  any2any_meta.set_input_sample_rate_hz(16000);
  any2any_meta.set_output_sample_rate_hz(24000);
  auto* qwen_proto =
      any2any_meta.mutable_any2any_model_type()->mutable_qwen_omni();
  qwen_proto->set_llm_hidden_size(4);
  qwen_proto->set_llm_vocab_size(32);
  qwen_proto->set_tts_hidden_size(4);
  qwen_proto->set_tts_vocab_size(8);
  qwen_proto->set_tts_num_mel_bins(80);
  qwen_proto->set_tts_num_codebooks(opts.num_codebooks);
  qwen_proto->set_vision_image_size(28);
  qwen_proto->set_vision_patch_size(14);
  qwen_proto->set_vision_spatial_merge_size(2);
  qwen_proto->set_audio_num_mel_bins(128);
  qwen_proto->set_audio_chunk_frames(10);
  qwen_proto->set_audio_pool_step(2);
  qwen_proto->set_generate_audio(false);
  qwen_proto->set_max_new_tokens(1);
  qwen_proto->set_max_tts_tokens(4);
  qwen_proto->set_voice("Chelsie");
  qwen_proto->set_language("en-US");
  qwen_proto->set_is_moe(opts.is_moe);
  qwen_proto->set_tts_text_bos_token_id(15);

  Any2AnyBundleOptions bundle_opts{
      .metadata = std::move(any2any_meta),
      .tokenizer_json = std::string(kMinimalTokenizerJson),
      .backbone_backend_constraint = "cpu",
      .backbone_prefer_activation_type = "fp32",
  };
  if (opts.include_vision_encoder) {
    bundle_opts.extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_vision_encoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"pixel_values", {1, 3, 28, 28}, tflite::TensorType_FLOAT32}},
            {{"vision_embeds", {1, 1, 4}, tflite::TensorType_FLOAT32}}),
    });
  }
  if (opts.include_audio_encoder) {
    bundle_opts.extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_audio_encoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"mel_features", {1, 128, 10}, tflite::TensorType_FLOAT32}},
            {{"audio_embeds", {1, 2, 4}, tflite::TensorType_FLOAT32}}),
    });
  }
  if (opts.include_tts_and_vocoder) {
    bundle_opts.extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_tts_projector",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"llm_hidden", {1, 1, 4}, tflite::TensorType_FLOAT32},
             {"token_id", {1, 1}, tflite::TensorType_INT32}},
            {{"cond_emb", {1, 1, 4}, tflite::TensorType_FLOAT32}}),
    });
    bundle_opts.extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_tts_decoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"cond_emb", {1, 1, 4}, tflite::TensorType_FLOAT32},
             {"prev_audio_code", {1, 1}, tflite::TensorType_INT32},
             {"step_idx", {1}, tflite::TensorType_INT32},
             {"kv_cache", {1, 2, 64, 4}, tflite::TensorType_FLOAT32}},
            {{"code_logits", {1, 1, 8}, tflite::TensorType_FLOAT32},
             {"new_kv_cache", {1, 2, 64, 4}, tflite::TensorType_FLOAT32}}),
    });
    bundle_opts.extra_sections.push_back(BundleSectionPayload{
        .data_type = lm::schema::AnySectionDataType_TFLiteModel,
        .model_type = "tf_lite_audio_vocoder",
        .bytes = BuildMinimalTfliteModelBuffer(
            {{"audio_codes", {1, 4}, tflite::TensorType_INT32}},
            {{"waveform", {1, 8}, tflite::TensorType_FLOAT32}}),
    });
  }
  return WriteAny2AnyTestBundle(filename, std::move(bundle_opts));
}

TEST(QwenOmniTest, CreateQwenOmniResourcesFailsWithNonexistentModels) {
  // Arrange
  QwenOmniModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));

  // Act & Assert
  std::shared_ptr<::litert::Environment> null_env;
  EXPECT_THAT(
      CreateQwenOmniResources(config, "/tmp/nonexistent_qwen_omni.litertlm", "",
                              lm::Backend::CPU, /*num_threads=*/1, null_env),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      CreateQwenOmniResources(config, "/tmp/nonexistent_qwen_omni_dir", "",
                              lm::Backend::CPU, /*num_threads=*/1, shared_env),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_FALSE(
      CreateQwenOmniResources(config, "/tmp/nonexistent_qwen_omni.litertlm", "",
                              lm::Backend::CPU, /*num_threads=*/1, shared_env)
          .ok());
}

TEST(QwenOmniTest, CreateQwenOmniComponentsRejectsNullAndMissingModels) {
  // Arrange
  QwenOmniModelConfig config;
  std::vector<std::unique_ptr<internal::StageBase>> stages;

  // Act & Assert
  std::shared_ptr<ModelResources> null_resources;
  EXPECT_THAT(
      CreateQwenOmniComponents(config, std::make_unique<PushInputSource>(),
                               null_resources, stages),
      StatusIs(absl::StatusCode::kInvalidArgument));

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  EXPECT_THAT(
      CreateQwenOmniComponents(config, std::make_unique<PushInputSource>(),
                               resources, stages),
      StatusIs(absl::StatusCode::kNotFound));

  std::string emb_buf = BuildMinimalTfliteModelBuffer(
      {{"token_ids", {1, 1}, tflite::TensorType_INT32}},
      {{"embeddings", {1, 1, 4}, tflite::TensorType_FLOAT32}});
  ModelOptions opts{.backend = lm::Backend::CPU, .num_threads = 1};
  ASSERT_OK_AND_ASSIGN(auto compiled_emb,
                       CreateCompiledModelFromBuffer(*shared_env, opts, emb_buf,
                                                     "text_embedder"));
  ASSERT_OK(resources->AddCompiledModel(
      "text_embedder",
      std::make_shared<CompiledModel>(std::move(compiled_emb))));
  EXPECT_THAT(
      CreateQwenOmniComponents(config, std::make_unique<PushInputSource>(),
                               resources, stages),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(QwenOmniTest,
     CreateQwenOmniResourcesAndBackboneDelegationFromLitertLmFile) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteTestQwenOmniLitertLmFile("qwen_omni_full_test.litertlm"));
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);
  config.force_fp32_on_gpu = true;

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));

  // Act
  ASSERT_OK_AND_ASSIGN(
      ModelResources resources,
      CreateQwenOmniResources(config, bundle_path, ":nocache", lm::Backend::CPU,
                              /*num_threads=*/1, shared_env));

  // Assert
  EXPECT_EQ(resources.environment(), shared_env);
  EXPECT_TRUE(resources.HasLmModelResources());
  EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
  EXPECT_TRUE(resources.HasCompiledModel("vision_encoder"));
  EXPECT_TRUE(resources.HasCompiledModel("audio_encoder"));
  EXPECT_TRUE(resources.HasCompiledModel("tts_projector"));
  EXPECT_TRUE(resources.HasCompiledModel("tts_decoder"));
  EXPECT_TRUE(resources.HasCompiledModel("audio_vocoder"));
  EXPECT_EQ(config.tts_text_bos_token_id, 15);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                       resources.GetLmRunner("llm_backbone"));
  lm::ModelResources* backbone_res = llm_runner->mutable_model_resources();
  ASSERT_NE(backbone_res, nullptr);

  using lm::proto::LlmMetadata;
  EXPECT_TRUE(
      backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_TRUE(backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_OK(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_OK(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
  ASSERT_OK_AND_ASSIGN(const lm::proto::Any2AnyMetadata* any2any_meta,
                       backbone_res->GetAny2AnyMetadata());
  EXPECT_EQ(any2any_meta->input_sample_rate_hz(), 16000);
  EXPECT_EQ(any2any_meta->output_sample_rate_hz(), 24000);
  EXPECT_TRUE(any2any_meta->any2any_model_type().has_qwen_omni());
  EXPECT_EQ(
      any2any_meta->any2any_model_type().qwen_omni().tts_text_bos_token_id(),
      15);
}

TEST(QwenOmniTest, CreateQwenOmniComponentsEndToEndTurnWithAudioAndText) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteTestQwenOmniLitertLmFile("qwen_omni_e2e_stages.litertlm"));
  QwenOmniModelConfig config =
      MakeSmallQwenOmniTestConfig(/*generate_audio=*/true);

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  ASSERT_OK_AND_ASSIGN(
      ModelResources model_resources,
      CreateQwenOmniResources(config, bundle_path, ":nocache", lm::Backend::CPU,
                              /*num_threads=*/1, std::move(shared_env)));
  auto resources = std::make_shared<ModelResources>(std::move(model_resources));
  config.llm_vocab_size = 28;
  config.tts_do_sample = false;

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  std::vector<std::unique_ptr<internal::StageBase>> stages;

  // Act
  ASSERT_OK_AND_ASSIGN(Stage<Output> * output_stage,
                       CreateQwenOmniComponents(config, std::move(input_source),
                                                resources, stages));
  ASSERT_EQ(output_stage, stages.back().get());
  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), output_stage));

  // Push ImageInput (4x4 RGB), TextInput, AudioInput (1600 samples), and
  // EndOfInputTurn. Use "Hello t15" so the deterministic checksum fake LM
  // produces a non-stop token ID within [0, llm_vocab_size).
  ASSERT_OK(raw_input->PushInput(OmniSession::ImageInput{
      .image_bytes = std::vector<uint8_t>(4 * 4 * 3, 128),
      .width = 4,
      .height = 4,
      .channels = 3,
  }));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Hello t15"}));
  std::vector<float> pcm_samples(1600, 0.1f);
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = std::move(pcm_samples)}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  raw_input->Finish();

  // Assert
  bool saw_text = false;
  bool saw_audio = false;
  bool saw_end_of_turn = false;
  while (true) {
    absl::StatusOr<OmniSession::Output> out = session->ProcessNext();
    if (absl::IsOutOfRange(out.status())) {
      break;
    }
    ASSERT_OK(out.status());
    if (std::holds_alternative<TextOutput>(*out)) {
      saw_text = true;
    } else if (std::holds_alternative<AudioOutput>(*out)) {
      saw_audio = true;
    } else if (std::holds_alternative<OmniSession::EndOfOutputTurn>(*out)) {
      saw_end_of_turn = true;
      break;
    }
  }
  EXPECT_TRUE(saw_text);
  EXPECT_TRUE(saw_audio);
  EXPECT_TRUE(saw_end_of_turn);
}

TEST(QwenOmniTest, OmniRegistryRegistersQwen25OmniNames) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteTestQwenOmniLitertLmFile(
          "qwen_omni_registry.litertlm",
          {.include_audio_encoder = false, .include_tts_and_vocoder = false}));
  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig session_cfg;
  session_cfg.set_generate_audio(false);
  session_cfg.set_voice("Ethan");

  // Act & Assert
  for (absl::string_view name :
       {"qwen2.5-omni-3b", "qwen2.5-omni", "Qwen/Qwen2.5-Omni-3B"}) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OmniSessionFactory> factory,
        OmniRegistry::Get().CreateSessionFactory(name, settings, &session_cfg,
                                                 /*metadata=*/nullptr));
    ASSERT_NE(factory, nullptr);
    auto* any2any_factory = dynamic_cast<Any2AnySessionFactory*>(factory.get());
    ASSERT_NE(any2any_factory, nullptr);
    EXPECT_THAT(any2any_factory->GetAvailableLanguages(), ElementsAre("en-US"));
  }
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("qwen3-omni"));
}

TEST(QwenOmniTest, PopulateQwenOmniConfigFromProtoMapsVoiceToBosAndClampsIds) {
  // Default voice is Chelsie with Chelsie BOS token ID.
  QwenOmniModelConfig default_config;
  EXPECT_EQ(default_config.voice_name, "Chelsie");
  EXPECT_EQ(default_config.tts_text_bos_token_id,
            qwen_omni::kChelsieTextBosTokenId);

  // Voice "Ethan" (case-insensitive) maps to kEthanTextBosTokenId.
  {
    QwenOmniModelConfig config;
    lm::proto::Any2AnyMetadata meta;
    meta.set_input_sample_rate_hz(48000);
    meta.set_output_sample_rate_hz(48000);
    meta.set_system_prompt("Custom metadata prompt");
    auto* qwen = meta.mutable_any2any_model_type()->mutable_qwen_omni();
    qwen->set_llm_hidden_size(1024);
    qwen->set_llm_vocab_size(qwen_omni::kLlmVocabSize);
    qwen->set_tts_hidden_size(512);
    qwen->set_tts_vocab_size(qwen_omni::kTtsVocabSize);
    qwen->set_tts_num_mel_bins(128);
    qwen->set_tts_num_codebooks(2);
    qwen->set_vision_image_size(336);
    qwen->set_vision_patch_size(16);
    qwen->set_vision_spatial_merge_size(4);
    qwen->set_audio_num_mel_bins(80);
    qwen->set_audio_chunk_frames(100);
    qwen->set_audio_pool_step(4);
    qwen->set_generate_audio(true);
    qwen->set_max_new_tokens(128);
    qwen->set_max_tts_tokens(256);
    qwen->set_voice("eThAn");
    qwen->set_language("zh-CN");
    qwen->set_is_moe(true);

    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.audio_input_sample_rate_hz, 48000);
    EXPECT_EQ(config.audio_output_sample_rate_hz, 48000);
    EXPECT_EQ(config.system_prompt, "Custom metadata prompt");
    EXPECT_EQ(config.llm_hidden_size, 1024);
    EXPECT_EQ(config.tts_hidden_size, 512);
    EXPECT_EQ(config.tts_num_mel_bins, 128);
    EXPECT_EQ(config.tts_num_codebooks, 2);
    EXPECT_EQ(config.vision_image_size, 336);
    EXPECT_EQ(config.vision_patch_size, 16);
    EXPECT_EQ(config.vision_spatial_merge_size, 4);
    EXPECT_EQ(config.audio_num_mel_bins, 80);
    EXPECT_EQ(config.audio_chunk_frames, 100);
    EXPECT_EQ(config.audio_pool_step, 4);
    EXPECT_TRUE(config.generate_audio);
    EXPECT_EQ(config.max_new_tokens, 128);
    EXPECT_EQ(config.max_tts_tokens, 256);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);
    EXPECT_EQ(config.language, "zh-CN");
    EXPECT_TRUE(config.is_moe);

    // Sentinel "default" (case-insensitive) leaves voice_name and BOS token ID
    // unchanged.
    qwen->set_voice("default");
    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    qwen->set_voice("DEFAULT");
    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    // Switching back to Chelsie (case-insensitive) canonicalizes name and
    // updates BOS token ID.
    qwen->set_voice("CHELSIE");
    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.voice_name, "Chelsie");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kChelsieTextBosTokenId);

    // Explicit proto tts_text_bos_token_id takes precedence over voice mapping.
    qwen->set_voice("Ethan");
    qwen->set_tts_text_bos_token_id(4242);
    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, 4242);
  }

  // Unknown/custom voice preserves previous tts_text_bos_token_id.
  {
    QwenOmniModelConfig config;
    config.tts_text_bos_token_id = qwen_omni::kEthanTextBosTokenId;
    lm::proto::Any2AnyMetadata meta;
    meta.mutable_any2any_model_type()->mutable_qwen_omni()->set_voice(
        "CustomVoice");
    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.voice_name, "CustomVoice");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);
  }

  // Small vocab sizes clamp out-of-range special token IDs safely.
  {
    QwenOmniModelConfig config;
    lm::proto::Any2AnyMetadata meta;
    auto* qwen = meta.mutable_any2any_model_type()->mutable_qwen_omni();
    qwen->set_llm_vocab_size(32);
    qwen->set_tts_vocab_size(8);
    qwen->set_voice("Ethan");
    PopulateQwenOmniConfigFromProto(meta, config);
    EXPECT_EQ(config.tts_text_pad_token_id, 0);
    EXPECT_EQ(config.tts_text_bos_token_id, 0);
    EXPECT_EQ(config.tts_text_eos_token_id, 0);
    EXPECT_EQ(config.tts_silence_token_id, 5);
    EXPECT_EQ(config.tts_audio_bos_token_id, 6);
    EXPECT_EQ(config.tts_eos_token_id, 7);
  }
}

TEST(QwenOmniTest,
     ApplySessionConfigToQwenOmniConfigMapsVoiceToBosAndRespectsVocabGuard) {
  // Full vocab size: voice overrides directly update tts_text_bos_token_id.
  {
    QwenOmniModelConfig config;
    ASSERT_GT(config.llm_vocab_size, qwen_omni::kChelsieTextBosTokenId);
    ASSERT_GT(config.llm_vocab_size, qwen_omni::kEthanTextBosTokenId);

    lm::proto::Any2AnySessionConfig ethan_cfg;
    ethan_cfg.set_voice("Ethan");
    ApplySessionConfigToQwenOmniConfig(ethan_cfg, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    lm::proto::Any2AnySessionConfig chelsie_cfg;
    chelsie_cfg.set_voice("cHeLsIe");
    ApplySessionConfigToQwenOmniConfig(chelsie_cfg, config);
    EXPECT_EQ(config.voice_name, "Chelsie");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kChelsieTextBosTokenId);

    lm::proto::Any2AnySessionConfig lower_ethan_cfg;
    lower_ethan_cfg.set_voice("ethan");
    ApplySessionConfigToQwenOmniConfig(lower_ethan_cfg, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    // Sentinel "default" (case-insensitive) leaves voice_name and BOS token ID
    // unchanged.
    lm::proto::Any2AnySessionConfig default_voice_cfg;
    default_voice_cfg.set_voice("default");
    ApplySessionConfigToQwenOmniConfig(default_voice_cfg, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    default_voice_cfg.set_voice("DEFAULT");
    ApplySessionConfigToQwenOmniConfig(default_voice_cfg, config);
    EXPECT_EQ(config.voice_name, "Ethan");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    // Unknown voice updates voice_name without clobbering existing BOS token.
    lm::proto::Any2AnySessionConfig unknown_cfg;
    unknown_cfg.set_voice("Narrator");
    ApplySessionConfigToQwenOmniConfig(unknown_cfg, config);
    EXPECT_EQ(config.voice_name, "Narrator");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);

    // Empty voice leaves both voice_name and tts_text_bos_token_id unchanged.
    lm::proto::Any2AnySessionConfig empty_voice_cfg;
    empty_voice_cfg.set_voice("");
    ApplySessionConfigToQwenOmniConfig(empty_voice_cfg, config);
    EXPECT_EQ(config.voice_name, "Narrator");
    EXPECT_EQ(config.tts_text_bos_token_id, qwen_omni::kEthanTextBosTokenId);
  }

  // Small vocab size: voice name updates, but out-of-range BOS IDs are guarded.
  {
    QwenOmniModelConfig small_vocab_config;
    small_vocab_config.llm_vocab_size = 32;
    small_vocab_config.tts_text_bos_token_id = 15;

    lm::proto::Any2AnySessionConfig ethan_cfg;
    ethan_cfg.set_voice("Ethan");
    ApplySessionConfigToQwenOmniConfig(ethan_cfg, small_vocab_config);
    EXPECT_EQ(small_vocab_config.voice_name, "Ethan");
    EXPECT_EQ(small_vocab_config.tts_text_bos_token_id, 15);

    lm::proto::Any2AnySessionConfig chelsie_cfg;
    chelsie_cfg.set_voice("Chelsie");
    ApplySessionConfigToQwenOmniConfig(chelsie_cfg, small_vocab_config);
    EXPECT_EQ(small_vocab_config.voice_name, "Chelsie");
    EXPECT_EQ(small_vocab_config.tts_text_bos_token_id, 15);
  }

  // Sampling and generation fields apply cleanly.
  {
    QwenOmniModelConfig config;
    lm::proto::Any2AnySessionConfig session_cfg;
    session_cfg.set_generate_audio(true);
    session_cfg.set_max_new_tokens(64);
    session_cfg.set_max_tts_tokens(128);
    session_cfg.set_temperature(0.5f);
    session_cfg.set_top_p(0.95f);
    session_cfg.set_top_k(30);
    session_cfg.set_tts_temperature(0.0f);
    session_cfg.set_tts_top_p(0.75f);
    session_cfg.set_tts_top_k(25);
    session_cfg.set_system_prompt("Session prompt");
    session_cfg.set_language("ja-JP");
    session_cfg.set_seed(12345);

    ApplySessionConfigToQwenOmniConfig(session_cfg, config);
    EXPECT_TRUE(config.generate_audio);
    EXPECT_EQ(config.max_new_tokens, 64);
    EXPECT_EQ(config.max_tts_tokens, 128);
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_TRUE(config.do_sample);
    EXPECT_FLOAT_EQ(config.top_p, 0.95f);
    EXPECT_EQ(config.top_k, 30);
    EXPECT_FLOAT_EQ(config.tts_temperature, 0.0f);
    EXPECT_FALSE(config.tts_do_sample);
    EXPECT_FLOAT_EQ(config.tts_top_p, 0.75f);
    EXPECT_EQ(config.tts_top_k, 25);
    EXPECT_EQ(config.system_prompt, "Session prompt");
    EXPECT_EQ(config.language, "ja-JP");
    EXPECT_THAT(config.seed, Optional(12345u));

    lm::proto::Any2AnySessionConfig greedy_cfg;
    greedy_cfg.set_temperature(0.0f);
    greedy_cfg.set_tts_temperature(0.8f);
    ApplySessionConfigToQwenOmniConfig(greedy_cfg, config);
    EXPECT_FALSE(config.do_sample);
    EXPECT_TRUE(config.tts_do_sample);
  }
}

}  // namespace
}  // namespace litert::omni::any2any
