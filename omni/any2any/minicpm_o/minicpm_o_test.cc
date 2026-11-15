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

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_factory.h"
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_vocoder_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/mock_litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "support/tokenizer/huggingface_tokenizer.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Optional;
using ::testing::Return;

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
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
      "t6": 6, "t7": 7, "t8": 8, "t9": 9, "t10": 10, "t11": 11, "t12": 12,
      "t13": 13, "t14": 14, "t15": 15, "t16": 16, "t17": 17, "t18": 18,
      "t19": 19, "t20": 20, "t21": 21, "t22": 22, "t23": 23, "t24": 24,
      "t25": 25, "t26": 26, "t27": 27, "t28": 28, "t29": 29, "t30": 30,
      "t31": 31
    },
    "unk_token": "<|endoftext|>"
  }
})json";

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/testdata";

struct TensorSpec {
  std::string name;
  std::vector<int32_t> dims;
  tflite::TensorType type;
};

std::string BuildMinimalTfliteModelBuffer(
    absl::Span<const TensorSpec> inputs, absl::Span<const TensorSpec> outputs) {
  flatbuffers::FlatBufferBuilder builder;

  auto opcode =
      tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector({opcode});

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<int32_t> sg_inputs;
  std::vector<int32_t> sg_outputs;
  std::vector<flatbuffers::Offset<tflite::TensorMap>> inputs_map;
  std::vector<flatbuffers::Offset<tflite::TensorMap>> outputs_map;
  std::vector<flatbuffers::Offset<tflite::Operator>> ops;

  for (const auto& in : inputs) {
    int32_t idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(in.dims), in.type,
                             /*buffer=*/0, builder.CreateString(in.name)));
    sg_inputs.push_back(idx);
    inputs_map.push_back(
        tflite::CreateTensorMap(builder, builder.CreateString(in.name), idx));
  }

  for (size_t j = 0; j < outputs.size(); ++j) {
    const auto& out = outputs[j];
    size_t num_elems = 1;
    for (int32_t d : out.dims) {
      num_elems *= static_cast<size_t>(d);
    }
    std::vector<uint8_t> zero_bytes(num_elems * sizeof(float), 0);
    uint32_t buf_a_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(zero_bytes)));
    uint32_t buf_b_idx = static_cast<uint32_t>(buffers.size());
    buffers.push_back(
        tflite::CreateBuffer(builder, builder.CreateVector(zero_bytes)));

    int32_t const_a_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_a_idx,
        builder.CreateString(absl::StrCat("const_a_", j))));
    int32_t const_b_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(out.dims), out.type, buf_b_idx,
        builder.CreateString(absl::StrCat("const_b_", j))));
    int32_t out_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(
        tflite::CreateTensor(builder, builder.CreateVector(out.dims), out.type,
                             /*buffer=*/0, builder.CreateString(out.name)));
    sg_outputs.push_back(out_idx);
    outputs_map.push_back(tflite::CreateTensorMap(
        builder, builder.CreateString(out.name), out_idx));

    std::vector<int32_t> op_inputs = {const_a_idx, const_b_idx};
    std::vector<int32_t> op_outputs = {out_idx};
    ops.push_back(tflite::CreateOperator(builder, /*opcode_index=*/0,
                                         builder.CreateVector(op_inputs),
                                         builder.CreateVector(op_outputs)));
  }

  auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(sg_inputs),
      builder.CreateVector(sg_outputs), builder.CreateVector(ops),
      builder.CreateString("main"));
  auto sig_def =
      tflite::CreateSignatureDef(builder, builder.CreateVector(inputs_map),
                                 builder.CreateVector(outputs_map),
                                 builder.CreateString("serving_default"),
                                 /*subgraph_index=*/0);
  auto model = tflite::CreateModel(
      builder, /*version=*/3, opcodes_vec, builder.CreateVector({subgraph}),
      builder.CreateString("test_model"), builder.CreateVector(buffers),
      /*metadata_buffer=*/0, /*metadata=*/0, builder.CreateVector({sig_def}));
  tflite::FinishModelBuffer(builder, model);

  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

absl::StatusOr<std::string> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return absl::InternalError(absl::StrCat("Failed to open: ", path.string()));
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

MiniCpmOModelConfig MakeSmallTestConfig(bool generate_audio = false) {
  MiniCpmOModelConfig config;
  config.generate_audio = generate_audio;
  config.llm_hidden_size = 4;
  // checksum_fake_lm_embedding_input.tflite outputs 32 floats per decode step.
  // When generate_audio is true, MiniCpmOLlmStage splits the 32-float output
  // into [llm_vocab_size (28)] logits + [llm_hidden_size (4)] hidden states.
  config.llm_vocab_size = generate_audio ? 28 : 32;
  config.max_new_tokens = 1;
  config.do_sample = false;
  config.tts_hidden_size = 4;
  config.tts_vocab_size = 8;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 8;
  config.max_tts_tokens = 4;
  config.tts_temperature = 0.0f;
  config.vision_patch_size = 2;
  config.vocoder_history_tokens = 1;
  config.vocoder_chunk_tokens = 2;
  config.vocoder_lookahead_tokens = 1;
  config.vocoder_samples_per_token = 4;
  return config;
}

struct TestBundleOptions {
  bool include_vision_encoder = true;
  bool include_audio_encoder = true;
  bool include_tts_and_vocoder = true;
  bool use_fallback_vision_input_names = false;
};

absl::StatusOr<std::string> WriteTestMiniCpmOLitertLmFile(
    absl::string_view filename, const TestBundleOptions& opts = {}) {
  const auto testdata_dir =
      std::filesystem::path(::testing::SrcDir()) / std::string(kTestdataDir);
  ABSL_ASSIGN_OR_RETURN(
      std::string llm_backbone_bytes,
      ReadFileBytes(testdata_dir / "checksum_fake_lm_embedding_input.tflite"));
  ABSL_ASSIGN_OR_RETURN(
      std::string text_embedder_bytes,
      ReadFileBytes(testdata_dir / "checksum_fake_lm_embedder.tflite"));

  lm::schema::ZlibBackendedSectionStream zlib_tok_stream(
      std::make_unique<lm::schema::StringBackedSectionStream>(
          std::string(kMinimalTokenizerJson)));
  ABSL_RETURN_IF_ERROR(zlib_tok_stream.Prepare());
  std::stringstream tok_ss;
  tok_ss << zlib_tok_stream.GetStream().rdbuf();
  std::string compressed_tokenizer_bytes = tok_ss.str();

  lm::proto::Any2AnyMetadata any2any_meta;
  any2any_meta.add_supported_languages("en");
  any2any_meta.set_input_sample_rate_hz(16000);
  std::string any2any_meta_bytes = any2any_meta.SerializeAsString();

  struct SectionPayload {
    lm::schema::AnySectionDataType data_type;
    std::string model_type;
    std::string backend_constraint;
    std::string prefer_activation_type;
    std::string bytes;
  };
  std::vector<SectionPayload> payloads;
  payloads.push_back({lm::schema::AnySectionDataType_Any2AnyMetadataProto, "",
                      "", "", std::move(any2any_meta_bytes)});
  payloads.push_back({lm::schema::AnySectionDataType_HF_Tokenizer_Zlib, "", "",
                      "", std::move(compressed_tokenizer_bytes)});
  payloads.push_back({lm::schema::AnySectionDataType_TFLiteModel,
                      "tf_lite_text_embedder", "cpu", "fp32",
                      std::move(text_embedder_bytes)});
  payloads.push_back({lm::schema::AnySectionDataType_TFLiteModel,
                      "tf_lite_llm_backbone", "cpu", "fp32",
                      std::move(llm_backbone_bytes)});

  if (opts.include_vision_encoder) {
    const std::string img_name =
        opts.use_fallback_vision_input_names ? "pixel_values" : "images";
    const std::string pos_name =
        opts.use_fallback_vision_input_names ? "position_ids" : "positions_xy";
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_vision_encoder",
         "", "",
         BuildMinimalTfliteModelBuffer(
             {{img_name, {1, 4, 12}, tflite::TensorType_FLOAT32},
              {pos_name, {1, 4, 2}, tflite::TensorType_INT32},
              {"vit_positions", {1, 4}, tflite::TensorType_INT32}},
             {{"features", {1, 2, 4}, tflite::TensorType_FLOAT32}})});
  }
  if (opts.include_audio_encoder) {
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_audio_encoder",
         "", "",
         BuildMinimalTfliteModelBuffer(
             {{"mel", {1, 80, 10}, tflite::TensorType_FLOAT32}},
             {{"features", {1, 2, 4}, tflite::TensorType_FLOAT32}})});
  }
  if (opts.include_tts_and_vocoder) {
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_tts_projector",
         "", "",
         BuildMinimalTfliteModelBuffer(
             {{"llm_hidden", {1, 4}, tflite::TensorType_FLOAT32},
              {"token_id", {1}, tflite::TensorType_INT32}},
             {{"cond_emb", {1, 4}, tflite::TensorType_FLOAT32}})});
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_tts_decoder", "",
         "",
         BuildMinimalTfliteModelBuffer(
             {{"cond_emb", {1, 4}, tflite::TensorType_FLOAT32},
              {"prev_audio_code", {1}, tflite::TensorType_INT32},
              {"step_idx", {1}, tflite::TensorType_INT32},
              {"kv_cache", {2, 1, 8, 4}, tflite::TensorType_FLOAT32}},
             {{"code_logits", {1, 8}, tflite::TensorType_FLOAT32},
              {"new_kv_cache", {2, 1, 8, 4}, tflite::TensorType_FLOAT32}})});
    payloads.push_back(
        {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_audio_vocoder",
         "", "",
         BuildMinimalTfliteModelBuffer(
             {{"codes", {1, 4}, tflite::TensorType_INT32}},
             {{"waveform", {1, 12}, tflite::TensorType_FLOAT32}})});
  }

  // Reserve a 16KB padded header region so section offsets are known upfront.
  constexpr uint64_t kDataStartOffset = 16384;
  flatbuffers::FlatBufferBuilder builder(2048);
  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> section_objects;
  uint64_t cursor = kDataStartOffset;
  for (const auto& p : payloads) {
    std::vector<flatbuffers::Offset<lm::schema::KeyValuePair>> items;
    if (!p.model_type.empty()) {
      items.push_back(
          lm::schema::CreateKeyValuePair(builder, "model_type", p.model_type));
    }
    if (!p.backend_constraint.empty()) {
      items.push_back(lm::schema::CreateKeyValuePair(
          builder, "backend_constraint", p.backend_constraint));
    }
    if (!p.prefer_activation_type.empty()) {
      items.push_back(lm::schema::CreateKeyValuePair(
          builder, "prefer_activation_type", p.prefer_activation_type));
    }
    auto items_vec = items.empty() ? 0 : builder.CreateVector(items);
    const uint64_t begin = cursor;
    const uint64_t end = begin + p.bytes.size();
    section_objects.push_back(lm::schema::CreateSectionObject(
        builder, items_vec, begin, end, p.data_type));
    cursor = end;
  }
  auto section_metadata = lm::schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objects));
  auto metadata =
      lm::schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  const auto out_path =
      std::filesystem::path(::testing::TempDir()) / std::string(filename);
  std::ofstream file(out_path, std::ios::binary);
  if (!file.good()) {
    return absl::InternalError("Failed to open test .litertlm file.");
  }
  file.write("LITERTLM", 8);
  const uint32_t major_version = 1;
  const uint32_t minor_version = 0;
  const uint32_t patch_version = 0;
  const uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), 4);
  file.write(reinterpret_cast<const char*>(&minor_version), 4);
  file.write(reinterpret_cast<const char*>(&patch_version), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);
  const uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  std::string header_pad(kDataStartOffset - header_end_offset, '\0');
  file.write(header_pad.data(), header_pad.size());
  for (const auto& p : payloads) {
    file.write(p.bytes.data(), p.bytes.size());
  }
  if (!file.good()) {
    return absl::InternalError("Failed to write test .litertlm file.");
  }
  return out_path.string();
}

class FakeFrontendStage
    : public SingleThreadedStageWithDeque<MiniCpmOFrontendOutput> {
 public:
  void Push(MiniCpmOFrontendOutput prompt) { PushOutput(std::move(prompt)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

class FakeLlmStage : public SingleThreadedStageWithDeque<MiniCpmOLlmOutput> {
 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

class FakeTtsStage : public SingleThreadedStageWithDeque<MiniCpmOTtsOutput> {
 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TEST(MiniCpmOTest, InitMiniCpmOResourcesFailsWithNonexistentModels) {
  MiniCpmOModelConfig config;
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  ModelResources resources(shared_env);

  EXPECT_THAT(
      InitMiniCpmOResources(config, "/tmp/nonexistent_minicpm_o_dir", "",
                            lm::Backend::CPU, 1, *shared_env, resources),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_FALSE(
      InitMiniCpmOResources(config, "/tmp/nonexistent_minicpm_o.litertlm", "",
                            lm::Backend::CPU, 1, *shared_env, resources)
          .ok());
}

TEST(MiniCpmOTest, CreateMiniCpmOComponentsRejectsNullAndMissingModels) {
  MiniCpmOModelConfig config;
  std::vector<std::unique_ptr<internal::StageBase>> stages;

  // Null ModelResources is rejected with kInvalidArgument.
  std::shared_ptr<ModelResources> null_resources;
  EXPECT_THAT(
      CreateMiniCpmOComponents(config, std::make_unique<PushInputSource>(),
                               null_resources, stages),
      StatusIs(absl::StatusCode::kInvalidArgument));

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  // Missing text_embedder is rejected with kNotFound.
  EXPECT_THAT(
      CreateMiniCpmOComponents(config, std::make_unique<PushInputSource>(),
                               resources, stages),
      StatusIs(absl::StatusCode::kNotFound));

  // Present text_embedder without LmModelResources is rejected with
  // kFailedPrecondition.
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
      CreateMiniCpmOComponents(config, std::make_unique<PushInputSource>(),
                               resources, stages),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MiniCpmOTest, InitMiniCpmOResourcesAndBackboneDelegationFromLitertLmFile) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteTestMiniCpmOLitertLmFile("minicpm_o_full_test.litertlm"));
  MiniCpmOModelConfig config = MakeSmallTestConfig(/*generate_audio=*/true);
  config.force_fp32_on_gpu = true;

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  ModelResources resources(shared_env);

  // Act
  ASSERT_OK(InitMiniCpmOResources(config, bundle_path, ":nocache",
                                  lm::Backend::CPU, /*num_threads=*/1,
                                  *shared_env, resources));

  // Assert
  EXPECT_TRUE(resources.HasLmModelResources());
  EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
  EXPECT_TRUE(resources.HasCompiledModel("vision_encoder"));
  EXPECT_TRUE(resources.HasCompiledModel("audio_encoder"));
  EXPECT_TRUE(resources.HasCompiledModel("tts_projector"));
  EXPECT_TRUE(resources.HasCompiledModel("tts_decoder"));
  EXPECT_TRUE(resources.HasCompiledModel("audio_vocoder"));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                       resources.GetLmRunner("llm_backbone"));
  lm::ModelResources* backbone_res = llm_runner->mutable_model_resources();
  ASSERT_NE(backbone_res, nullptr);

  // Verify BackboneLmModelResources delegation across
  // LlmMetadata::TfLiteModelType and absl::string_view overloads (both mapped
  // and unmapped).
  using lm::proto::LlmMetadata;
  EXPECT_TRUE(
      backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_TRUE(backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_FALSE(backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_AUX));
  EXPECT_TRUE(backbone_res->HasTFLiteModel("tf_lite_prefill_decode"));
  EXPECT_TRUE(backbone_res->HasTFLiteModel("tf_lite_embedder"));
  EXPECT_TRUE(backbone_res->HasTFLiteModel("tf_lite_vision_encoder"));
  EXPECT_FALSE(backbone_res->HasTFLiteModel("tf_lite_aux"));

  EXPECT_OK(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_OK(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_THAT(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_AUX),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_OK(backbone_res->GetTFLiteModel("tf_lite_prefill_decode"));
  EXPECT_OK(backbone_res->GetTFLiteModel("tf_lite_vision_encoder"));
  EXPECT_THAT(backbone_res->GetTFLiteModel("tf_lite_aux"),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_OK(
      backbone_res->GetTFLiteModelBuffer(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_OK(backbone_res->GetTFLiteModelBuffer(LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_THAT(backbone_res->GetTFLiteModelBuffer(LlmMetadata::TF_LITE_AUX),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_OK(backbone_res->GetTFLiteModelBuffer("tf_lite_prefill_decode"));
  EXPECT_OK(backbone_res->GetTFLiteModelBuffer("tf_lite_vision_encoder"));
  EXPECT_THAT(backbone_res->GetTFLiteModelBuffer("tf_lite_aux"),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(backbone_res->GetWeightsSectionOffset(
                  LlmMetadata::TF_LITE_PREFILL_DECODE),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(backbone_res->GetWeightsSectionOffset(LlmMetadata::TF_LITE_AUX),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(backbone_res->GetWeightsSectionOffset("tf_lite_prefill_decode"),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_OK(backbone_res->GetTFLiteModelSectionFileRegion(
      LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_OK(backbone_res->GetTFLiteModelSectionFileRegion(
      LlmMetadata::TF_LITE_EMBEDDER));
  EXPECT_THAT(
      backbone_res->GetTFLiteModelSectionFileRegion(LlmMetadata::TF_LITE_AUX),
      StatusIs(absl::StatusCode::kNotFound));
  EXPECT_OK(
      backbone_res->GetTFLiteModelSectionFileRegion("tf_lite_vision_encoder"));

  EXPECT_THAT(backbone_res->GetTFLiteModelBackendConstraint(
                  LlmMetadata::TF_LITE_PREFILL_DECODE),
              Optional(std::string("cpu")));
  EXPECT_EQ(
      backbone_res->GetTFLiteModelBackendConstraint(LlmMetadata::TF_LITE_AUX),
      std::nullopt);
  EXPECT_THAT(
      backbone_res->GetTFLiteModelBackendConstraint("tf_lite_prefill_decode"),
      Optional(std::string("cpu")));

  EXPECT_THAT(backbone_res->GetTFLiteModelPreferActivationType(
                  LlmMetadata::TF_LITE_PREFILL_DECODE),
              Optional(std::string("fp32")));
  EXPECT_EQ(backbone_res->GetTFLiteModelPreferActivationType(
                LlmMetadata::TF_LITE_AUX),
            std::nullopt);
  EXPECT_THAT(backbone_res->GetTFLiteModelPreferActivationType(
                  "tf_lite_prefill_decode"),
              Optional(std::string("fp32")));

  EXPECT_EQ(
      backbone_res->GetWeightInMemoryMap(LlmMetadata::TF_LITE_PREFILL_DECODE),
      nullptr);
  EXPECT_EQ(backbone_res->GetWeightInMemoryMap(LlmMetadata::TF_LITE_AUX),
            nullptr);
  EXPECT_EQ(backbone_res->GetWeightInMemoryMap("tf_lite_vision_encoder"),
            nullptr);

  EXPECT_OK(backbone_res->GetScopedFile());
  EXPECT_OK(backbone_res->GetTokenizer());
  EXPECT_THAT(backbone_res->GetLlmMetadata(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(backbone_res->GetExecutorMetadata(),
              StatusIs(absl::StatusCode::kNotFound));
  ASSERT_OK_AND_ASSIGN(const lm::proto::Any2AnyMetadata* any2any_meta,
                       backbone_res->GetAny2AnyMetadata());
  EXPECT_EQ(any2any_meta->input_sample_rate_hz(), 16000);
}

TEST(MiniCpmOTest, CreateMiniCpmOComponentsEndToEndTurnWithAllStages) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteTestMiniCpmOLitertLmFile("minicpm_o_e2e_stages.litertlm"));
  MiniCpmOModelConfig config = MakeSmallTestConfig(/*generate_audio=*/true);

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  ASSERT_OK(InitMiniCpmOResources(config, bundle_path, ":nocache",
                                  lm::Backend::CPU, /*num_threads=*/1,
                                  *shared_env, *resources));

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  std::vector<std::unique_ptr<internal::StageBase>> stages;

  // Act
  ASSERT_OK_AND_ASSIGN(Stage<Output> * output_stage,
                       CreateMiniCpmOComponents(config, std::move(input_source),
                                                resources, stages));
  ASSERT_EQ(output_stage, stages.back().get());
  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), output_stage));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Hello"}));
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

TEST(MiniCpmOTest,
     CreateMiniCpmOComponentsWithoutAudioEncoderAndFallbackVisionInputs) {
  // Arrange
  TestBundleOptions bundle_opts{
      .include_vision_encoder = true,
      .include_audio_encoder = false,
      .include_tts_and_vocoder = false,
      .use_fallback_vision_input_names = true,
  };
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteTestMiniCpmOLitertLmFile("minicpm_o_no_audio_fallback_vis.litertlm",
                                    bundle_opts));
  MiniCpmOModelConfig config = MakeSmallTestConfig(/*generate_audio=*/false);

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);
  ASSERT_OK(InitMiniCpmOResources(config, bundle_path, ":nocache",
                                  lm::Backend::CPU, /*num_threads=*/1,
                                  *shared_env, *resources));
  EXPECT_FALSE(resources->HasCompiledModel("audio_encoder"));
  EXPECT_TRUE(resources->HasCompiledModel("vision_encoder"));

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  std::vector<std::unique_ptr<internal::StageBase>> stages;

  // Act
  ASSERT_OK_AND_ASSIGN(Stage<Output> * output_stage,
                       CreateMiniCpmOComponents(config, std::move(input_source),
                                                resources, stages));
  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), output_stage));

  // Pushing AudioInput when audio_encoder is absent fails with
  // kFailedPrecondition on Schedule without advancing seqnum.
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {0.1f, 0.2f}}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // A subsequent text turn still completes normally without hanging.
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
}

TEST(MiniCpmOTest, TtsAndVocoderStagesRejectMissingModelsWhenAudioEnabled) {
  MiniCpmOModelConfig config;
  config.generate_audio = true;

  FakeLlmStage llm_stage;
  EXPECT_THAT(MiniCpmOTtsStage::Create(&llm_stage, config).status(),
              StatusIs(absl::StatusCode::kNotFound));

  FakeTtsStage tts_stage;
  EXPECT_THAT(MiniCpmOVocoderStage::Create(&tts_stage, config).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(MiniCpmOTest, LlmTtsAndVocoderPipelineGeneratesTextWithMockLmRunner) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<support::HuggingFaceTokenizer> hf_tokenizer,
      support::HuggingFaceTokenizer::CreateFromJson(
          std::string(kMinimalTokenizerJson)));
  std::shared_ptr<support::Tokenizer> tokenizer(std::move(hf_tokenizer));

  MiniCpmOModelConfig config;
  config.generate_audio = false;
  config.llm_hidden_size = 8;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 4;
  config.do_sample = false;

  auto mock_runner = std::make_shared<MockLiteRtLmRunner>();
  EXPECT_CALL(*mock_runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_runner, Decode(_))
      .WillOnce([](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
        std::vector<float> logits(8, -10.0f);
        logits[3] = 10.0f;  // "Hello"
        auto buf = support::CreateTensorBuffer<float>({1, 8});
        (void)buf->Write<float>(absl::MakeConstSpan(logits));
        return std::move(*buf);
      })
      .WillOnce([](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
        std::vector<float> logits(8, -10.0f);
        logits[4] = 10.0f;  // "world"
        auto buf = support::CreateTensorBuffer<float>({1, 8});
        (void)buf->Write<float>(absl::MakeConstSpan(logits));
        return std::move(*buf);
      })
      .WillRepeatedly(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> logits(8, -10.0f);
            logits[2] = 10.0f;  // "<|im_end|>" stop token + turn-end commits
            auto buf = support::CreateTensorBuffer<float>({1, 8});
            (void)buf->Write<float>(absl::MakeConstSpan(logits));
            return std::move(*buf);
          });

  auto frontend = std::make_unique<FakeFrontendStage>();
  FakeFrontendStage* raw_frontend = frontend.get();
  ASSERT_OK_AND_ASSIGN(
      auto llm,
      MiniCpmOLlmStage::Create(raw_frontend, tokenizer, mock_runner, config));
  ASSERT_OK_AND_ASSIGN(auto tts, MiniCpmOTtsStage::Create(llm.get(), config));
  ASSERT_OK_AND_ASSIGN(auto vocoder,
                       MiniCpmOVocoderStage::Create(tts.get(), config));
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(llm));
  stages.push_back(std::move(tts));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  MiniCpmOFrontendOutput prompt;
  prompt.prompt_embeddings.assign(2 * config.llm_hidden_size, 0.25f);
  prompt.prompt_len = 2;
  prompt.generate_audio = false;
  raw_frontend->Push(std::move(prompt));

  ASSERT_OK_AND_ASSIGN(Output out, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out));
  EXPECT_EQ(std::get<TextOutput>(out).confirmed_text, "Hello world");
}

TEST(MiniCpmOTest, LlmStagePrefillsMultiTokenPromptInSingleCall) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<support::HuggingFaceTokenizer> hf_tokenizer,
      support::HuggingFaceTokenizer::CreateFromJson(
          std::string(kMinimalTokenizerJson)));
  std::shared_ptr<support::Tokenizer> tokenizer(std::move(hf_tokenizer));

  MiniCpmOModelConfig config;
  config.generate_audio = false;
  config.llm_hidden_size = 8;
  config.llm_vocab_size = 8;
  config.max_new_tokens = 2;
  config.do_sample = false;

  auto mock_runner = std::make_shared<MockLiteRtLmRunner>();
  // Prompt has 4 tokens:
  // - Prefill 1: single 4-token prefill call
  // - Decode 1 (generation step 0): samples "Hello"
  // - Decode 2 (generation step 1): feeds "Hello" and samples "<|im_end|>"
  // - Decode 3 & 4 (turn-end commit): commits "<|im_end|>" and "\n"
  EXPECT_CALL(*mock_runner, Prefill(_)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_runner, Decode(_))
      .WillOnce([](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
        std::vector<float> logits(8, -10.0f);
        logits[3] = 10.0f;  // "Hello"
        auto buf = support::CreateTensorBuffer<float>({1, 8});
        (void)buf->Write<float>(absl::MakeConstSpan(logits));
        return std::move(*buf);
      })
      .WillRepeatedly(
          [](const lm::ExecutorInputs&) -> absl::StatusOr<TensorBuffer> {
            std::vector<float> logits(8, -10.0f);
            logits[2] = 10.0f;  // "<|im_end|>" stop token + turn-end commits
            auto buf = support::CreateTensorBuffer<float>({1, 8});
            (void)buf->Write<float>(absl::MakeConstSpan(logits));
            return std::move(*buf);
          });

  auto frontend = std::make_unique<FakeFrontendStage>();
  FakeFrontendStage* raw_frontend = frontend.get();
  ASSERT_OK_AND_ASSIGN(
      auto llm,
      MiniCpmOLlmStage::Create(raw_frontend, tokenizer, mock_runner, config));
  ASSERT_OK_AND_ASSIGN(auto tts, MiniCpmOTtsStage::Create(llm.get(), config));
  ASSERT_OK_AND_ASSIGN(auto vocoder,
                       MiniCpmOVocoderStage::Create(tts.get(), config));
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(llm));
  stages.push_back(std::move(tts));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  MiniCpmOFrontendOutput prompt;
  prompt.prompt_embeddings.assign(4 * config.llm_hidden_size, 0.25f);
  prompt.prompt_len = 4;
  prompt.generate_audio = false;
  raw_frontend->Push(std::move(prompt));

  ASSERT_OK_AND_ASSIGN(Output out, session->Flush());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out));
  EXPECT_EQ(std::get<TextOutput>(out).confirmed_text, "Hello");
}

}  // namespace
}  // namespace litert::omni::any2any
