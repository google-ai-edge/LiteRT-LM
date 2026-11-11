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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <numbers>
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
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/any2any_session.h"
#include "omni/any2any/lfm2_audio/lfm2_audio_factory.h"
#include "omni/any2any/lfm2_audio/lfm2_audio_model_config.h"
#include "omni/any2any/lfm2_audio/lfm2_audio_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::NiceMock;

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|startoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 6, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 7, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|pad|>": 0,
      "<|startoftext|>": 1,
      "t2": 2,
      "Hello": 3,
      "world": 4,
      "\n": 5,
      "<|im_start|>": 6,
      "<|im_end|>": 7,
      "t8": 8, "t9": 9, "t10": 10, "t11": 11, "t12": 12,
      "t13": 13, "t14": 14, "t15": 15, "t16": 16, "t17": 17, "t18": 18,
      "t19": 19, "t20": 20, "t21": 21, "t22": 22, "t23": 23, "t24": 24,
      "t25": 25, "t26": 26, "t27": 27, "t28": 28, "t29": 29, "t30": 30,
      "t31": 31
    },
    "unk_token": "<|pad|>"
  }
})json";

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/testdata";

class FakeTokenizer : public support::Tokenizer {
 public:
  explicit FakeTokenizer(int out_of_range_token_id = -1)
      : out_of_range_token_id_(out_of_range_token_id) {}

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    if (text == "OUT_OF_RANGE" && out_of_range_token_id_ != -1) {
      return std::vector<int>{out_of_range_token_id_};
    }
    if (text == "\n") {
      return std::vector<int>{5};
    }
    return std::vector<int>{3};
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<|pad|>") return 0;
    if (token == "<|startoftext|>") return 1;
    if (token == "<|im_start|>") return 6;
    if (token == "<|im_end|>") return 7;
    if (token == "<|audio_start|>") return 128;
    if (token == "<|text_end|>") return 130;
    return absl::NotFoundError("Unknown token");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool /*skip_special_tokens*/) override {
    std::string out;
    for (int id : token_ids) {
      if (id == 3) {
        absl::StrAppend(&out, "Hello");
      } else if (id == 4) {
        absl::StrAppend(&out, " world");
      }
    }
    return out;
  }

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  std::vector<std::string> GetTokens() const override { return {}; }

  int GetVocabSize() const override { return 256; }

 private:
  int out_of_range_token_id_;
};

class ScriptedLiteRtLmRunner : public LiteRtLmRunner {
 public:
  struct StepSpec {
    int top_token_id = 7;
    std::vector<float> hidden_state;
  };

  ScriptedLiteRtLmRunner(int vocab_size, int hidden_dim,
                         std::vector<StepSpec> steps)
      : vocab_size_(vocab_size),
        hidden_dim_(hidden_dim),
        steps_(std::move(steps)) {}

  absl::Status Prefill(const lm::ExecutorInputs& inputs) override {
    ++prefill_calls_;
    auto emb_ptr = inputs.GetProjectedAudioEmbeddingsPtr();
    if (!emb_ptr.ok() || *emb_ptr == nullptr) {
      return absl::InvalidArgumentError(
          "Expected continuous prompt embeddings in Prefill.");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<TensorBuffer> Decode(
      const lm::ExecutorInputs& inputs) override {
    const bool has_audio_emb = inputs.GetAudioDataPtr().ok();
    decode_used_audio_emb_.push_back(has_audio_emb);
    if (has_audio_emb) {
      ABSL_ASSIGN_OR_RETURN(const TensorBuffer* emb_buf,
                            inputs.GetProjectedAudioEmbeddingsPtr());
      std::vector<float> emb(hidden_dim_, 0.0f);
      LITERT_RETURN_IF_ERROR(
          const_cast<TensorBuffer*>(emb_buf)->Read<float>(absl::MakeSpan(emb)));
      received_audio_embeddings_.push_back(std::move(emb));
    } else {
      ABSL_ASSIGN_OR_RETURN(const TensorBuffer* tok_buf,
                            inputs.GetTextTokenIdsPtr());
      int32_t tok = -999;
      LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer*>(tok_buf)->Read<int32_t>(
          absl::MakeSpan(&tok, 1)));
      received_text_tokens_.push_back(tok);
    }

    StepSpec spec;
    if (step_cursor_ < steps_.size()) {
      spec = steps_[step_cursor_++];
    } else {
      spec.top_token_id = 7;
      spec.hidden_state.assign(hidden_dim_, 0.0f);
    }

    std::vector<float> out(vocab_size_ + hidden_dim_, -10.0f);
    if (spec.top_token_id >= 0 && spec.top_token_id < vocab_size_) {
      out[spec.top_token_id] = 10.0f;
    }
    for (int i = 0;
         i < hidden_dim_ && i < static_cast<int>(spec.hidden_state.size());
         ++i) {
      out[vocab_size_ + i] = spec.hidden_state[i];
    }

    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer buf,
        support::CreateTensorBuffer<float>({1, 1, vocab_size_ + hidden_dim_}));
    LITERT_RETURN_IF_ERROR(buf.Write<float>(absl::MakeConstSpan(out)));
    return buf;
  }

  absl::Status Reset() override {
    step_cursor_ = 0;
    return absl::OkStatus();
  }

  int prefill_calls() const { return prefill_calls_; }
  const std::vector<bool>& decode_used_audio_emb() const {
    return decode_used_audio_emb_;
  }
  const std::vector<int32_t>& received_text_tokens() const {
    return received_text_tokens_;
  }
  const std::vector<std::vector<float>>& received_audio_embeddings() const {
    return received_audio_embeddings_;
  }

 private:
  int vocab_size_;
  int hidden_dim_;
  std::vector<StepSpec> steps_;
  size_t step_cursor_ = 0;
  int prefill_calls_ = 0;
  std::vector<bool> decode_used_audio_emb_;
  std::vector<int32_t> received_text_tokens_;
  std::vector<std::vector<float>> received_audio_embeddings_;
};

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

absl::StatusOr<std::string> WriteLfm2AudioBundle(absl::string_view filename) {
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
  any2any_meta.set_input_sample_rate_hz(16000);
  any2any_meta.set_output_sample_rate_hz(24000);
  any2any_meta.set_system_prompt("Respond with interleaved text and audio.");
  auto* lfm2 = any2any_meta.mutable_any2any_model_type()->mutable_lfm2_audio();
  lfm2->set_llm_hidden_size(4);
  lfm2->set_llm_vocab_size(32);
  lfm2->set_tts_hidden_size(4);
  lfm2->set_tts_vocab_size(8);
  lfm2->set_tts_num_codebooks(8);
  lfm2->set_audio_mel_bins(128);
  lfm2->set_audio_subsampling_factor(8);
  lfm2->set_generate_audio(false);
  lfm2->set_max_new_tokens(1);
  lfm2->set_max_tts_tokens(1);
  lfm2->set_voice("lfm2_voice");
  lfm2->set_language("en-US");
  std::string any2any_meta_bytes = any2any_meta.SerializeAsString();

  struct SectionPayload {
    lm::schema::AnySectionDataType data_type;
    std::string model_type;
    std::string bytes;
  };
  std::vector<SectionPayload> payloads = {
      {lm::schema::AnySectionDataType_Any2AnyMetadataProto, "",
       std::move(any2any_meta_bytes)},
      {lm::schema::AnySectionDataType_HF_Tokenizer_Zlib, "",
       std::move(compressed_tokenizer_bytes)},
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_text_embedder",
       std::move(text_embedder_bytes)},
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_llm_backbone",
       std::move(llm_backbone_bytes)},
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_audio_encoder",
       BuildMinimalTfliteModelBuffer(
           {{"mel", {1, 128, 16}, tflite::TensorType_FLOAT32}},
           {{"features", {1, 2, 4}, tflite::TensorType_FLOAT32}})},
  };

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

TEST(Lfm2AudioStageTest,
     ExtractsNemoLogMelWithPerFeatureNormalizationAndRunsFastConformer) {
  // Arrange
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.audio_mel_bins = 128;
  config.audio_subsampling_factor = 8;

  // Verify default LFM2.5-Audio-1.5B architecture constants.
  EXPECT_EQ(lfm2_audio::kTtsHiddenSize, 1024);
  EXPECT_EQ(lfm2_audio::kTtsVocabSize, 2049);
  EXPECT_EQ(lfm2_audio::kTtsNumLayers, 6);
  EXPECT_EQ(lfm2_audio::kTtsNumCodebooks, 8);
  EXPECT_EQ(lfm2_audio::kDetokenizerUpsampleSamples, 1920);
  EXPECT_EQ(lfm2_audio::kBosTokenId, 1);
  EXPECT_EQ(lfm2_audio::kImStartTokenId, 6);
  EXPECT_EQ(lfm2_audio::kImEndTokenId, 7);
  EXPECT_EQ(lfm2_audio::kAudioStartTokenId, 128);
  EXPECT_EQ(lfm2_audio::kTextEndTokenId, 130);

  std::vector<float> pcm(1600);
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * 440.0f *
                             static_cast<float>(i) / 16000.0f) +
             0.2f * std::sin(2.0f * std::numbers::pi_v<float> * 880.0f *
                             static_cast<float>(i) / 16000.0f *
                             static_cast<float>(i + 1) / 800.0f);
  }

  int valid_frames = 0;
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> log_mel,
      ExtractNemoLogMelSpectrogram(pcm, config, &valid_frames));
  // NeMo FilterbankFeatures produces N // 160 + 1 = 11 frames (10 normalized
  // frames + 1 trailing zero-masked STFT frame).
  EXPECT_EQ(valid_frames, 11);
  ASSERT_EQ(log_mel.size(), 11u * 128u);

  // Verify per-feature mean is ~0 across the 10 normalized frames, and the 11th
  // frame is zero-masked.
  double m0_sum = 0.0;
  for (int t = 0; t < 10; ++t) {
    m0_sum += log_mel[t * 128 + 10];
  }
  EXPECT_NEAR(m0_sum / 10.0, 0.0, 1e-4);
  for (int m = 0; m < 128; ++m) {
    EXPECT_FLOAT_EQ(log_mel[10 * 128 + m], 0.0f);
  }

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));

  int32_t observed_mel_len = -1;
  auto mock_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto mel_buf, support::CreateTensorBuffer<float>({1, 128, 16}));
            LITERT_ASSIGN_OR_RETURN(auto len_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(mel_buf));
            bufs.push_back(std::move(len_buf));
            return bufs;
          });
  ON_CALL(*mock_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 2, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  ON_CALL(*mock_runner, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&observed_mel_len, 1)));
            std::vector<float> fake_out = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
            auto& out_buf = const_cast<TensorBuffer&>(outputs[0]);
            LITERT_RETURN_IF_ERROR(
                out_buf.Write<float>(absl::MakeConstSpan(fake_out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto audio_stage,
      Lfm2AudioStage::Create(fan_out->audio_queue(),
                             fan_out->audio_input_metadata_queue(),
                             std::move(mock_runner), config));

  // Act
  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = std::move(pcm)}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  EXPECT_TRUE(audio_stage->NeedSchedule());
  ASSERT_OK(audio_stage->Schedule());

  // Assert: `mel2emb_len(11) = ceil(11 / 8) = 2` frames with placeholder token
  // ID `-100` (no `<|audio_start|>`/`<|audio_end|>` wrapping around input
  // audio).
  EXPECT_EQ(observed_mel_len, 11);
  ASSERT_TRUE(audio_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Sequenced<Lfm2AudioEmbeddingSegment> out,
                       audio_stage->GetOutput());
  EXPECT_EQ(out.seqnum, 0u);
  EXPECT_EQ(out.item.num_tokens, 2);
  EXPECT_THAT(out.item.token_ids,
              ElementsAre(lfm2_audio::kAudioEmbedPlaceholderTokenId,
                          lfm2_audio::kAudioEmbedPlaceholderTokenId));
  EXPECT_EQ(out.item.embeddings.size(), 8u);
}

std::shared_ptr<NiceMock<MockLiteRtRunner>> CreateMockTextEmbedderRunner(
    int hidden_dim) {
  auto mock_emb = std::make_shared<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_emb, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*mock_emb, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([hidden_dim](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, 1, hidden_dim}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  ON_CALL(*mock_emb, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [hidden_dim](absl::string_view, absl::Span<const TensorBuffer> inputs,
                       absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t tok = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                    absl::MakeSpan(&tok, 1)));
            std::vector<float> emb(hidden_dim, static_cast<float>(tok) * 0.01f);
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(emb)));
            return absl::OkStatus();
          });
  return mock_emb;
}

TEST(Lfm2AudioTextAndFrontendTest,
     BuildsChatStatePromptAndRejectsOutOfRangeTokensAndImageInputs) {
  // Arrange
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out_no_audio,
                       Lfm2AudioInputFanOutStage::Create(
                           &input_source, /*has_audio_encoder=*/false));
  // Rejects AudioInput when has_audio_encoder is false.
  ASSERT_OK(input_source.PushInput(AudioInput{.pcm_samples = {0.1f, 0.2f}}));
  ASSERT_OK(input_source.Schedule());
  EXPECT_THAT(fan_out_no_audio->Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  fan_out_no_audio->Reset();

  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));

  // Rejects ImageInput with kInvalidArgument.
  ASSERT_OK(input_source.PushInput(ImageInput{}));
  ASSERT_OK(input_source.Schedule());
  EXPECT_THAT(fan_out->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  fan_out->Reset();

  auto tokenizer =
      std::make_shared<FakeTokenizer>(/*out_of_range_token_id=*/300);
  auto embedder_runner = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(
                           fan_out->text_queue(), fan_out->end_of_turn_queue(),
                           tokenizer, embedder_runner, config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), /*audio_stage=*/nullptr,
                                     tokenizer, embedder_runner, config));

  // Act 1: Out-of-range token ID (`300 >= 256`) must fail with
  // kInvalidArgument (never clamped).
  ASSERT_OK(
      input_source.PushInput(OmniSession::TextInput{.text = "OUT_OF_RANGE"}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  EXPECT_THAT(text_stage->Schedule(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  fan_out->Reset();
  text_stage->Reset();
  frontend->Reset();

  // Act 2: Valid user turn builds ChatState prompt starting with
  // `<|startoftext|>` (`1`) and `<|im_start|>` (`6`).
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(frontend->Schedule());

  // Assert
  ASSERT_TRUE(frontend->HasOutput());
  ASSERT_OK_AND_ASSIGN(Lfm2AudioFrontendOutput prompt, frontend->GetOutput());
  ASSERT_GT(prompt.prompt_len, 4);
  EXPECT_EQ(prompt.prompt_token_ids.front(), lfm2_audio::kBosTokenId);
  EXPECT_EQ(prompt.prompt_token_ids[1], lfm2_audio::kImStartTokenId);
  EXPECT_EQ(prompt.prompt_embeddings.size(),
            static_cast<size_t>(prompt.prompt_len) * 4u);
}

TEST(Lfm2AudioGenerationStageTest,
     InterleavedTextAndAudioOutWithDepthformerAndDetokenizer) {
  // Arrange
  Lfm2AudioModelConfig config;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = 256;
  config.tts_hidden_size = 4;
  config.tts_vocab_size = 32;  // EOAudio = 31, valid detok codes in [0, 30]
  config.tts_num_layers = 1;
  config.tts_num_codebooks = 8;
  config.tts_max_cache_len = 8;
  config.vocoder_chunk_frames = 2;
  config.detokenizer_upsample_samples = 16;
  config.generate_audio = true;
  config.tts_temperature = 0.0f;
  config.max_new_tokens = 8;

  PushInputSource input_source;
  ASSERT_OK_AND_ASSIGN(auto fan_out,
                       Lfm2AudioInputFanOutStage::Create(&input_source));
  auto tokenizer = std::make_shared<FakeTokenizer>();
  auto embedder_runner = CreateMockTextEmbedderRunner(config.llm_hidden_size);
  ASSERT_OK_AND_ASSIGN(auto text_stage,
                       Lfm2AudioTextStage::Create(
                           fan_out->text_queue(), fan_out->end_of_turn_queue(),
                           tokenizer, embedder_runner, config));
  ASSERT_OK_AND_ASSIGN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), /*audio_stage=*/nullptr,
                                     tokenizer, embedder_runner, config));

  // Scripted LFM2 backbone steps:
  // Step 0 (TEXT): emits token 3 ("Hello").
  // Step 1 (TEXT): emits token 128 (`<|audio_start|>`), switching to AUDIO_OUT.
  // Step 2 (AUDIO_OUT, decoding token 128): hidden_state = {1, 0, 0, 0} ->
  //   Depthformer generates 1 valid 8-codebook frame `[10..17]` and feeds its
  //   `audio_embedding` back into `llm_runner->Decode`.
  // Step 3 (AUDIO_OUT, decoding audio_embedding): hidden_state = {0, 1, 0, 0}
  //   -> Depthformer emits `EOAudio` (`31`) at codebook 0, switching back to
  //   TEXT and feeding `audio_embedding([31]*8)` into `llm_runner->Decode`.
  // Step 4 (TEXT, decoding EOAudio embedding): emits token 4 (" world").
  // Step 5 (TEXT): emits token 7 (`<|im_end|>`), stopping generation.
  auto scripted_llm = std::make_shared<ScriptedLiteRtLmRunner>(
      config.llm_vocab_size, config.llm_hidden_size,
      std::vector<ScriptedLiteRtLmRunner::StepSpec>{
          {.top_token_id = 3, .hidden_state = {0.1f, 0.2f, 0.3f, 0.4f}},
          {.top_token_id = lfm2_audio::kAudioStartTokenId,
           .hidden_state = {0.5f, 0.6f, 0.7f, 0.8f}},
          {.top_token_id = 0, .hidden_state = {1.0f, 0.0f, 0.0f, 0.0f}},
          {.top_token_id = 0, .hidden_state = {0.0f, 1.0f, 0.0f, 0.0f}},
          {.top_token_id = 4, .hidden_state = {0.2f, 0.2f, 0.2f, 0.2f}},
          {.top_token_id = lfm2_audio::kImEndTokenId,
           .hidden_state = {0.0f, 0.0f, 0.0f, 0.0f}},
      });

  // Mock TTS projector (`depth_linear` + `audio_embedding`).
  auto mock_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_proj, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto h_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto c_buf, support::CreateTensorBuffer<int32_t>({1, 8}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(h_buf));
            bufs.push_back(std::move(c_buf));
            return bufs;
          });
  ON_CALL(*mock_proj, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto df_in, support::CreateTensorBuffer<float>({1, 8, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto aud_emb, support::CreateTensorBuffer<float>({1, 1, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(df_in));
            bufs.push_back(std::move(aud_emb));
            return bufs;
          });
  ON_CALL(*mock_proj, Run(absl::string_view(""), _, _))
      .WillByDefault([](absl::string_view,
                        absl::Span<const TensorBuffer> inputs,
                        absl::Span<const TensorBuffer> outputs)
                         -> absl::Status {
        std::vector<float> h(4, 0.0f);
        std::vector<int32_t> codes(8, 0);
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(inputs[0]).Read<float>(
            absl::MakeSpan(h)));
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                absl::MakeSpan(codes)));
        std::vector<float> df_in(32, h[0]);
        std::vector<float> aud_emb = {
            static_cast<float>(codes[0]), static_cast<float>(codes[1]),
            static_cast<float>(codes[2]), static_cast<float>(codes[7])};
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                   .Write<float>(absl::MakeConstSpan(df_in)));
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                   .Write<float>(absl::MakeConstSpan(aud_emb)));
        return absl::OkStatus();
      });

  // Mock 8-codebook Depthformer (`tts_decoder`).
  auto mock_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_dec, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
            LITERT_ASSIGN_OR_RETURN(
                auto prev_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(auto step_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(prev_buf));
            bufs.push_back(std::move(step_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*mock_dec, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf,
                support::CreateTensorBuffer<float>({1, 1, 32}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*mock_dec, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [](absl::string_view, absl::Span<const TensorBuffer> inputs,
             absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> cond(4, 0.0f);
            int32_t step_idx = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                    absl::MakeSpan(cond)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                    absl::MakeSpan(&step_idx, 1)));
            std::vector<float> logits(32, -5.0f);
            if (cond[0] > 0.5f) {
              // Frame 0: emit code `10 + step_idx`, and also set `logits[31]`
              // high on `step_idx > 0` to verify `Lfm2AudioGenerationStage`
              // masks out `EOAudio` (`31`) on codebooks 1..7!
              logits[10 + step_idx] = 10.0f;
              if (step_idx > 0) {
                logits[31] = 20.0f;
              }
            } else {
              // Frame 1: emit `EOAudio` (`31`) at codebook 0.
              logits[31] = 10.0f;
            }
            std::vector<float> kv(64, 0.0f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(logits)));
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                       .Write<float>(absl::MakeConstSpan(kv)));
            return absl::OkStatus();
          });

  // Mock LFM2AudioDetokenizer (`[1, 8, 2] -> [1, 32]`).
  std::vector<int32_t> observed_vocoder_codes;
  auto mock_voc = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*mock_voc, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto in_buf, support::CreateTensorBuffer<int32_t>({1, 8, 2}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(in_buf));
            return bufs;
          });
  ON_CALL(*mock_voc, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto out_buf, support::CreateTensorBuffer<float>({1, 32}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(out_buf));
            return bufs;
          });
  ON_CALL(*mock_voc, Run(absl::string_view(""), _, _))
      .WillByDefault(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            observed_vocoder_codes.resize(16);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                    absl::MakeSpan(observed_vocoder_codes)));
            std::vector<float> pcm_out(32, 0.25f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(pcm_out)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto gen_stage,
      Lfm2AudioGenerationStage::Create(
          frontend.get(), tokenizer, scripted_llm, std::move(mock_proj),
          std::move(mock_dec), std::move(mock_voc), config));

  // Act
  ASSERT_OK(input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(input_source.PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK(input_source.Schedule());
  ASSERT_OK(fan_out->Schedule());
  ASSERT_OK(text_stage->Schedule());
  ASSERT_OK(frontend->Schedule());
  ASSERT_OK(gen_stage->Schedule());

  // Assert:
  // 1. Both audio frames (valid frame `[10..17]` and EOAudio frame `[31]*8`)
  //    fed their `audio_embedding` back into `llm_runner->Decode`!
  ASSERT_EQ(scripted_llm->received_audio_embeddings().size(), 2u);
  EXPECT_THAT(scripted_llm->received_audio_embeddings()[0],
              ElementsAre(10.0f, 11.0f, 12.0f, 17.0f));
  EXPECT_THAT(scripted_llm->received_audio_embeddings()[1],
              ElementsAre(31.0f, 31.0f, 31.0f, 31.0f));

  // 2. Detokenizer received `[1, 8, 2]` with frame 0 = `[10..17]` and frame 1
  //    right-padded with `-1` (and the EOAudio frame excluded).
  ASSERT_EQ(observed_vocoder_codes.size(), 16u);
  for (int k = 0; k < 8; ++k) {
    EXPECT_EQ(observed_vocoder_codes[k * 2 + 0], 10 + k);
    EXPECT_EQ(observed_vocoder_codes[k * 2 + 1], -1);
  }

  // 3. Stage emitted TextOutput("Hello world"), AudioOutput (16 samples for 1
  //    valid frame), and EndOfOutputTurn.
  ASSERT_TRUE(gen_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out1, gen_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Hello world");

  ASSERT_TRUE(gen_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out2, gen_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_EQ(std::get<AudioOutput>(out2).pcm_samples.size(), 16u);
  EXPECT_EQ(std::get<AudioOutput>(out2).sample_rate_hz, 24000);

  ASSERT_TRUE(gen_stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out3, gen_stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(out3));
}

TEST(Lfm2AudioGenerationStageTest,
     OverlapDiscardChunkedVocoderMatchesOneShotAndEnforcesMaxKvLen) {
  // Arrange: simulate a causal detokenizer whose output at frame `t` depends on
  // `[max(0, t - 17) .. min(T - 1, t + 1)]` (17 left causal LFM2 frames + 1
  // left/right ISTFT overlap frame), and verify that chunked detokenization
  // with `vocoder_chunk_frames = 24` produces bit-for-bit identical output to
  // one-shot detokenization (`vocoder_chunk_frames = 40`) on 32 audio frames.
  constexpr int kNumAudioFrames = 32;
  constexpr int kSamplesPerFrame = 4;

  auto run_with_chunk_frames =
      [&](int chunk_frames,
          int max_kv_len) -> absl::StatusOr<std::vector<float>> {
    Lfm2AudioModelConfig config;
    config.llm_hidden_size = 4;
    config.llm_vocab_size = 256;
    config.max_kv_len = max_kv_len;
    config.tts_hidden_size = 4;
    config.tts_vocab_size = 64;
    config.tts_num_layers = 1;
    config.tts_num_codebooks = 8;
    config.tts_max_cache_len = 8;
    config.vocoder_chunk_frames = chunk_frames;
    config.detokenizer_upsample_samples = kSamplesPerFrame;
    config.generate_audio = true;
    config.generation_mode = Lfm2AudioGenerationMode::kSequential;
    config.system_prompt = "Perform TTS.";
    config.tts_temperature = 0.0f;
    config.max_new_tokens = 64;
    config.max_tts_tokens = 64;

    PushInputSource input_source;
    ABSL_ASSIGN_OR_RETURN(auto fan_out,
                          Lfm2AudioInputFanOutStage::Create(&input_source));
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto embedder = CreateMockTextEmbedderRunner(config.llm_hidden_size);
    ABSL_ASSIGN_OR_RETURN(auto text_stage, Lfm2AudioTextStage::Create(
                                               fan_out->text_queue(),
                                               fan_out->end_of_turn_queue(),
                                               tokenizer, embedder, config));
    ABSL_ASSIGN_OR_RETURN(
        auto frontend, Lfm2AudioFrontendStage::Create(
                           text_stage.get(), /*audio_stage=*/nullptr, tokenizer,
                           embedder, config));

    // Step 0: emit `<|audio_start|>` (128).
    // Steps 1..32: emit 32 valid audio frames (`hidden[0] = f + 1`).
    // Step 33: emit EOAudio (`hidden[0] = -1.0f`).
    // Step 34: emit `<|im_end|>` (7).
    std::vector<ScriptedLiteRtLmRunner::StepSpec> steps;
    steps.push_back({.top_token_id = lfm2_audio::kAudioStartTokenId,
                     .hidden_state = {0.0f, 0.0f, 0.0f, 0.0f}});
    for (int f = 0; f < kNumAudioFrames; ++f) {
      steps.push_back(
          {.top_token_id = 0,
           .hidden_state = {static_cast<float>(f + 1), 0.0f, 0.0f, 0.0f}});
    }
    steps.push_back(
        {.top_token_id = 0, .hidden_state = {-1.0f, 0.0f, 0.0f, 0.0f}});
    steps.push_back({.top_token_id = lfm2_audio::kImEndTokenId,
                     .hidden_state = {0.0f, 0.0f, 0.0f, 0.0f}});
    auto scripted_llm = std::make_shared<ScriptedLiteRtLmRunner>(
        config.llm_vocab_size, config.llm_hidden_size, std::move(steps));

    auto mock_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_proj, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto h_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto c_buf, support::CreateTensorBuffer<int32_t>({1, 8}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(h_buf));
              bufs.push_back(std::move(c_buf));
              return bufs;
            });
    ON_CALL(*mock_proj, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto df_in, support::CreateTensorBuffer<float>({1, 8, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto aud_emb, support::CreateTensorBuffer<float>({1, 1, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(df_in));
              bufs.push_back(std::move(aud_emb));
              return bufs;
            });
    ON_CALL(*mock_proj, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [](absl::string_view, absl::Span<const TensorBuffer> inputs,
               absl::Span<const TensorBuffer> outputs) -> absl::Status {
              std::vector<float> h(4, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                      absl::MakeSpan(h)));
              std::vector<float> df_in(32, h[0]);
              std::vector<float> aud_emb(4, 0.1f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(df_in)));
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[1])
                      .Write<float>(absl::MakeConstSpan(aud_emb)));
              return absl::OkStatus();
            });

    auto mock_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_dec, CreateInputBuffers(absl::string_view("")))
        .WillByDefault(
            [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
              LITERT_ASSIGN_OR_RETURN(
                  auto cond_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
              LITERT_ASSIGN_OR_RETURN(
                  auto prev_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
              LITERT_ASSIGN_OR_RETURN(
                  auto step_buf, support::CreateTensorBuffer<int32_t>({1}));
              LITERT_ASSIGN_OR_RETURN(
                  auto kv_buf,
                  support::CreateTensorBuffer<float>({1, 2, 8, 4}));
              std::vector<TensorBuffer> bufs;
              bufs.push_back(std::move(cond_buf));
              bufs.push_back(std::move(prev_buf));
              bufs.push_back(std::move(step_buf));
              bufs.push_back(std::move(kv_buf));
              return bufs;
            });
    ON_CALL(*mock_dec, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto logits_buf, support::CreateTensorBuffer<float>({1, 1, 64}));
          LITERT_ASSIGN_OR_RETURN(
              auto kv_buf, support::CreateTensorBuffer<float>({1, 2, 8, 4}));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(logits_buf));
          bufs.push_back(std::move(kv_buf));
          return bufs;
        });
    ON_CALL(*mock_dec, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [](absl::string_view, absl::Span<const TensorBuffer> inputs,
               absl::Span<const TensorBuffer> outputs) -> absl::Status {
              std::vector<float> cond(4, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                      absl::MakeSpan(cond)));
              std::vector<float> logits(64, -5.0f);
              if (cond[0] > 0.0f) {
                const int code = static_cast<int>(cond[0]) % 50;
                logits[code] = 10.0f;
              } else {
                logits[63] = 10.0f;  // EOAudio = 63
              }
              std::vector<float> kv(64, 0.0f);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(logits)));
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[1])
                      .Write<float>(absl::MakeConstSpan(kv)));
              return absl::OkStatus();
            });

    auto mock_voc = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*mock_voc, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([chunk_frames](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto in_buf,
              support::CreateTensorBuffer<int32_t>({1, 8, chunk_frames}));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(in_buf));
          return bufs;
        });
    ON_CALL(*mock_voc, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([chunk_frames](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(auto out_buf,
                                  support::CreateTensorBuffer<float>(
                                      {1, chunk_frames * kSamplesPerFrame}));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(out_buf));
          return bufs;
        });
    ON_CALL(*mock_voc, Run(absl::string_view(""), _, _))
        .WillByDefault(
            [chunk_frames](
                absl::string_view, absl::Span<const TensorBuffer> inputs,
                absl::Span<const TensorBuffer> outputs) -> absl::Status {
              std::vector<int32_t> codes(8 * chunk_frames, -1);
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                      absl::MakeSpan(codes)));
              // Causal receptive field of 18 left frames + 1 right ISTFT frame
              // (ignoring -1 padded slots).
              std::vector<float> pcm(chunk_frames * kSamplesPerFrame, 0.0f);
              for (int t = 0; t < chunk_frames; ++t) {
                if (codes[t] < 0) continue;
                float acc = 0.0f;
                for (int d = -18; d <= 1; ++d) {
                  const int idx = t + d;
                  if (idx >= 0 && idx < chunk_frames && codes[idx] >= 0) {
                    acc += static_cast<float>(codes[idx] + 1) *
                           static_cast<float>(20 - std::abs(d)) * 0.01f;
                  }
                }
                for (int s = 0; s < kSamplesPerFrame; ++s) {
                  pcm[t * kSamplesPerFrame + s] =
                      acc + static_cast<float>(s) * 0.001f;
                }
              }
              LITERT_RETURN_IF_ERROR(
                  const_cast<TensorBuffer&>(outputs[0])
                      .Write<float>(absl::MakeConstSpan(pcm)));
              return absl::OkStatus();
            });

    ABSL_ASSIGN_OR_RETURN(
        auto gen_stage,
        Lfm2AudioGenerationStage::Create(
            frontend.get(), tokenizer, scripted_llm, std::move(mock_proj),
            std::move(mock_dec), std::move(mock_voc), config));

    ABSL_RETURN_IF_ERROR(
        input_source.PushInput(OmniSession::TextInput{.text = "Hello"}));
    ABSL_RETURN_IF_ERROR(input_source.PushInput(OmniSession::EndOfInputTurn{}));
    ABSL_RETURN_IF_ERROR(input_source.Schedule());
    ABSL_RETURN_IF_ERROR(fan_out->Schedule());
    ABSL_RETURN_IF_ERROR(text_stage->Schedule());
    ABSL_RETURN_IF_ERROR(frontend->Schedule());
    ABSL_RETURN_IF_ERROR(gen_stage->Schedule());

    while (gen_stage->HasOutput()) {
      ABSL_ASSIGN_OR_RETURN(Output out, gen_stage->GetOutput());
      if (const auto* aud = std::get_if<AudioOutput>(&out)) {
        return aud->pcm_samples;
      }
    }
    return absl::InternalError("Missing AudioOutput.");
  };

  ASSERT_OK_AND_ASSIGN(std::vector<float> wav_oneshot,
                       run_with_chunk_frames(/*chunk_frames=*/40,
                                             /*max_kv_len=*/512));
  ASSERT_OK_AND_ASSIGN(std::vector<float> wav_chunked,
                       run_with_chunk_frames(/*chunk_frames=*/24,
                                             /*max_kv_len=*/512));
  ASSERT_EQ(wav_chunked.size(),
            static_cast<size_t>(kNumAudioFrames * kSamplesPerFrame));
  ASSERT_EQ(wav_oneshot.size(), wav_chunked.size());
  for (size_t i = 0; i < wav_oneshot.size(); ++i) {
    EXPECT_NEAR(wav_chunked[i], wav_oneshot[i], 1e-5f) << "at sample " << i;
  }

  // Verify max_kv_len enforcement returns kResourceExhausted when prompt
  // exceeds max_kv_len.
  EXPECT_THAT(run_with_chunk_frames(/*chunk_frames=*/24, /*max_kv_len=*/4),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(Lfm2AudioSessionTest, RegisteredInOmniRegistryAndRunsTextAndAudioTurn) {
  // Arrange
  EXPECT_THAT(
      OmniRegistry::Get().GetRegisteredModels(),
      testing::IsSupersetOf({"lfm2.5-audio-1.5b", "lfm2_audio", "lfm2-audio",
                             "LiquidAI/LFM2.5-Audio-1.5B"}));

  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteLfm2AudioBundle("lfm2_audio_test.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);
  init_cfg.set_max_new_tokens(1);

  // Act
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                       OmniRegistry::Get().CreateSessionFactory(
                           "LiquidAI/LFM2.5-Audio-1.5B", settings, &init_cfg));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US"));
  EXPECT_EQ(factory->GetInputSampleRateHz(), 16000);
  EXPECT_EQ(factory->GetDefaultVoice("en-US"), "lfm2_voice");

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       factory->Create(std::move(input_source)));

  // Second concurrent session on same factory must fail with
  // kFailedPrecondition.
  EXPECT_THAT(factory->Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  // Assert
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
}

}  // namespace
}  // namespace litert::omni::any2any
