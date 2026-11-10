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

#include "omni/any2any/any2any_session.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {

class Any2AnySessionTest : public ::testing::Test {
 protected:
  static absl::StatusOr<std::unique_ptr<MultiStagedSession>>
  CallCreateWithInputSource(
      Any2AnySessionFactory& factory,
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> input_source) {
    return factory.CreateWithInputSource(session_config,
                                         std::move(input_source));
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

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

absl::StatusOr<std::string> WriteMiniCpmOBundleWithEmbeddedMetadata(
    absl::string_view filename) {
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
  any2any_meta.set_input_sample_rate_hz(24000);
  any2any_meta.set_output_sample_rate_hz(24000);
  any2any_meta.set_system_prompt("Embedded system prompt");
  auto* mcpm = any2any_meta.mutable_any2any_model_type()->mutable_minicpm_o();
  mcpm->set_llm_hidden_size(4);
  mcpm->set_llm_vocab_size(32);
  mcpm->set_tts_hidden_size(4);
  mcpm->set_tts_vocab_size(8);
  mcpm->set_tts_num_mel_bins(80);
  mcpm->set_vision_scale_resolution(448);
  mcpm->set_vision_patch_size(2);
  mcpm->set_vision_max_slice_nums(2);
  mcpm->set_num_image_queries(2);
  mcpm->set_audio_pool_step(5);
  mcpm->set_generate_audio(false);
  mcpm->set_max_new_tokens(1);
  mcpm->set_max_tts_tokens(1);
  mcpm->set_voice("embedded_voice");
  mcpm->set_language("en-US");
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
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_vision_encoder",
       BuildMinimalTfliteModelBuffer(
           {{"images", {1, 4, 12}, tflite::TensorType_FLOAT32},
            {"positions_xy", {1, 4, 2}, tflite::TensorType_INT32},
            {"vit_positions", {1, 4}, tflite::TensorType_INT32}},
           {{"features", {1, 2, 4}, tflite::TensorType_FLOAT32}})},
      {lm::schema::AnySectionDataType_TFLiteModel, "tf_lite_audio_encoder",
       BuildMinimalTfliteModelBuffer(
           {{"mel", {1, 80, 10}, tflite::TensorType_FLOAT32}},
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

// Fake Any2Any processing stage that pulls `Input` items from `InputSource`
// until `EndOfInputTurn`, and emits `TextOutput`, optionally `AudioOutput`, and
// `EndOfOutputTurn`.
class FakeAny2AnyStage : public SingleThreadedStageWithDeque<Output> {
 public:
  explicit FakeAny2AnyStage(Stage<Input>* input_source, bool emit_audio = false)
      : input_source_(*input_source), emit_audio_(emit_audio) {}

 protected:
  bool NeedScheduleInternal() const override {
    return !is_finished_ &&
           (input_source_.NeedSchedule() || input_source_.HasOutput());
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);

    while (!is_finished_) {
      if (!input_source_.HasOutput()) {
        if (!input_source_.NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_.Schedule();
        if (absl::IsOutOfRange(status)) {
          is_finished_ = true;
          break;
        }
        if (absl::IsNotFound(status)) {
          break;
        }
        ABSL_RETURN_IF_ERROR(status);
        if (!input_source_.HasOutput()) {
          break;
        }
      }

      ABSL_ASSIGN_OR_RETURN(Input input, input_source_.GetOutput());
      if (std::holds_alternative<EndOfInput>(input)) {
        is_finished_ = true;
        break;
      }
      if (std::holds_alternative<EndOfInputTurn>(input)) {
        EmitTurnOutput();
        SetState(State::kIdle);
        return absl::OkStatus();
      }
      if (const auto* aud_meta = std::get_if<AudioInputMetadata>(&input)) {
        current_sample_rate_ = aud_meta->sample_rate_hz;
      } else if (const auto* txt = std::get_if<TextInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "text:", txt->text);
      } else if (const auto* img = std::get_if<ImageInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "img:", img->image_bytes.size(), "x",
                        img->width, "x", img->height);
      } else if (const auto* aud = std::get_if<AudioInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "aud:", aud->pcm_samples.size(), "@",
                        current_sample_rate_);
      }
    }

    if (is_finished_) {
      if (!summary_.empty()) {
        EmitTurnOutput();
      }
      SetState(State::kIdle);
      if (HasOutput()) {
        return absl::OkStatus();
      }
      return absl::OutOfRangeError("End of input reached.");
    }

    SetState(State::kIdle);
    return absl::OkStatus();
  }

  void ResetInternal() override {
    summary_.clear();
    current_sample_rate_ = kDefaultInputSampleRateHz;
    is_finished_ = false;
  }

 private:
  void EmitTurnOutput() {
    PushOutput(TextOutput{.confirmed_text = std::move(summary_)});
    summary_.clear();
    if (emit_audio_) {
      PushOutput(AudioOutput{
          .pcm_samples = {0.25f, -0.25f},
          .sample_rate_hz = 24000,
      });
    }
    PushOutput(EndOfOutputTurn{});
  }

  Stage<Input>& input_source_;
  bool emit_audio_ = false;
  std::string summary_;
  int current_sample_rate_ = kDefaultInputSampleRateHz;
  bool is_finished_ = false;
};

TEST_F(Any2AnySessionTest,
       PushInputSourceEndOfTurnEnablesMultiTurnWithoutReset) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto any2any_stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/true);
  Stage<Output>* raw_out = any2any_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(any2any_stage));

  ASSERT_OK_AND_ASSIGN(auto session,
                       MultiStagedSession::Create(std::move(stages), raw_out));

  // Before EndOfInputTurn or Finish(), ProcessNext() waits for turn completion
  // and returns kNotFound.
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Describe "}));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "this:"}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  ASSERT_OK(raw_input->PushInput(OmniSession::ImageInput{
      .image_bytes = {255, 0, 0, 0, 255, 0},
      .width = 2,
      .height = 1,
      .channels = 3,
  }));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInputMetadata{.sample_rate_hz = 24000}));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {0.1f, 0.2f, 0.3f}}));

  // Signal end of turn with EndOfInputTurn control input.
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  // First ProcessNext() yields the TextOutput for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text,
            "text:Describe |text:this:|img:6x2x1|aud:3@24000");

  // Second ProcessNext() yields the AudioOutput for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_EQ(std::get<AudioOutput>(out2).sample_rate_hz, 24000);
  EXPECT_THAT(std::get<AudioOutput>(out2).pcm_samples,
              ElementsAre(0.25f, -0.25f));

  // Third ProcessNext() yields EndOfOutputTurn for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out_turn1, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(out_turn1));

  // Turn 2 on the SAME session (no Reset() called):
  ASSERT_OK(raw_input->PushInput(
      OmniSession::TextInput{.text = "Followup question"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out3, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out3));
  EXPECT_EQ(std::get<TextOutput>(out3).confirmed_text,
            "text:Followup question");
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out4, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out4));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out_turn2, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(out_turn2));

  // Calling Finish() ends the session.
  raw_input->Finish();
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST_F(Any2AnySessionTest, PushInputSourceProcessAsyncStreamsTextAndAudio) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto any2any_stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/true);
  Stage<Output>* raw_out = any2any_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(any2any_stage));

  ::litert::lm::ThreadPool pool("any2any_test_pool", 2);
  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), raw_out, &pool));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Async"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  raw_input->Finish();

  std::vector<std::string> texts;
  std::vector<float> audio_samples;
  int end_of_turn_count = 0;
  absl::Status final_status;
  absl::Notification done;

  ASSERT_OK(session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (!res.ok()) {
          final_status = res.status();
          done.Notify();
          return res.status();
        }
        if (const auto* txt = std::get_if<TextOutput>(&*res)) {
          texts.push_back(txt->confirmed_text);
        } else if (const auto* aud = std::get_if<AudioOutput>(&*res)) {
          audio_samples.insert(audio_samples.end(), aud->pcm_samples.begin(),
                               aud->pcm_samples.end());
        } else if (std::holds_alternative<OmniSession::EndOfOutputTurn>(*res)) {
          ++end_of_turn_count;
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(final_status, StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(texts, ElementsAre("text:Async"));
  EXPECT_THAT(audio_samples, ElementsAre(0.25f, -0.25f));
  EXPECT_EQ(end_of_turn_count, 1);
}

class FakeAny2AnySessionFactory : public Any2AnySessionFactory {
 public:
  FakeAny2AnySessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices = {},
      std::shared_ptr<ModelResources> absl_nullable model_resources = nullptr)
      : Any2AnySessionFactory(settings, session_config, metadata,
                              std::move(available_voices),
                              std::move(model_resources)) {}

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    auto any2any_stage = std::make_unique<FakeAny2AnyStage>(
        input_source.get(), session_config.generate_audio());
    Stage<Output>* raw_output_stage = any2any_stage.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(input_source));
    stages.push_back(std::move(any2any_stage));
    return MultiStagedSession::Create(std::move(stages), raw_output_stage,
                                      thread_pool());
  }
};

TEST_F(Any2AnySessionTest, FactoryMetadataAndConfigMerging) {
  lm::proto::Any2AnyMetadata metadata;
  metadata.add_supported_languages("en-US");
  metadata.add_supported_languages("zh-CN");
  metadata.set_input_sample_rate_hz(48000);

  lm::proto::Any2AnySessionConfig base_cfg;
  base_cfg.set_language("en-US");
  base_cfg.set_generate_audio(true);
  base_cfg.set_temperature(0.7f);

  OmniEngine::Settings settings;
  settings.num_threads = 2;

  FakeAny2AnySessionFactory factory(settings, &base_cfg, metadata,
                                    {"voice_a", "voice_b"},
                                    /*model_resources=*/nullptr);

  EXPECT_THAT(factory.GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_THAT(factory.available_voices(), ElementsAre("voice_a", "voice_b"));
  EXPECT_TRUE(factory.HasVoice("voice_a"));
  EXPECT_TRUE(factory.HasVoice("voice_b"));
  EXPECT_FALSE(factory.HasVoice("unknown_voice"));
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "voice_a");
  EXPECT_EQ(factory.GetInputSampleRateHz(), 48000);
  EXPECT_EQ(factory.engine_settings().num_threads, 2);
  EXPECT_EQ(factory.model_resources(), nullptr);

  lm::proto::Any2AnySessionConfig override_cfg;
  override_cfg.set_generate_audio(false);
  override_cfg.set_temperature(0.2f);
  lm::proto::Any2AnySessionConfig merged =
      factory.MergeSessionConfig(override_cfg);
  EXPECT_EQ(merged.language(), "en-US");
  EXPECT_FALSE(merged.generate_audio());
  EXPECT_FLOAT_EQ(merged.temperature(), 0.2f);
}

TEST_F(Any2AnySessionTest, FactoryDefaultsWhenEmpty) {
  lm::proto::Any2AnyMetadata metadata;
  FakeAny2AnySessionFactory factory(OmniEngine::Settings{},
                                    /*session_config=*/nullptr, metadata);

  EXPECT_TRUE(factory.GetAvailableLanguages().empty());
  EXPECT_TRUE(factory.available_voices().empty());
  EXPECT_FALSE(factory.HasVoice("voice_a"));
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "");
  EXPECT_EQ(factory.GetInputSampleRateHz(), kDefaultInputSampleRateHz);
  EXPECT_EQ(factory.session_config().language(), "");
}

TEST_F(Any2AnySessionTest, FactoryCreateAndExecuteSession) {
  lm::proto::Any2AnyMetadata metadata;
  lm::proto::Any2AnySessionConfig session_config;
  session_config.set_generate_audio(true);

  FakeAny2AnySessionFactory factory(OmniEngine::Settings{}, &session_config,
                                    metadata, {"voice_a"});

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();

  ASSERT_OK_AND_ASSIGN(auto session, factory.Create(std::move(input_source)));

  ASSERT_OK(
      raw_input->PushInput(OmniSession::TextInput{.text = "Test via Factory"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text,
            "text:Test via Factory");

  ASSERT_OK_AND_ASSIGN(OmniSession::Output audio_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
}

TEST_F(Any2AnySessionTest, MiniCpmORegisteredInOmniRegistry) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              testing::IsSupersetOf(
                  {"minicpm-o-4.5", "minicpm-o-4_5", "minicpm-o", "minicpm_o",
                   "openbmb/minicpm-o-4.5", "openbmb/MiniCPM-o-4_5",
                   "openbmb/MiniCPM-o-4.5"}));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("minicpm-o-4.5"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("unknown-any2any-model"));

  ASSERT_OK_AND_ASSIGN(const auto reg,
                       OmniRegistry::Get().GetRegisteredModel("minicpm-o-4.5"));
  EXPECT_EQ(reg.model_remote_url,
            "https://huggingface.co/litert-community/MiniCPM-o-4_5/resolve/"
            "main/minicpm_o_4_5_int8.litertlm");

  OmniEngine::Settings dir_settings;
  dir_settings.model_path = "/nonexistent/minicpm_o_dir";
  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("minicpm-o-4.5", dir_settings),
      StatusIs(absl::StatusCode::kInvalidArgument));

  OmniEngine::Settings file_settings;
  file_settings.model_path = "/nonexistent/minicpm_o.litertlm";
  EXPECT_FALSE(OmniRegistry::Get()
                   .CreateSessionFactory("minicpm-o-4.5", file_settings)
                   .ok());
}

TEST_F(Any2AnySessionTest,
       MiniCpmOSessionFactoryResolvesEmbeddedMetadataEnforcesLeaseAndRunsTurn) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteMiniCpmOBundleWithEmbeddedMetadata(
                           "minicpm_o_session_test.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;
  settings.force_fp32_on_gpu = true;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);
  init_cfg.set_max_new_tokens(1);
  init_cfg.set_max_tts_tokens(1);
  init_cfg.set_temperature(0.0f);
  init_cfg.set_top_p(0.9f);
  init_cfg.set_top_k(10);
  init_cfg.set_tts_temperature(0.0f);
  init_cfg.set_tts_top_p(0.85f);
  init_cfg.set_tts_top_k(15);
  init_cfg.set_system_prompt("Custom system prompt");
  init_cfg.set_voice("override_voice");
  init_cfg.set_language("zh-CN");

  lm::proto::Any2AnyMetadata caller_meta;
  caller_meta.set_input_sample_rate_hz(16000);

  // Act: create factory through OmniRegistry with both embedded and caller
  // metadata plus initial Any2AnySessionConfig.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> base_factory,
      OmniRegistry::Get().CreateSessionFactory(
          "openbmb/MiniCPM-o-4_5", settings, &init_cfg, &caller_meta));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);

  // Assert: embedded metadata fields were loaded and merged with caller_meta,
  // and default supported_languages were populated.
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_EQ(factory->GetInputSampleRateHz(), 16000);
  EXPECT_EQ(factory->GetDefaultVoice("zh-CN"), "override_voice");

  // Create session 1.
  auto input_source1 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input1 = input_source1.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session1,
                       factory->Create(std::move(input_source1)));

  // Attempting to create a second concurrent session while session1 is alive
  // must fail with kFailedPrecondition to protect the shared stateful
  // llm_backbone runner.
  EXPECT_THAT(factory->Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Execute a turn on session1.
  ASSERT_OK(raw_input1->PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(raw_input1->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session1->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session1->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));

  // Destroying session1 releases the active-session lease, allowing a new
  // session to be created and executed.
  session1.reset();

  lm::proto::Any2AnySessionConfig session2_cfg;
  session2_cfg.set_generate_audio(true);
  // Creating with generate_audio=true when TTS models are absent fails in
  // CreateMiniCpmOComponents AND releases the lease via absl::Cleanup!
  EXPECT_THAT(CallCreateWithInputSource(*factory, session2_cfg,
                                        std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kNotFound));

  // Verify the lease was released after the failed creation above.
  auto input_source2 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input2 = input_source2.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session2,
                       factory->Create(std::move(input_source2)));
  ASSERT_OK(raw_input2->PushInput(OmniSession::TextInput{.text = "world"}));
  ASSERT_OK(raw_input2->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out2, session2->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out2));
}

}  // namespace
}  // namespace litert::omni::any2any
