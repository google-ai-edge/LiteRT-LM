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

// Checks that the NPU executor never creates LiteRT's default environment.
// `TensorBuffer::GetDefaultEnvironment()` caches that environment in a
// function-local static and logs its creation only once per process, so these
// tests live in their own binary and each one checks the logs: no other test
// may create it first.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/internal/litert_logging.h"  // from @litert
#include "litert/cc/internal/litert_logging.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

using ::litert::ElementType;
using ::litert::Layout;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::status::StatusIs;

RankedTensorType MakeType(ElementType type, const std::vector<int32_t>& dims) {
  ::litert::Dimensions dimensions;
  for (auto d : dims) dimensions.push_back(d);
  return RankedTensorType(type, Layout(std::move(dimensions)));
}

std::vector<int32_t> DimensionsOf(const TensorBuffer& buffer) {
  auto type = buffer.TensorType();
  ABSL_CHECK(type.HasValue());
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

struct TensorSpec {
  absl::string_view name;
  std::vector<int32_t> dims;
  tflite::TensorType type = tflite::TensorType_FLOAT32;
};

struct SyntheticSigSpec {
  absl::string_view key;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
};

// Builds a minimal CPU-executable TFLite flatbuffer where each signature output
// is computed as ABS(zero_constant) of its declared shape.
std::vector<uint8_t> BuildSyntheticModelBuffer(
    absl::Span<const SyntheticSigSpec> specs) {
  flatbuffers::FlatBufferBuilder builder;
  auto opcodes = builder.CreateVector(
      {tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ABS)});
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder)};  // Buffer 0: dynamic tensors.
  auto make_tensor = [&](const TensorSpec& t, uint32_t buf_idx,
                         absl::string_view suffix = "") {
    std::string name = absl::StrCat(t.name, suffix);
    return tflite::CreateTensor(builder, builder.CreateVector(t.dims), t.type,
                                buf_idx, builder.CreateString(name));
  };
  auto make_map = [&](absl::string_view name, uint32_t idx) {
    return tflite::CreateTensorMap(
        builder, builder.CreateString(name.data(), name.size()), idx);
  };

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs;
  for (size_t s = 0; s < specs.size(); ++s) {
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    std::vector<int32_t> sg_in, sg_out;
    std::vector<flatbuffers::Offset<tflite::TensorMap>> in_maps, out_maps;
    std::vector<flatbuffers::Offset<tflite::Operator>> ops;
    for (const TensorSpec& in : specs[s].inputs) {
      sg_in.push_back(tensors.size());
      in_maps.push_back(make_map(in.name, tensors.size()));
      tensors.push_back(make_tensor(in, /*buf_idx=*/0));
    }
    for (const TensorSpec& out : specs[s].outputs) {
      size_t bytes = sizeof(float);
      for (int32_t d : out.dims) bytes *= d;
      buffers.push_back(tflite::CreateBuffer(
          builder, builder.CreateVector(std::vector<uint8_t>(bytes, 0))));
      const int32_t zero_idx = tensors.size();
      tensors.push_back(make_tensor(out, buffers.size() - 1, "_zero"));
      const int32_t out_idx = tensors.size();
      tensors.push_back(make_tensor(out, /*buf_idx=*/0));
      ops.push_back(tflite::CreateOperator(builder, /*opcode_index=*/0,
                                           builder.CreateVector({zero_idx}),
                                           builder.CreateVector({out_idx})));
      sg_out.push_back(out_idx);
      out_maps.push_back(make_map(out.name, out_idx));
    }
    auto key_str =
        builder.CreateString(specs[s].key.data(), specs[s].key.size());
    subgraphs.push_back(tflite::CreateSubGraph(
        builder, builder.CreateVector(tensors), builder.CreateVector(sg_in),
        builder.CreateVector(sg_out), builder.CreateVector(ops), key_str));
    sig_defs.push_back(tflite::CreateSignatureDef(
        builder, builder.CreateVector(in_maps), builder.CreateVector(out_maps),
        key_str, /*subgraph_index=*/s));
  }
  tflite::FinishModelBuffer(
      builder, tflite::CreateModel(builder, /*version=*/3, opcodes,
                                   builder.CreateVector(subgraphs),
                                   builder.CreateString("synthetic"),
                                   builder.CreateVector(buffers), 0, 0,
                                   builder.CreateVector(sig_defs)));
  return {builder.GetBufferPointer(),
          builder.GetBufferPointer() + builder.GetSize()};
}

class SyntheticNpuModelResources : public ModelResources {
 public:
  explicit SyntheticNpuModelResources(int max_num_tokens) {
    metadata_.set_max_num_tokens(max_num_tokens);
  }

  using ModelResources::GetTFLiteModel;

  void AddModel(::litert::Environment& env,
                proto::LlmMetadata::TfLiteModelType type,
                absl::Span<const SyntheticSigSpec> specs) {
    buffers_.push_back(BuildSyntheticModelBuffer(specs));
    const auto& bytes = buffers_.back();
    auto model = ::litert::Model::CreateFromBuffer(
        env, ::litert::BufferRef<uint8_t>(bytes.data(), bytes.size()));
    ABSL_CHECK(model.HasValue()) << model.Error().Message();
    models_.insert_or_assign(type, std::move(*model));
  }

  absl::StatusOr<const ::litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType type) override {
    auto it = models_.find(type);
    if (it == models_.end()) return absl::NotFoundError("Missing model.");
    return &it->second;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return &metadata_;
  }

 private:
  std::vector<std::vector<uint8_t>> buffers_;
  absl::flat_hash_map<proto::LlmMetadata::TfLiteModelType, ::litert::Model>
      models_;
  proto::LlmMetadata metadata_;
};

constexpr int kPrefillLen = 4;
constexpr int kContextLen = 8;
constexpr int kDim = 4;
constexpr int kVocabSize = 8;

TensorSpec Int32Spec(absl::string_view name, std::vector<int32_t> dims) {
  return TensorSpec{name, std::move(dims), tflite::TensorType_INT32};
}

SyntheticSigSpec CacheUpdateSpec(absl::string_view key, int seq_len) {
  return SyntheticSigSpec{key,
                          {Int32Spec("input_pos", {seq_len}),
                           {"kv_cache_k_0", {1, 1, kContextLen, kDim}},
                           {"kv_cache_v_0", {1, 1, kDim, kContextLen}},
                           {"kv_slice_k_0", {1, 1, seq_len, kDim}},
                           {"kv_slice_v_0", {1, 1, kDim, seq_len}}},
                          {{"kv_cache_k_0", {1, 1, kContextLen, kDim}},
                           {"kv_cache_v_0", {1, 1, kDim, kContextLen}}}};
}

// Builds a single-layer NPU model with a text decoder, an auxiliary model and
// an embedder. Without `decoder_takes_embeddings`, the text decoder has no
// "embeddings" input for the embedder to write to.
std::unique_ptr<SyntheticNpuModelResources> MakeSyntheticNpuModel(
    ::litert::Environment& env, bool decoder_takes_embeddings = true) {
  auto decoder_inputs = [&](int seq_len) {
    std::vector<TensorSpec> inputs = {
        {"mask_global", {1, 1, seq_len, kContextLen}},
        {"pos_emb_cos", {1, seq_len, 1, kDim}},
        {"pos_emb_sin", {1, seq_len, 1, kDim}},
        {"kv_cache_k_0", {1, 1, kContextLen, kDim}},
        {"kv_cache_v_0", {1, 1, kDim, kContextLen}}};
    if (decoder_takes_embeddings) {
      inputs.push_back({"embeddings", {1, seq_len, kDim}});
    }
    return inputs;
  };

  auto resources = std::make_unique<SyntheticNpuModelResources>(kContextLen);
  resources->AddModel(env, proto::LlmMetadata::TF_LITE_PREFILL_DECODE,
                      {{"prefill_4",
                        decoder_inputs(kPrefillLen),
                        {{"kv_slice_k_0", {1, 1, kPrefillLen, kDim}},
                         {"kv_slice_v_0", {1, 1, kDim, kPrefillLen}}}},
                       {"decode",
                        decoder_inputs(1),
                        {{"logits", {1, 1, kVocabSize}},
                         {"kv_slice_k_0", {1, 1, 1, kDim}},
                         {"kv_slice_v_0", {1, 1, kDim, 1}}}}});
  resources->AddModel(
      env, proto::LlmMetadata::TF_LITE_AUX,
      {{"prefill_mask_4",
        {Int32Spec("time_step", {1}),
         Int32Spec("input_tokens", {1, kPrefillLen})},
        {{"mask_global", {1, 1, kPrefillLen, kContextLen}}}},
       {"decode_mask",
        {Int32Spec("time_step", {1}), Int32Spec("input_tokens", {1, 1})},
        {{"mask_global", {1, 1, 1, kContextLen}}}},
       {"prefill_rope_4",
        {Int32Spec("input_pos", {kPrefillLen})},
        {{"pos_emb_cos", {1, kPrefillLen, 1, kDim}},
         {"pos_emb_sin", {1, kPrefillLen, 1, kDim}}}},
       {"decode_rope",
        {Int32Spec("input_pos", {1})},
        {{"pos_emb_cos", {1, 1, 1, kDim}}, {"pos_emb_sin", {1, 1, 1, kDim}}}},
       CacheUpdateSpec("prefill_cache_update_4", kPrefillLen),
       CacheUpdateSpec("decode_cache_update", 1)});
  resources->AddModel(env, proto::LlmMetadata::TF_LITE_EMBEDDER,
                      {{"prefill_embedder_4",
                        {Int32Spec("token_ids", {1, kPrefillLen})},
                        {{"embeddings", {1, kPrefillLen, kDim}}}},
                       {"decode_embedder",
                        {Int32Spec("token_ids", {1, 1})},
                        {{"embeddings", {1, 1, kDim}}}}});
  return resources;
}

class LlmLiteRtNpuDefaultEnvironmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<::litert::EnvironmentOptions::Option> env_options = {
        {::litert::EnvironmentOptions::Tag::kAutoRegisterAccelerators,
         static_cast<int64_t>(::litert::HwAccelerators::kCpu)},
    };
    auto env = ::litert::Environment::Create(
        ::litert::EnvironmentOptions(env_options));
    ASSERT_TRUE(env.HasValue()) << env.Error().Message();
    env_.emplace(std::move(*env));

    ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                         ModelAssets::Create("dummy_model"));
    ASSERT_OK_AND_ASSIGN(LlmExecutorSettings settings,
                         LlmExecutorSettings::CreateDefault(
                             std::move(model_assets), Backend::NPU));
    settings.SetMaxNumTokens(kContextLen);
    settings_.emplace(std::move(settings));

    // Intercept LiteRT logs from here on. Any default-constructed
    // TensorBuffer() or environment-less helper triggers
    // TensorBuffer::GetDefaultEnvironment() -> Environment::Create({}),
    // logging the messages checked in TearDown().
    intercepted_logs_.emplace();
    LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(),
                               kLiteRtLogSeverityInfo);
  }

  void TearDown() override {
    if (!intercepted_logs_.has_value()) return;
    std::ostringstream log_stream;
    log_stream << *intercepted_logs_;
    const std::string logs = log_stream.str();
    EXPECT_THAT(logs,
                Not(HasSubstr("Creating LiteRT environment with options")));
    EXPECT_THAT(
        logs,
        Not(HasSubstr("NPU accelerator could not be loaded and registered")));
    intercepted_logs_.reset();
  }

  std::optional<::litert::Environment> env_;
  std::optional<LlmExecutorSettings> settings_;
  std::optional<::litert::InterceptLogs> intercepted_logs_;
};

TEST_F(LlmLiteRtNpuDefaultEnvironmentTest,
       ExecutorCreateAndLifecycleDoNotCreateDefaultEnvironment) {
  ::litert::Environment& env = *env_;
  std::unique_ptr<SyntheticNpuModelResources> resources =
      MakeSyntheticNpuModel(env);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor,
      LlmLiteRtNpuCompiledModelExecutor::Create(*settings_, *resources, env));

  const std::vector<int32_t> prompt_ids = {1, 2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer prefill_tokens,
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  MakeType(ElementType::Int32, {1, 3}),
                                  prompt_ids.size() * sizeof(int32_t)));
  LITERT_ASSERT_OK(
      prefill_tokens.Write<int32_t>(absl::MakeConstSpan(prompt_ids)));
  ASSERT_OK(executor->Prefill(
      ExecutorInputs(ExecutorTextData(std::move(prefill_tokens)), std::nullopt,
                     std::nullopt)));

  ASSERT_OK_AND_ASSIGN(TensorBuffer logits,
                       executor->DecodeLogits(ExecutorInputs()));
  EXPECT_THAT(DimensionsOf(logits), ElementsAre(1, 1, kVocabSize));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmContext> cloned_ctx,
                       executor->CloneContext());
  ASSERT_OK(executor->RestoreContext(std::move(cloned_ctx)));
}

TEST_F(LlmLiteRtNpuDefaultEnvironmentTest,
       CreateFailsWhenTextDecoderHasNoEmbeddingsInput) {
  std::unique_ptr<SyntheticNpuModelResources> resources =
      MakeSyntheticNpuModel(*env_, /*decoder_takes_embeddings=*/false);
  EXPECT_THAT(
      LlmLiteRtNpuCompiledModelExecutor::Create(*settings_, *resources, *env_),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("has no input \"embeddings\"")));
}

}  // namespace
}  // namespace litert::lm
