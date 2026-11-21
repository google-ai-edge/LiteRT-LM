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
//
// The `SaveContext` / `RestoreContext` tests also check the executor's rollback
// logic with a synthetic model whose second layer has a ring buffer KV cache.

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
#include "runtime/executor/litert/legacy_map_state.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#include "runtime/executor/llm_processed_context.h"
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

// Ring buffer model: layer 0 has a global KV cache of `kRingModelContextLen`
// entries, layer 1 a ring buffer of `kRingLen` entries, which the NPU executor
// detects from the shorter KV cache. `kRingLen` differs from `kDim`, so that
// the sequence axis of the KV cache buffers is unambiguous.
constexpr int kRingModelContextLen = 16;
constexpr int kRingLen = 8;

SyntheticSigSpec RingCacheUpdateSpec(absl::string_view key, int seq_len) {
  return SyntheticSigSpec{key,
                          {Int32Spec("input_pos", {seq_len}),
                           {"kv_cache_k_0", {1, 1, kRingModelContextLen, kDim}},
                           {"kv_cache_v_0", {1, 1, kDim, kRingModelContextLen}},
                           {"kv_cache_k_1", {1, 1, kRingLen, kDim}},
                           {"kv_cache_v_1", {1, 1, kDim, kRingLen}},
                           {"kv_slice_k_0", {1, 1, seq_len, kDim}},
                           {"kv_slice_v_0", {1, 1, kDim, seq_len}},
                           {"kv_slice_k_1", {1, 1, seq_len, kDim}},
                           {"kv_slice_v_1", {1, 1, kDim, seq_len}}},
                          {{"kv_cache_k_0", {1, 1, kRingModelContextLen, kDim}},
                           {"kv_cache_v_0", {1, 1, kDim, kRingModelContextLen}},
                           {"kv_cache_k_1", {1, 1, kRingLen, kDim}},
                           {"kv_cache_v_1", {1, 1, kDim, kRingLen}}}};
}

std::unique_ptr<SyntheticNpuModelResources> MakeSyntheticRingBufferNpuModel(
    ::litert::Environment& env) {
  auto decoder_inputs = [&](int seq_len) {
    return std::vector<TensorSpec>{
        {"mask_global", {1, 1, seq_len, kRingModelContextLen}},
        {"pos_emb_cos", {1, seq_len, 1, kDim}},
        {"pos_emb_sin", {1, seq_len, 1, kDim}},
        {"kv_cache_k_0", {1, 1, kRingModelContextLen, kDim}},
        {"kv_cache_v_0", {1, 1, kDim, kRingModelContextLen}},
        {"kv_cache_k_1", {1, 1, kRingLen, kDim}},
        {"kv_cache_v_1", {1, 1, kDim, kRingLen}},
        {"embeddings", {1, seq_len, kDim}}};
  };
  auto decoder_outputs = [&](int seq_len) {
    return std::vector<TensorSpec>{{"kv_slice_k_0", {1, 1, seq_len, kDim}},
                                   {"kv_slice_v_0", {1, 1, kDim, seq_len}},
                                   {"kv_slice_k_1", {1, 1, seq_len, kDim}},
                                   {"kv_slice_v_1", {1, 1, kDim, seq_len}}};
  };
  std::vector<TensorSpec> decode_outputs = decoder_outputs(1);
  decode_outputs.push_back({"logits", {1, 1, kVocabSize}});

  auto resources =
      std::make_unique<SyntheticNpuModelResources>(kRingModelContextLen);
  resources->AddModel(
      env, proto::LlmMetadata::TF_LITE_PREFILL_DECODE,
      {{"prefill_4", decoder_inputs(kPrefillLen), decoder_outputs(kPrefillLen)},
       {"decode", decoder_inputs(1), decode_outputs}});
  resources->AddModel(
      env, proto::LlmMetadata::TF_LITE_AUX,
      {{"prefill_mask_4",
        {Int32Spec("time_step", {1}),
         Int32Spec("input_tokens", {1, kPrefillLen})},
        {{"mask_global", {1, 1, kPrefillLen, kRingModelContextLen}}}},
       {"decode_mask",
        {Int32Spec("time_step", {1}), Int32Spec("input_tokens", {1, 1})},
        {{"mask_global", {1, 1, 1, kRingModelContextLen}}}},
       {"prefill_rope_4",
        {Int32Spec("input_pos", {kPrefillLen})},
        {{"pos_emb_cos", {1, kPrefillLen, 1, kDim}},
         {"pos_emb_sin", {1, kPrefillLen, 1, kDim}}}},
       {"decode_rope",
        {Int32Spec("input_pos", {1})},
        {{"pos_emb_cos", {1, 1, 1, kDim}}, {"pos_emb_sin", {1, 1, 1, kDim}}}},
       RingCacheUpdateSpec("prefill_cache_update_4", kPrefillLen),
       RingCacheUpdateSpec("decode_cache_update", 1)});
  resources->AddModel(env, proto::LlmMetadata::TF_LITE_EMBEDDER,
                      {{"prefill_embedder_4",
                        {Int32Spec("token_ids", {1, kPrefillLen})},
                        {{"embeddings", {1, kPrefillLen, kDim}}}},
                       {"decode_embedder",
                        {Int32Spec("token_ids", {1, 1})},
                        {{"embeddings", {1, 1, kDim}}}}});
  return resources;
}

// Prefills `ids` (as one prefill call).
absl::Status PrefillIds(LlmLiteRtNpuCompiledModelExecutor& executor,
                        ::litert::Environment& env,
                        const std::vector<int32_t>& ids) {
  auto tokens = TensorBuffer::CreateManaged(
      env, ::litert::TensorBufferType::kHostMemory,
      MakeType(ElementType::Int32, {1, static_cast<int32_t>(ids.size())}),
      ids.size() * sizeof(int32_t));
  if (!tokens.HasValue()) {
    return absl::InternalError(tokens.Error().Message());
  }
  if (auto written = tokens->Write<int32_t>(absl::MakeConstSpan(ids));
      !written.HasValue()) {
    return absl::InternalError(written.Error().Message());
  }
  return executor.Prefill(ExecutorInputs(ExecutorTextData(std::move(*tokens)),
                                         std::nullopt, std::nullopt));
}

class LlmLiteRtNpuSaveContextTest : public LlmLiteRtNpuDefaultEnvironmentTest {
 protected:
  // Creates an executor for the ring buffer model, which needs the HW KV cache
  // update.
  absl::StatusOr<std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor>>
  CreateRingBufferExecutor() {
    if (ring_resources_ == nullptr) {
      ring_resources_ = MakeSyntheticRingBufferNpuModel(*env_);
    }
    LlmExecutorSettings settings = *settings_;
    settings.SetMaxNumTokens(kRingModelContextLen);
    absl::StatusOr<NpuConfig> npu_config =
        settings.MutableBackendConfig<NpuConfig>();
    if (!npu_config.ok()) {
      return npu_config.status();
    }
    npu_config->use_hw_cache_update_for_npu = true;
    settings.SetBackendConfig(*npu_config);
    return LlmLiteRtNpuCompiledModelExecutor::Create(settings, *ring_resources_,
                                                     *env_);
  }

  std::unique_ptr<SyntheticNpuModelResources> ring_resources_;
};

// Rolling back more than one token after the ring buffer wrapped around
// requires restoring a saved context: until then, prefill, `CloneContext`,
// `SaveContext` and moving to another step fail, since the ring buffer holds
// the entries of the rolled back steps.
TEST_F(LlmLiteRtNpuSaveContextTest, RollbackAfterWrapAroundRequiresRestore) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateRingBufferExecutor());
  ASSERT_OK(PrefillIds(*executor, *env_, {1, 2, 3}));
  ASSERT_OK_AND_ASSIGN(const int saved_step, executor->GetCurrentStep());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcessedContext> saved,
                       executor->SaveContext());
  // The ring buffer (layer 1) is saved, the global KV cache (layer 0) is not.
  auto* saved_context = dynamic_cast<LlmProcessedContext*>(saved.get());
  ASSERT_NE(saved_context, nullptr);
  auto* saved_state =
      dynamic_cast<LegacyMapState*>(saved_context->state().get());
  ASSERT_NE(saved_state, nullptr);
  EXPECT_EQ(saved_state->buffers().size(), 2);
  EXPECT_TRUE(saved_state->Contains("kv_cache_k_1"));
  EXPECT_TRUE(saved_state->Contains("kv_cache_v_1"));
  EXPECT_EQ(saved_context->processed_tokens().TokenCount(), saved_step);

  // Wraps the ring buffer around.
  ASSERT_OK(PrefillIds(*executor, *env_, {4, 5, 6, 7, 8, 9, 10, 11}));
  ASSERT_OK_AND_ASSIGN(const int wrapped_step, executor->GetCurrentStep());
  ASSERT_GT(wrapped_step, kRingLen);

  ASSERT_OK(executor->SetCurrentStep(saved_step));
  EXPECT_THAT(PrefillIds(*executor, *env_, {4}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(executor->CloneContext(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(executor->SaveContext(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(executor->SetCurrentStep(saved_step + 1),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK(executor->RestoreContext(std::move(saved)));
  ASSERT_OK_AND_ASSIGN(const int restored_step, executor->GetCurrentStep());
  EXPECT_EQ(restored_step, saved_step);
  ASSERT_OK(PrefillIds(*executor, *env_, {4, 5}));
  ASSERT_OK(executor->CloneContext().status());
}

// A one-token rollback (e.g. dropping a stop token) and rolling back before the
// ring buffer wrapped around need no restore.
TEST_F(LlmLiteRtNpuSaveContextTest, RollbackWithoutWrapAroundNeedsNoRestore) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateRingBufferExecutor());
  ASSERT_OK(PrefillIds(*executor, *env_, {1, 2, 3}));
  ASSERT_OK(executor->SetCurrentStep(1));
  ASSERT_OK(PrefillIds(*executor, *env_, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  ASSERT_OK_AND_ASSIGN(const int step, executor->GetCurrentStep());
  ASSERT_OK(executor->SetCurrentStep(step - 1));
  ASSERT_OK(PrefillIds(*executor, *env_, {12}));
}

// A context saved before the executor switches to a context size with other
// ring buffers can't restore them, and restoring it changes no state.
TEST_F(LlmLiteRtNpuSaveContextTest, RestoreFailsForOtherRingBuffers) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateRingBufferExecutor());
  ASSERT_OK(PrefillIds(*executor, *env_, {1, 2, 3}));
  ASSERT_OK_AND_ASSIGN(const int step, executor->GetCurrentStep());
  auto other = std::make_unique<LlmProcessedContext>(
      /*lora_id=*/std::nullopt,
      std::make_unique<LegacyMapState>(
          absl::flat_hash_map<std::string, ::litert::TensorBuffer>()));
  EXPECT_THAT(executor->RestoreContext(std::move(other)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_OK_AND_ASSIGN(const int step_after, executor->GetCurrentStep());
  EXPECT_EQ(step_after, step);
  ASSERT_OK(PrefillIds(*executor, *env_, {4}));
}

// The resource manager clones a context handler (e.g. `Conversation::Clone()`)
// with the executor's runtime config and state, and switches between a clone
// and its source, which share the KV cache, by rolling back to the step of the
// shared processed context.
TEST_F(LlmLiteRtNpuSaveContextTest, RuntimeStateRollsBackToSharedStep) {
  std::unique_ptr<SyntheticNpuModelResources> resources =
      MakeSyntheticNpuModel(*env_);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor,
      LlmLiteRtNpuCompiledModelExecutor::Create(*settings_, *resources, *env_));
  ASSERT_OK(PrefillIds(*executor, *env_, {1, 2, 3}));
  ASSERT_OK_AND_ASSIGN(RuntimeConfig runtime_config,
                       executor->GetRuntimeConfig());
  EXPECT_TRUE(runtime_config.sampler_params.has_value());
  ASSERT_OK_AND_ASSIGN(RuntimeState shared_state, executor->GetRuntimeState());
  EXPECT_EQ(shared_state.current_step, 3);

  // The clone continues past the shared step.
  ASSERT_OK(PrefillIds(*executor, *env_, {4, 5}));
  ASSERT_OK_AND_ASSIGN(RuntimeState clone_state, executor->GetRuntimeState());
  // Switches back to the source.
  ASSERT_OK(executor->UpdateRuntimeState(shared_state));
  ASSERT_OK_AND_ASSIGN(const int step, executor->GetCurrentStep());
  EXPECT_EQ(step, shared_state.current_step);
  // The clone's later steps are not in the shared KV cache anymore.
  EXPECT_THAT(executor->UpdateRuntimeState(clone_state),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(executor->UpdateRuntimeState(shared_state));
  ASSERT_OK(PrefillIds(*executor, *env_, {6, 7}));
}

// After the ring buffer wrapped around, switching back to the shared step
// leaves the ring buffer with the clone's entries, so the next prefill fails
// instead of using them.
TEST_F(LlmLiteRtNpuSaveContextTest,
       RuntimeStateRollbackAfterWrapAroundRequiresRestore) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateRingBufferExecutor());
  ASSERT_OK(PrefillIds(*executor, *env_, {1, 2, 3}));
  ASSERT_OK_AND_ASSIGN(RuntimeState shared_state, executor->GetRuntimeState());
  ASSERT_OK(PrefillIds(*executor, *env_, {4, 5, 6, 7, 8, 9, 10, 11}));
  ASSERT_OK(executor->UpdateRuntimeState(shared_state));
  EXPECT_THAT(PrefillIds(*executor, *env_, {4}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// Without a ring buffer, `SaveContext` copies no buffers, and rolling back more
// than one token needs no restore. A context saved after a rollback holds the
// processed tokens up to the current step (they are rolled back lazily).
TEST_F(LlmLiteRtNpuSaveContextTest, WithoutRingBufferSavesNoBuffers) {
  std::unique_ptr<SyntheticNpuModelResources> resources =
      MakeSyntheticNpuModel(*env_);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LlmLiteRtNpuCompiledModelExecutor> executor,
      LlmLiteRtNpuCompiledModelExecutor::Create(*settings_, *resources, *env_));
  ASSERT_OK(PrefillIds(*executor, *env_, {1, 2, 3, 4, 5}));
  EXPECT_THAT(executor->SetCurrentStep(6),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(executor->SetCurrentStep(2));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcessedContext> saved,
                       executor->SaveContext());
  auto* saved_context = dynamic_cast<LlmProcessedContext*>(saved.get());
  ASSERT_NE(saved_context, nullptr);
  auto* saved_state =
      dynamic_cast<LegacyMapState*>(saved_context->state().get());
  ASSERT_NE(saved_state, nullptr);
  EXPECT_TRUE(saved_state->buffers().empty());
  EXPECT_EQ(saved_context->processed_tokens().TokenCount(), 2);
  ASSERT_OK(PrefillIds(*executor, *env_, {6, 7}));
  ASSERT_OK(executor->SetCurrentStep(2));
  ASSERT_OK(executor->RestoreContext(std::move(saved)));
  ASSERT_OK_AND_ASSIGN(const int step, executor->GetCurrentStep());
  EXPECT_EQ(step, 2);
  ASSERT_OK(PrefillIds(*executor, *env_, {6, 7}));
}

}  // namespace
}  // namespace litert::lm
