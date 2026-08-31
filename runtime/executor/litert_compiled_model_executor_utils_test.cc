// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/litert_compiled_model_executor_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/core/model/model.h"  // from @litert
#include "litert/core/util/flatbuffer_tools.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/util/file_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"  // from @org_tensorflow
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::status::StatusIs;

constexpr uint64_t kLiteRtLmPrefixSize = 32;
constexpr uint64_t kLiteRtLmSectionAlignment = 16 * 1024;

uint64_t AlignLiteRtLmSection(uint64_t offset) {
  return ((offset + kLiteRtLmSectionAlignment - 1) /
          kLiteRtLmSectionAlignment) *
         kLiteRtLmSectionAlignment;
}

void BuildAotTestHeader(flatbuffers::FlatBufferBuilder& builder,
                        uint64_t prefill_begin, uint64_t prefill_end,
                        uint64_t aux_begin, uint64_t aux_end) {
  builder.Clear();
  auto prefill_model_type = schema::CreateKeyValuePair(
      builder, "model_type",
      ModelTypeToString(ModelType::kTfLitePrefillDecode));
  auto prefill_items = builder.CreateVector(
      std::vector<flatbuffers::Offset<schema::KeyValuePair>>{
          prefill_model_type});
  auto prefill_section = schema::CreateSectionObject(
      builder, prefill_items, prefill_begin, prefill_end,
      schema::AnySectionDataType_TFLiteModel);

  auto aux_model_type = schema::CreateKeyValuePair(
      builder, "model_type", ModelTypeToString(ModelType::kTfLiteAux));
  auto aux_items = builder.CreateVector(
      std::vector<flatbuffers::Offset<schema::KeyValuePair>>{aux_model_type});
  auto aux_section =
      schema::CreateSectionObject(builder, aux_items, aux_begin, aux_end,
                                  schema::AnySectionDataType_TFLiteModel);

  auto sections = builder.CreateVector(
      std::vector<flatbuffers::Offset<schema::SectionObject>>{prefill_section,
                                                              aux_section});
  auto section_metadata = schema::CreateSectionMetadata(builder, sections);
  auto metadata = schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);
}

void WriteAotTestModel(const std::filesystem::path& path,
                       litert::BufferRef<uint8_t> model_buffer) {
  flatbuffers::FlatBufferBuilder builder;
  BuildAotTestHeader(builder, 0, 0, 0, 0);
  const uint64_t prefill_begin =
      AlignLiteRtLmSection(kLiteRtLmPrefixSize + builder.GetSize());
  const uint64_t prefill_end = prefill_begin + model_buffer.Size();
  const uint64_t aux_begin = AlignLiteRtLmSection(prefill_end);
  const uint64_t aux_end = aux_begin + model_buffer.Size();
  BuildAotTestHeader(builder, prefill_begin, prefill_end, aux_begin, aux_end);

  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.good());
  file.write("LITERTLM", 8);
  const uint32_t major_version = 1;
  const uint32_t minor_version = 0;
  const uint32_t patch_version = 0;
  const uint32_t padding = 0;
  file.write(reinterpret_cast<const char*>(&major_version), 4);
  file.write(reinterpret_cast<const char*>(&minor_version), 4);
  file.write(reinterpret_cast<const char*>(&patch_version), 4);
  file.write(reinterpret_cast<const char*>(&padding), 4);
  const uint64_t header_end = kLiteRtLmPrefixSize + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end), 8);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  file.write(std::string(prefill_begin - header_end, '\0').data(),
             prefill_begin - header_end);
  file.write(reinterpret_cast<const char*>(model_buffer.Data()),
             model_buffer.Size());
  file.write(std::string(aux_begin - prefill_end, '\0').data(),
             aux_begin - prefill_end);
  file.write(reinterpret_cast<const char*>(model_buffer.Data()),
             model_buffer.Size());
  ASSERT_TRUE(file.good());
}

bool IsFileBacked(const ::litert::Model& model) {
  const auto& flatbuffer =
      ::litert::internal::GetTflFlatbuffer(*model.Get()).FlatbufferModel();
  const auto* allocation = flatbuffer.allocation();
  return allocation != nullptr &&
         allocation->type() == tflite::Allocation::Type::kMMap;
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_Gemma2JAX) {
  std::vector<absl::string_view> input_names = {"token_ids", "positions",
                                                "attn_mask"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "token_ids");
  EXPECT_EQ(signatures.input_positions, "positions");
  EXPECT_EQ(signatures.input_attn_mask, "attn_mask");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_PyTorch) {
  std::vector<absl::string_view> input_names = {"tokens", "input_pos", "mask"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "tokens");
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_EQ(signatures.input_attn_mask, "mask");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_PyTorchCpuOnly) {
  std::vector<absl::string_view> input_names = {"tokens", "input_pos"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "tokens");
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_FALSE(signatures.input_attn_mask.has_value());
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_Gemini) {
  std::vector<absl::string_view> input_names = {"token_ids", "positions",
                                                "attn_mask"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_EQ(signatures.input_tokens, "token_ids");
  EXPECT_EQ(signatures.input_positions, "positions");
  EXPECT_EQ(signatures.input_attn_mask, "attn_mask");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_ExternalEmbeddingModel) {
  std::vector<absl::string_view> input_names = {
      "input_pos", "mask", "embeddings", "per_layer_embeddings"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_TRUE(signatures.input_tokens.empty());
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_EQ(signatures.input_attn_mask, "mask");
  EXPECT_EQ(signatures.input_embeddings, "embeddings");
  EXPECT_EQ(signatures.input_per_layer_embeddings, "per_layer_embeddings");
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_ExternalEmbeddingModelWithoutPle) {
  std::vector<absl::string_view> input_names = {"input_pos", "mask",
                                                "embeddings"};
  std::vector<absl::string_view> output_names = {"logits"};
  ASSERT_OK_AND_ASSIGN(auto signatures, GetModelSignaturesFromInputOutputNames(
                                            input_names, output_names));
  EXPECT_TRUE(signatures.input_tokens.empty());
  EXPECT_EQ(signatures.input_positions, "input_pos");
  EXPECT_EQ(signatures.input_attn_mask, "mask");
  EXPECT_EQ(signatures.input_embeddings, "embeddings");
  EXPECT_FALSE(signatures.input_per_layer_embeddings.has_value());
  EXPECT_EQ(signatures.output_logits, "logits");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetModelSignaturesFromInputOutputNames_Unsupported) {
  std::vector<absl::string_view> input_names = {"unknown_input"};
  std::vector<absl::string_view> output_names = {"logits"};
  EXPECT_THAT(GetModelSignaturesFromInputOutputNames(input_names, output_names),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_KvCache) {
  std::vector<absl::string_view> input_names = {"kv_cache_k_0", "kv_cache_v_0"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "kv_cache_k_");
  EXPECT_EQ(v_root_name, "kv_cache_v_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_KCache) {
  std::vector<absl::string_view> input_names = {"k_cache_0", "v_cache_0"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "k_cache_");
  EXPECT_EQ(v_root_name, "v_cache_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetKVCacheRootNames_KCacheOutput) {
  std::vector<absl::string_view> input_names = {};
  std::vector<absl::string_view> output_names = {"k_cache_0", "v_cache_0"};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "k_cache_");
  EXPECT_EQ(v_root_name, "v_cache_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_NotFound) {
  std::vector<absl::string_view> input_names = {"other_input"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  EXPECT_THAT(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetKVCacheRootNames_KvCacheC) {
  std::vector<absl::string_view> input_names = {"kv_cache_c_0"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "kv_cache_c_");
  EXPECT_EQ(v_root_name, "kv_cache_c_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetKVCacheRootNames_HybridConvAndKVCache) {
  // Hybrid models like LFM start with conv cache (kv_cache_c_0) followed by
  // attention layers (kv_cache_k_2, kv_cache_v_2). Standard KV cache names
  // should take precedence.
  std::vector<absl::string_view> input_names = {"kv_cache_c_0", "kv_cache_c_1",
                                                "kv_cache_k_2", "kv_cache_v_2"};
  std::vector<absl::string_view> output_names = {};
  std::string k_root_name;
  std::string v_root_name;
  ASSERT_OK(
      GetKVCacheRootNames(input_names, output_names, k_root_name, v_root_name));
  EXPECT_EQ(k_root_name, "kv_cache_k_");
  EXPECT_EQ(v_root_name, "kv_cache_v_");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillSingleBufferCacheParamTensor) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({12}));
  RankedTensorType ranked_tensor_type(ElementType::Int32, std::move(layout));
  auto param_tensor =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(int) * 12);
  ASSERT_TRUE(param_tensor);

  ASSERT_OK(FillSingleBufferCacheParamTensor(*param_tensor, /*start_index=*/5,
                                             /*update_length=*/10));
  auto lock = litert::TensorBufferScopedLock::Create(
      *param_tensor, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  int* param_ptr = static_cast<int*>(lock->second);
  EXPECT_EQ(param_ptr[0], 5);
  EXPECT_EQ(param_ptr[1], 15);
  EXPECT_EQ(param_ptr[2], 15);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_ExactMultiples) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 2048));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_1024", 1024),
                                       Pair("prefill_1024", 1024)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_MixedRunners) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 1536));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_1024", 1024),
                                       Pair("prefill_512", 512)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_RemainderWithSmallerRunner) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  // 1100 = 1024 + 76. The smallest runner >= 76 is prefill_128.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 1100));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_1024", 1024), Pair("prefill_128", 76)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_SingleRunner) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 1024));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_1024", 1024)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_LargerThanLargest) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {128, "prefill_128"}, {32, "prefill_32"}};
  // 600 = 512 + 88. The smallest runner >= 88 is prefill_128.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 600));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_128", 88)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_SmallestRunnerOnly) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {1024, "prefill_1024"}, {512, "prefill_512"}, {128, "prefill_128"}};
  // 100 < 128. Uses the smallest runner.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 100));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_128", 100)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_CautiousUpgradeTopLevel) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {128, "prefill_128"}, {32, "prefill_32"}};
  // Remainder 256 >= 512/2 -> Upgrade to 512
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 768));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_512", 256)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_CpuGreedyChunking) {
  SortedPrefillSignatureMap prefill_runner_set = {{1024, "prefill_1024"},
                                                  {128, "prefill_128"}};
  // For 512 tokens with greedy chunking enabled (CPU), cascades to 4x 128
  // instead of 1x 1024.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(
                           prefill_runner_set, 512,
                           /*max_prefill_sequence_length=*/std::nullopt,
                           /*use_greedy_chunking=*/true));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_128", 128), Pair("prefill_128", 128),
                          Pair("prefill_128", 128), Pair("prefill_128", 128)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_CpuGreedyCascadesRemainder) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {128, "prefill_128"}, {32, "prefill_32"}};
  // For 768 tokens (512 + 256): 1 full chunk of 512, remainder 256 cascades to
  // 2x 128 instead of upgrading to a second 512 chunk.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(
                           prefill_runner_set, 768,
                           /*max_prefill_sequence_length=*/std::nullopt,
                           /*use_greedy_chunking=*/true));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_128", 128),
                          Pair("prefill_128", 128)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_DeepCascadeUpgrade) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {128, "prefill_128"}, {32, "prefill_32"}};
  // Remainder 80 < 512/2, but 80 >= 128/2 -> Cascade and upgrade to 128
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 592));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_128", 80)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_FallbackOverride) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {384, "prefill_384"}, {128, "prefill_128"}};
  // Remainder 256 >= 512/2, BUT next chunk 384 > 512/2. So fallback to 384!
  // At 384, Remainder 256 >= 384/2. Upgrades to 384.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 768));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_384", 256)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_NoUpgradeAfterFallback) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {384, "prefill_384"}, {128, "prefill_128"}};
  // Remainder 128. Fallback triggered at 512 because 384 > 256.
  // At 384, 128 < 384/2, so no upgrade.
  // Ends up using perfectly fitting 128.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 640));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_512", 512), Pair("prefill_128", 128)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_MultipleFullChunksThenUpgrade) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {100, "prefill_100"}, {30, "prefill_30"}, {10, "prefill_10"}};
  // 2 full 100 chunks. Remainder 80 >= 100/2. Upgrade to a third 100 chunk!
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 280));
  EXPECT_THAT(work_groups,
              ElementsAre(Pair("prefill_100", 100), Pair("prefill_100", 100),
                          Pair("prefill_100", 80)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_DenseTierCascadeTrap) {
  SortedPrefillSignatureMap prefill_runner_set = {{600, "prefill_600"},
                                                  {500, "prefill_500"}};
  // Input 599. Remainder 599. Next size 500.
  // 500 * 2 >= 600, but 599 > 500!
  // Fallback is skipped! Upgrades to 600 instead of outputting {{500, 500},
  // {500, 99}}.
  ASSERT_OK_AND_ASSIGN(auto work_groups,
                       GetOptimizedPrefillWorkGroups(prefill_runner_set, 599));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_600", 599)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_MaxPrefillSequenceLengthAdapts) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {256, "prefill_256"}, {128, "prefill_128"}};
  // input_length = 200, but max_prefill_sequence_length = 256.
  // 512 runner is filtered out, adapting to 256 runner.
  ASSERT_OK_AND_ASSIGN(
      auto work_groups,
      GetOptimizedPrefillWorkGroups(prefill_runner_set, 200,
                                    /*max_prefill_sequence_length=*/256));
  EXPECT_THAT(work_groups, ElementsAre(Pair("prefill_256", 200)));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetOptimizedPrefillWorkGroups_MaxPrefillSequenceLengthExceededError) {
  SortedPrefillSignatureMap prefill_runner_set = {
      {512, "prefill_512"}, {256, "prefill_256"}, {128, "prefill_128"}};
  // input_length = 300, max_prefill_sequence_length = 256.
  // 512 runner filtered out. 256 runner covers 256 tokens, remaining 44 tokens
  // cannot fit in 0 capacity.
  EXPECT_THAT(GetOptimizedPrefillWorkGroups(
                  prefill_runner_set, 300, /*max_prefill_sequence_length=*/256),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetPrefillRunnerSetFromModel) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK_AND_ASSIGN(auto litert_model, model_resources->GetTFLiteModel(
                                              ModelType::kTfLitePrefillDecode));
  ASSERT_NE(litert_model, nullptr);

  ASSERT_OK_AND_ASSIGN(
      auto prefill_runner_set,
      GetPrefillRunnerSetFromModel(*litert_model, "prefill",
                                   /*input_positions_name=*/"input_pos"));
  EXPECT_THAT(prefill_runner_set, ElementsAre(Pair(160, "prefill")));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, InitializeAttentionMask_Float32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 128}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 128);
  ASSERT_TRUE(mask_buffer);

  // Test with is_f16 = false
  {
    ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));
    auto lock = litert::TensorBufferScopedLock::Create(
        *mask_buffer, litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock);
    float* mask_ptr = static_cast<float*>(lock->second);
    float expected_val = -0.7f * std::numeric_limits<float>::max();
    int count_float = 128;
    for (int i = 0; i < count_float; ++i) {
      EXPECT_EQ(mask_ptr[i], expected_val);
    }
  }

  // Test with is_f16 = true
  {
    ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/true));
    auto lock = litert::TensorBufferScopedLock::Create(
        *mask_buffer, litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock);
    float* mask_ptr = static_cast<float*>(lock->second);
    float expected_val = -45824.0f;
    int count_float = 128;
    for (int i = 0; i < count_float; ++i) {
      EXPECT_EQ(mask_ptr[i], expected_val);
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, InitializeAttentionMask_Float16) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 128}));
  RankedTensorType ranked_tensor_type(ElementType::Float16, std::move(layout));
  auto mask_buffer = TensorBuffer::CreateManaged(
      env, ::litert::TensorBufferType::kHostMemory, ranked_tensor_type,
      sizeof(tflite::half) * 128);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/true));
  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  tflite::half* mask_ptr = static_cast<tflite::half*>(lock->second);
  float expected_val = -45824.0f;
  int count_float = 128;
  for (int i = 0; i < count_float; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(mask_ptr[i]), expected_val);
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, InitializeAttentionMask_Bool) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 128}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 128);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));
  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);
  int count_bool = 128;
  for (int i = 0; i < count_bool; ++i) {
    EXPECT_FALSE(mask_ptr[i]);
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, FillAttentionMask_Float32) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=4, 1, max_kv_len=10]
  auto layout = ::litert::Layout(::litert::Dimensions({1, 4, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 40);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default float value.
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 5 for 2 steps.
  // channel_size = 10
  // i = 0: current_step = 5. Fills indices [0, 5] with 0.0f.
  // i = 1: current_step = 6. Fills indices [10, 16] with 0.0f.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/5, /*steps=*/2,
                              proto::ATTENTION_MASK_TYPE_CAUSAL));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);
  float init_val = -0.7f * std::numeric_limits<float>::max();

  for (int i = 0; i < 40; ++i) {
    if ((i >= 0 && i <= 5) || (i >= 10 && i <= 16)) {
      EXPECT_EQ(mask_ptr[i], 0.0f) << " at index " << i;
    } else {
      EXPECT_EQ(mask_ptr[i], init_val) << " at index " << i;
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_Float32_MultipleBatches) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=4, 1, max_kv_len=10]
  auto layout = ::litert::Layout(::litert::Dimensions({3, 4, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 120);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default float value.
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 5 for 2 steps.
  // channel_size = 10
  // i = 0: current_step = 5. Fills indices [0, 5] with 0.0f.
  // i = 1: current_step = 6. Fills indices [10, 16] with 0.0f.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/5, /*steps=*/2,
                              proto::ATTENTION_MASK_TYPE_CAUSAL));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);
  float init_val = -0.7f * std::numeric_limits<float>::max();

  for (int b = 0; b < 3; ++b) {
    for (int i = 0; i < 40; ++i) {
      if ((i >= 0 && i <= 5) || (i >= 10 && i <= 16)) {
        EXPECT_EQ(mask_ptr[b * 40 + i], 0.0f) << " at index " << i;
      } else {
        EXPECT_EQ(mask_ptr[b * 40 + i], init_val) << " at index " << i;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, FillAttentionMask_Bool) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=3, 1, max_kv_len=8]
  auto layout = ::litert::Layout(::litert::Dimensions({1, 3, 1, 8}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 24);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default bool value (false).
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 2 for 3 steps.
  // channel_size = 8
  // i = 0: current_step = 2. Fills indices [0, 2] with true.
  // i = 1: current_step = 3. Fills indices [8, 11] with true.
  // i = 2: current_step = 4. Fills indices [16, 20] with true.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/2, /*steps=*/3,
                              proto::ATTENTION_MASK_TYPE_CAUSAL));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  for (int i = 0; i < 24; ++i) {
    if ((i >= 0 && i <= 2) || (i >= 8 && i <= 11) || (i >= 16 && i <= 20)) {
      EXPECT_TRUE(mask_ptr[i]) << " at index " << i;
    } else {
      EXPECT_FALSE(mask_ptr[i]) << " at index " << i;
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_Bool_MultipleBatches) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  // Mask shape: [batch=1, seq_len=3, 1, max_kv_len=8]
  auto layout = ::litert::Layout(::litert::Dimensions({4, 3, 1, 8}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 96);
  ASSERT_TRUE(mask_buffer);

  // Initialize the mask with the default bool value (false).
  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // Fill attention mask starting from timestep 2 for 3 steps.
  // channel_size = 8
  // i = 0: current_step = 2. Fills indices [0, 2] with true.
  // i = 1: current_step = 3. Fills indices [8, 11] with true.
  // i = 2: current_step = 4. Fills indices [16, 20] with true.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/2, /*steps=*/3,
                              proto::ATTENTION_MASK_TYPE_CAUSAL));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  for (int b = 0; b < 4; ++b) {
    for (int i = 0; i < 24; ++i) {
      if ((i >= 0 && i <= 2) || (i >= 8 && i <= 11) || (i >= 16 && i <= 20)) {
        EXPECT_TRUE(mask_ptr[b * 24 + i]) << " at index " << i;
      } else {
        EXPECT_FALSE(mask_ptr[b * 24 + i]) << " at index " << i;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, FillAttentionMask_Bidirectional) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  auto layout = ::litert::Layout(::litert::Dimensions({1, 7, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 70);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  std::vector<int> token_ids = {1, 2, 3, 4, 5, 6, 7};
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/0, /*steps=*/7,
                              proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL,
                              token_ids));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);

  // All queries (q=0 to 6) should attend to [0, 6]
  for (int q = 0; q < 7; ++q) {
    int offset = q * 10;
    for (int i = 0; i < 10; ++i) {
      if (i <= 6) {
        EXPECT_EQ(mask_ptr[offset + i], 0.0f) << "q=" << q << ", k=" << i;
      } else {
        EXPECT_LT(mask_ptr[offset + i], -1000.0f) << "q=" << q << ", k=" << i;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_Bidirectional_SlidingWindow) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  auto layout = ::litert::Layout(::litert::Dimensions({1, 7, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 70);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  std::vector<int> token_ids = {1, 2, 3, 4, 5, 6, 7};
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/0, /*steps=*/7,
                              proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL,
                              token_ids,
                              /*sliding_window_size=*/2));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);

  // For q=0 to 6, expected unmasked keys satisfy std::abs(q - k) < 2 and k <= 6
  for (int q = 0; q < 7; ++q) {
    int offset = q * 10;
    for (int k = 0; k < 10; ++k) {
      bool expected_unmasked = (std::abs(q - k) < 2) && (k <= 6);
      if (expected_unmasked) {
        EXPECT_EQ(mask_ptr[offset + k], 0.0f) << "q=" << q << ", k=" << k;
      } else {
        EXPECT_LT(mask_ptr[offset + k], -1000.0f) << "q=" << q << ", k=" << k;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_VisionBidirectional) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  auto layout = ::litert::Layout(::litert::Dimensions({1, 8, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 80);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  std::vector<int> token_ids = {10, -1, -1, -3, -1, -1, 12, 13};
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/0, /*steps=*/8,
                              proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL,
                              token_ids));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);

  // q = 0: Expected unmasked: 0
  int offset = 0;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k == 0);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=0, k=" << k;
  }

  // q = 1: Expected unmasked: 0, 1, 2
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 2);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=1, k=" << k;
  }

  // q = 2: Expected unmasked: 0, 1, 2
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 2);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=2, k=" << k;
  }

  // q = 3: Expected unmasked: 0, 1, 2, 3
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 3);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=3, k=" << k;
  }

  // q = 4: Expected unmasked: 0, 1, 2, 3, 4, 5
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 5);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=4, k=" << k;
  }

  // q = 5: Expected unmasked: 0, 1, 2, 3, 4, 5
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 5);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=5, k=" << k;
  }

  // q = 6: Expected unmasked: 0 to 6
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 6);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=6, k=" << k;
  }

  // q = 7: Expected unmasked: 0 to 7
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 7);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=7, k=" << k;
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_Causal_SlidingWindow) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  auto layout = ::litert::Layout(::litert::Dimensions({1, 3, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 30);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  // start_timestep = 2, steps = 3. (q = 2, 3, 4).
  // sliding_window_size = 2.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/2, /*steps=*/3,
                              proto::ATTENTION_MASK_TYPE_CAUSAL,
                              /*token_ids=*/std::nullopt,
                              /*sliding_window_size=*/2));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);

  // q = 2: Expected unmasked: [1, 2].
  int offset = 0;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 1 && k <= 2);
    if (expected_unmasked) {
      EXPECT_EQ(mask_ptr[offset + k], 0.0f) << "q=2, k=" << k;
    } else {
      EXPECT_LT(mask_ptr[offset + k], -1000.0f) << "q=2, k=" << k;
    }
  }

  // q = 3: Expected unmasked: [2, 3].
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 2 && k <= 3);
    if (expected_unmasked) {
      EXPECT_EQ(mask_ptr[offset + k], 0.0f) << "q=3, k=" << k;
    } else {
      EXPECT_LT(mask_ptr[offset + k], -1000.0f) << "q=3, k=" << k;
    }
  }

  // q = 4: Expected unmasked: [3, 4].
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 3 && k <= 4);
    if (expected_unmasked) {
      EXPECT_EQ(mask_ptr[offset + k], 0.0f) << "q=4, k=" << k;
    } else {
      EXPECT_LT(mask_ptr[offset + k], -1000.0f) << "q=4, k=" << k;
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_VisionBidirectional_SlidingWindow) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  auto layout = ::litert::Layout(::litert::Dimensions({1, 8, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 80);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(InitializeAttentionMask(*mask_buffer, /*is_f16=*/false));

  std::vector<int> token_ids = {10, -1, -1, -3, -1, -1, 12, 13};
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/0, /*steps=*/8,
                              proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL,
                              token_ids,
                              /*sliding_window_size=*/2));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  float* mask_ptr = static_cast<float*>(lock->second);

  // q = 0: Expected unmasked: 0
  int offset = 0;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k == 0);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=0, k=" << k;
  }

  // q = 1: Expected unmasked: 0, 1, 2
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k <= 2);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=1, k=" << k;
  }

  // q = 2: Expected unmasked: 1, 2
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 1 && k <= 2);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=2, k=" << k;
  }

  // q = 3: Expected unmasked: 2, 3
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 2 && k <= 3);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=3, k=" << k;
  }

  // q = 4: Expected unmasked: 3, 4, 5
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k == 3 || k == 4 || k == 5);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=4, k=" << k;
  }

  // q = 5: Expected unmasked: 4, 5
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k == 4 || k == 5);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=5, k=" << k;
  }

  // q = 6: Expected unmasked: 5, 6
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 5 && k <= 6);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=6, k=" << k;
  }

  // q = 7: Expected unmasked: 6, 7
  offset += 10;
  for (int k = 0; k < 10; ++k) {
    bool expected_unmasked = (k >= 6 && k <= 7);
    EXPECT_EQ(mask_ptr[offset + k] == 0.0f, expected_unmasked)
        << "q=7, k=" << k;
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_VisionBidirectional_EmptyTokensFails) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  auto layout = ::litert::Layout(::litert::Dimensions({1, 2, 1, 10}));
  RankedTensorType ranked_tensor_type(ElementType::Float32, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(float) * 20);
  ASSERT_TRUE(mask_buffer);

  EXPECT_THAT(FillAttentionMask(*mask_buffer, /*start_timestep=*/5, /*steps=*/2,
                                proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL,
                                /*token_ids=*/std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_RingBuffer_Prefill_Causal) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  // Shape: [1, 1, chunk_len=4, S + chunk_len=12] where S = 8.
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 4, 12}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 48);
  ASSERT_TRUE(mask_buffer);

  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/0, /*steps=*/4,
                              proto::ATTENTION_MASK_TYPE_CAUSAL,
                              /*token_ids=*/std::nullopt,
                              /*sliding_window_size=*/8,
                              RingBufferAttentionMaskMode::kPrefill));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  // For start_timestep = 0:
  // Cache columns [0..7]: all false.
  // Chunk columns [8..11]: causal mask for i in [0..3].
  for (int i = 0; i < 4; ++i) {
    int row_offset = i * 12;
    for (int col = 0; col < 8; ++col) {
      EXPECT_FALSE(mask_ptr[row_offset + col])
          << "i=" << i << ", cache col=" << col;
    }
    for (int j = 0; j < 4; ++j) {
      bool expected = (j <= i);
      EXPECT_EQ(mask_ptr[row_offset + 8 + j], expected)
          << "i=" << i << ", chunk col=" << j;
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_RingBuffer_Prefill_VisionBidirectional) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  // Shape: [1, 1, chunk_len=6, S + chunk_len=14] where S = 8.
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 6, 14}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 84);
  ASSERT_TRUE(mask_buffer);

  // token_ids: token 0 is text, tokens 1..3 are vision (-1), tokens 4..5 are
  // text.
  std::vector<int> token_ids = {10, -1, -1, -1, 11, 12};
  ASSERT_OK(FillAttentionMask(
      *mask_buffer, /*start_timestep=*/0, /*steps=*/6,
      proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL, token_ids,
      /*sliding_window_size=*/8, RingBufferAttentionMaskMode::kPrefill));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  // Row 0 (i=0, text token): attends to chunk col 8 (j=0) only.
  EXPECT_TRUE(mask_ptr[0 * 14 + 8 + 0]);
  EXPECT_FALSE(mask_ptr[0 * 14 + 8 + 1]);

  // Row 1 (i=1, vision token): attends to chunk cols 8, 9, 10, 11 (j=0..3).
  EXPECT_TRUE(mask_ptr[1 * 14 + 8 + 0]);
  EXPECT_TRUE(mask_ptr[1 * 14 + 8 + 1]);
  EXPECT_TRUE(mask_ptr[1 * 14 + 8 + 2]);
  EXPECT_TRUE(mask_ptr[1 * 14 + 8 + 3]);
  EXPECT_FALSE(mask_ptr[1 * 14 + 8 + 4]);

  // Row 2 (i=2, vision token): attends to chunk cols 8, 9, 10, 11 (j=0..3).
  EXPECT_TRUE(mask_ptr[2 * 14 + 8 + 0]);
  EXPECT_TRUE(mask_ptr[2 * 14 + 8 + 1]);
  EXPECT_TRUE(mask_ptr[2 * 14 + 8 + 2]);
  EXPECT_TRUE(mask_ptr[2 * 14 + 8 + 3]);
  EXPECT_FALSE(mask_ptr[2 * 14 + 8 + 4]);

  // Row 3 (i=3, vision token): attends to chunk cols 8, 9, 10, 11 (j=0..3).
  EXPECT_TRUE(mask_ptr[3 * 14 + 8 + 0]);
  EXPECT_TRUE(mask_ptr[3 * 14 + 8 + 1]);
  EXPECT_TRUE(mask_ptr[3 * 14 + 8 + 2]);
  EXPECT_TRUE(mask_ptr[3 * 14 + 8 + 3]);
  EXPECT_FALSE(mask_ptr[3 * 14 + 8 + 4]);

  // Row 4 (i=4, text token): causal up to j=4.
  EXPECT_TRUE(mask_ptr[4 * 14 + 8 + 4]);
  EXPECT_FALSE(mask_ptr[4 * 14 + 8 + 5]);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_RingBuffer_Decode) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  // Decode mask shape: [1, 1, 1, S=8].
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 1, 8}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 8);
  ASSERT_TRUE(mask_buffer);

  // Decode step at start_timestep = 3, window size = 8.
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/3, /*steps=*/1,
                              proto::ATTENTION_MASK_TYPE_CAUSAL,
                              /*token_ids=*/std::nullopt,
                              /*sliding_window_size=*/8,
                              RingBufferAttentionMaskMode::kDecode));

  auto lock = litert::TensorBufferScopedLock::Create(
      *mask_buffer, litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock);
  bool* mask_ptr = static_cast<bool*>(lock->second);

  // Slots 0..3 are attended, slots 4..7 are not yet written in cache.
  for (int k = 0; k < 8; ++k) {
    bool expected = (k <= 3);
    EXPECT_EQ(mask_ptr[k], expected) << "k=" << k;
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     FillAttentionMask_RingBuffer_Verifier) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  // Verifier mask shape: [1, 1, steps=4, S=1028] where sliding_window_size =
  // 1024.
  auto layout = ::litert::Layout(::litert::Dimensions({1, 1, 4, 1028}));
  RankedTensorType ranked_tensor_type(ElementType::Bool, std::move(layout));
  auto mask_buffer =
      TensorBuffer::CreateManaged(env, ::litert::TensorBufferType::kHostMemory,
                                  ranked_tensor_type, sizeof(bool) * 4 * 1028);
  ASSERT_TRUE(mask_buffer);

  // 1. Initial verification at start_timestep = 0:
  ASSERT_OK(FillAttentionMask(*mask_buffer, /*start_timestep=*/0, /*steps=*/4,
                              proto::ATTENTION_MASK_TYPE_CAUSAL,
                              /*token_ids=*/std::nullopt,
                              /*sliding_window_size=*/1024,
                              RingBufferAttentionMaskMode::kVerify));

  {
    auto lock = litert::TensorBufferScopedLock::Create(
        *mask_buffer, litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock);
    bool* mask_ptr = static_cast<bool*>(lock->second);

    for (int step = 0; step < 4; ++step) {
      int row_offset = step * 1028;
      for (int col = 0; col < 1028; ++col) {
        bool expected = (col <= step);
        EXPECT_EQ(mask_ptr[row_offset + col], expected)
            << "step=" << step << ", col=" << col;
      }
    }
  }

  // 2. Verification after wrap-around at start_timestep = 1500:
  ASSERT_OK(FillAttentionMask(
      *mask_buffer, /*start_timestep=*/1500, /*steps=*/4,
      proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL,
      /*token_ids=*/std::nullopt,
      /*sliding_window_size=*/1024, RingBufferAttentionMaskMode::kVerify));

  {
    auto lock = litert::TensorBufferScopedLock::Create(
        *mask_buffer, litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock);
    bool* mask_ptr = static_cast<bool*>(lock->second);

    for (int step = 0; step < 4; ++step) {
      int query_step = 1500 + step;
      int row_offset = step * 1028;
      for (int col = 0; col < 1028; ++col) {
        int diff = query_step - col;
        int distance = ((diff % 1028) + 1028) % 1028;
        bool expected = (distance <= 1023) && (distance <= query_step) &&
                        (distance <= (1028 - 4));
        EXPECT_EQ(mask_ptr[row_offset + col], expected)
            << "step=" << step << ", col=" << col;
      }
    }
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromPath) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";

  auto model_assets = ModelAssets::Create(model_path.string());
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     AotNpuOverrideEnablesFileBackedLoading) {
  const auto source_model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto source_file,
                       ScopedFile::Open(source_model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto source_loader,
                       LitertLmLoader::Create(std::move(source_file)));
  auto model_buffer =
      source_loader->GetTFLiteModel(ModelType::kTfLitePrefillDecode);
  ASSERT_GT(model_buffer.Size(), 0);

  const auto aot_model_path = std::filesystem::path(::testing::TempDir()) /
                              "aot_file_backed_override.litertlm";
  WriteAotTestModel(aot_model_path, model_buffer);
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(aot_model_path.string()));

  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      BuildLiteRtCompiledModelResources(
          model_assets, /*enable_file_backed_model_loading=*/false,
          /*enable_file_backed_for_aot_npu=*/true));
  ASSERT_OK_AND_ASSIGN(const auto* model, model_resources->GetTFLiteModel(
                                              ModelType::kTfLitePrefillDecode));
  EXPECT_TRUE(IsFileBacked(*model));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GenericNpuOverrideKeepsBufferBackedLoading) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));

  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      BuildLiteRtCompiledModelResources(
          model_assets, /*enable_file_backed_model_loading=*/false,
          /*enable_file_backed_for_aot_npu=*/true));
  ASSERT_OK_AND_ASSIGN(const auto* model, model_resources->GetTFLiteModel(
                                              ModelType::kTfLitePrefillDecode));
  EXPECT_FALSE(IsFileBacked(*model));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromScopedFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));

  auto model_assets =
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file)));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesTaskBundleFromMmapFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto mmap_file,
                       MemoryMappedFile::Create(model_path.string()));
  auto model_assets = ModelAssets::Create(std::move(mmap_file));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     BuildModelResourcesLitertLmFromMmapFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  ASSERT_OK_AND_ASSIGN(auto mmap_file,
                       MemoryMappedFile::Create(model_path.string()));
  auto model_assets = ModelAssets::Create(std::move(mmap_file));
  ASSERT_OK(model_assets);

  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       BuildLiteRtCompiledModelResources(*model_assets));
  ASSERT_NE(model_resources, nullptr);
  ASSERT_OK(model_resources->GetTFLiteModel(ModelType::kTfLitePrefillDecode));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCpuCacheOptions_WithScopedFile) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_options, ::litert::CpuOptions::Create());

  std::string test_file_path =
      (std::filesystem::path(::testing::TempDir()) / "cpu_cache_test.bin")
          .string();
  {
    std::ofstream touch_file(test_file_path);
  }
  ASSERT_OK_AND_ASSIGN(auto scoped_file,
                       ScopedFile::OpenWritable(test_file_path));
  auto scoped_file_ptr = std::make_shared<ScopedFile>(std::move(scoped_file));

  std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>
      weight_cache = scoped_file_ptr;
  ASSERT_OK(SetCpuCacheOptions(weight_cache, "test_prefix", cpu_options));

  auto fd_expected = cpu_options.GetXNNPackWeightCacheFileDescriptor();
  ASSERT_TRUE(fd_expected.HasValue());
  EXPECT_GT(fd_expected.Value(), 0);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCpuCacheOptions_WithWeightCacheFile) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_options, ::litert::CpuOptions::Create());

  std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>
      weight_cache = "weight_cache.bin";
  ASSERT_OK(SetCpuCacheOptions(weight_cache, "test_prefix", cpu_options));

  auto path_expected = cpu_options.GetXNNPackWeightCachePath();
  ASSERT_TRUE(path_expected.HasValue());
  EXPECT_EQ(path_expected.Value(), "weight_cache.bin");
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, SetCpuCacheOptions_WithoutCache) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_options, ::litert::CpuOptions::Create());

  ASSERT_OK(SetCpuCacheOptions(absl::NotFoundError("No cache file"),
                               "test_prefix", cpu_options));

  auto path_expected = cpu_options.GetXNNPackWeightCachePath();
  ASSERT_TRUE(path_expected.HasValue());
  EXPECT_TRUE(path_expected.Value().empty());

  auto fd_expected = cpu_options.GetXNNPackWeightCacheFileDescriptor();
  ASSERT_TRUE(fd_expected.HasValue());
  EXPECT_EQ(fd_expected.Value(), -1);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, SetGpuCacheOptions_Nocache) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, ::litert::GpuOptions::Create());

  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir(":nocache");

  ASSERT_OK(SetGpuCacheOptions(
      executor_settings.GetWeightCacheFile(),
      executor_settings.GetProgramCacheFile(), "test_key", "test_prefix",
      /*cache_compiled_shaders_only=*/false, gpu_options));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetGpuModelCacheData_Nocache) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir(":nocache");

  ASSERT_OK_AND_ASSIGN(auto cache_data,
                       GetGpuModelCacheData(executor_settings, "test_cache"));
  EXPECT_FALSE(cache_data.program_cache_file.ok());
  EXPECT_FALSE(cache_data.weight_cache_file.ok());
  EXPECT_TRUE(cache_data.cache_key.empty());
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetGpuModelCacheData_WithCache) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir("/dummy/cache/dir");

  // Create dummy scoped files to use as cache.
  std::string temp_prog_cache =
      (std::filesystem::path(::testing::TempDir()) / "prog_cache.bin").string();
  std::string temp_weight_cache =
      (std::filesystem::path(::testing::TempDir()) / "weight_cache.bin")
          .string();
  {
    std::ofstream touch1(temp_prog_cache);
    std::ofstream touch2(temp_weight_cache);
  }

  ASSERT_OK_AND_ASSIGN(auto prog_file,
                       ScopedFile::OpenWritable(temp_prog_cache));
  ASSERT_OK_AND_ASSIGN(auto weight_file,
                       ScopedFile::OpenWritable(temp_weight_cache));

  auto prog_file_ptr = std::make_shared<ScopedFile>(std::move(prog_file));
  auto weight_file_ptr = std::make_shared<ScopedFile>(std::move(weight_file));

  executor_settings.SetScopedProgramCacheFile(prog_file_ptr);
  executor_settings.SetScopedCacheFile(weight_file_ptr);

  ASSERT_OK_AND_ASSIGN(auto cache_data,
                       GetGpuModelCacheData(executor_settings, "test_cache"));

  ASSERT_OK(cache_data.program_cache_file);
  ASSERT_OK(cache_data.weight_cache_file);

  EXPECT_EQ(
      std::get<std::shared_ptr<ScopedFile>>(*cache_data.program_cache_file),
      prog_file_ptr);
  EXPECT_EQ(
      std::get<std::shared_ptr<ScopedFile>>(*cache_data.weight_cache_file),
      weight_file_ptr);

  ASSERT_OK_AND_ASSIGN(std::string expected_metadata_id,
                       GetFileCacheIdentifier(model_path.string()));
  std::string expected_cache_key =
      absl::StrCat("test_lm.task", "test_cache", "_", expected_metadata_id);
  EXPECT_EQ(cache_data.cache_key, expected_cache_key);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest, GetGpuModelCacheData_WithFd) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));

  ASSERT_OK_AND_ASSIGN(std::string expected_metadata_id,
                       GetFileCacheIdentifier(model_file));

  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file))));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir("/dummy/cache/dir");

  // Create dummy scoped files to use as cache.
  std::string temp_program_cache =
      (std::filesystem::path(::testing::TempDir()) / "program_cache.bin")
          .string();
  std::string temp_weight_cache =
      (std::filesystem::path(::testing::TempDir()) / "weight_cache.bin")
          .string();
  {
    std::ofstream touch1(temp_program_cache);
    std::ofstream touch2(temp_weight_cache);
  }

  ASSERT_OK_AND_ASSIGN(auto program_cache_file,
                       ScopedFile::OpenWritable(temp_program_cache));
  ASSERT_OK_AND_ASSIGN(auto weight_cache_file,
                       ScopedFile::OpenWritable(temp_weight_cache));

  auto program_cache_file_ptr =
      std::make_shared<ScopedFile>(std::move(program_cache_file));
  auto weight_cache_file_ptr =
      std::make_shared<ScopedFile>(std::move(weight_cache_file));

  executor_settings.SetScopedProgramCacheFile(program_cache_file_ptr);
  executor_settings.SetScopedCacheFile(weight_cache_file_ptr);

  ASSERT_OK_AND_ASSIGN(auto cache_data,
                       GetGpuModelCacheData(executor_settings, "test_cache"));

  ASSERT_OK(cache_data.program_cache_file);
  ASSERT_OK(cache_data.weight_cache_file);

  EXPECT_EQ(
      std::get<std::shared_ptr<ScopedFile>>(*cache_data.program_cache_file),
      program_cache_file_ptr);
  EXPECT_EQ(
      std::get<std::shared_ptr<ScopedFile>>(*cache_data.weight_cache_file),
      weight_cache_file_ptr);

  std::string expected_cache_key = absl::StrCat("fd_", expected_metadata_id);
  EXPECT_EQ(cache_data.cache_key, expected_cache_key);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetGpuModelCacheData_UnresolvablePathFallsBackToFd) {
  // Back the model with a real file (for a valid fd), but reference it by a
  // path that cannot be resolved via stat() -- this mimics the sandboxed
  // AiCoreIsolatedService, where the model is a virtual/relative MDD path.
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(std::string expected_metadata_id,
                       GetFileCacheIdentifier(model_file));

  constexpr absl::string_view kUnresolvablePath =
      "/nonexistent/dir/virtual_model.task";
  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file)),
                          kUnresolvablePath));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir("/dummy/cache/dir");

  // Provide valid cache fds so a cache key is generated.
  std::string temp_program_cache =
      (std::filesystem::path(::testing::TempDir()) / "fb_program_cache.bin")
          .string();
  std::string temp_weight_cache =
      (std::filesystem::path(::testing::TempDir()) / "fb_weight_cache.bin")
          .string();
  {
    std::ofstream touch1(temp_program_cache);
    std::ofstream touch2(temp_weight_cache);
  }
  ASSERT_OK_AND_ASSIGN(auto program_cache_file,
                       ScopedFile::OpenWritable(temp_program_cache));
  ASSERT_OK_AND_ASSIGN(auto weight_cache_file,
                       ScopedFile::OpenWritable(temp_weight_cache));
  executor_settings.SetScopedProgramCacheFile(
      std::make_shared<ScopedFile>(std::move(program_cache_file)));
  executor_settings.SetScopedCacheFile(
      std::make_shared<ScopedFile>(std::move(weight_cache_file)));

  ASSERT_OK_AND_ASSIGN(auto cache_data,
                       GetGpuModelCacheData(executor_settings, "test_cache"));

  // The key falls back to the fd-based identifier while keeping the
  // (unresolvable) path's basename.
  std::string expected_cache_key = absl::StrCat(
      "virtual_model.task", "test_cache", "_", expected_metadata_id);
  EXPECT_EQ(cache_data.cache_key, expected_cache_key);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetGpuModelCacheData_UnresolvablePathNoFdReturnsError) {
  constexpr absl::string_view kUnresolvablePath =
      "/nonexistent/dir/virtual_model.task";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(kUnresolvablePath));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir("/dummy/cache/dir");

  // No model fd is available, so the original unresolvable-path error is
  // propagated instead of being masked.
  auto cache_data = GetGpuModelCacheData(executor_settings, "test_cache");
  EXPECT_FALSE(cache_data.ok());
  EXPECT_EQ(cache_data.status().code(), absl::StatusCode::kInternal);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetGpuModelCacheData_ResolvablePathPrefersPathOverFd) {
  // Provide BOTH a resolvable path and a valid fd. The cache key must be
  // derived from the path identifier -- a regression guard ensuring the path
  // remains preferred (no fd-first reordering that would change existing keys).
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto model_assets,
      ModelAssets::Create(std::make_shared<ScopedFile>(std::move(model_file)),
                          model_path.string()));

  class StubExecutorSettings : public ExecutorSettingsBase {
   public:
    explicit StubExecutorSettings(const ModelAssets& model_assets)
        : ExecutorSettingsBase(model_assets) {}
  };
  StubExecutorSettings executor_settings(model_assets);
  executor_settings.SetCacheDir("/dummy/cache/dir");

  std::string temp_program_cache =
      (std::filesystem::path(::testing::TempDir()) / "rp_program_cache.bin")
          .string();
  std::string temp_weight_cache =
      (std::filesystem::path(::testing::TempDir()) / "rp_weight_cache.bin")
          .string();
  {
    std::ofstream touch1(temp_program_cache);
    std::ofstream touch2(temp_weight_cache);
  }
  ASSERT_OK_AND_ASSIGN(auto program_cache_file,
                       ScopedFile::OpenWritable(temp_program_cache));
  ASSERT_OK_AND_ASSIGN(auto weight_cache_file,
                       ScopedFile::OpenWritable(temp_weight_cache));
  executor_settings.SetScopedProgramCacheFile(
      std::make_shared<ScopedFile>(std::move(program_cache_file)));
  executor_settings.SetScopedCacheFile(
      std::make_shared<ScopedFile>(std::move(weight_cache_file)));

  ASSERT_OK_AND_ASSIGN(auto cache_data,
                       GetGpuModelCacheData(executor_settings, "test_cache"));

  ASSERT_OK_AND_ASSIGN(std::string expected_metadata_id,
                       GetFileCacheIdentifier(model_path.string()));
  std::string expected_cache_key =
      absl::StrCat("test_lm.task", "test_cache", "_", expected_metadata_id);
  EXPECT_EQ(cache_data.cache_key, expected_cache_key);
}

class DummyExecutorSettings : public ExecutorSettingsBase {
 public:
  DummyExecutorSettings()
      : ExecutorSettingsBase(*ModelAssets::Create("dummy_path")) {}
};

LiteRtDelegatePrecision GetGpuPrecision(const litert::GpuOptions& gpu_options) {
  LiteRtDelegatePrecision precision;
  EXPECT_EQ(LrtGetGpuAcceleratorCompilationOptionsPrecision(&precision,
                                                            gpu_options.Get()),
            kLiteRtStatusOk);
  return precision;
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCpuOptions_ConfigureSuccessfully) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_options, litert::CpuOptions::Create());
  EXPECT_OK(SetCpuOptions(cpu_options, 8));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCommonGpuOptions_SetsAllCommonOptions) {
  DummyExecutorSettings executor_settings;
  LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());

  EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options));

  bool constant_sharing = false;
  EXPECT_EQ(LrtGetGpuOptionsConstantTensorsSharing(&constant_sharing,
                                                   gpu_options.Get()),
            kLiteRtStatusOk);
  EXPECT_TRUE(constant_sharing);

  bool madvise = false;
  EXPECT_EQ(LrtGetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
                &madvise, gpu_options.Get()),
            kLiteRtStatusOk);
  EXPECT_TRUE(madvise);

  bool convert_weights = false;
  EXPECT_EQ(LrtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
                &convert_weights, gpu_options.Get()),
            kLiteRtStatusOk);
  EXPECT_TRUE(convert_weights);

  bool prefer_texture = false;
  EXPECT_EQ(LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
                &prefer_texture, gpu_options.Get()),
            kLiteRtStatusOk);
#if defined(__APPLE__)
  EXPECT_FALSE(prefer_texture);
#else
  EXPECT_TRUE(prefer_texture);
#endif
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCommonGpuOptions_DefaultAndFallbackPrecisions) {
  DummyExecutorSettings executor_settings;

  // Without fallback_activation_data_type specified, defaults to FLOAT32.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp32);
  }

  // With FLOAT32 fallback specified, configures kFp32.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT32));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp32);
  }

  // With FLOAT16, INT16, and INT8 fallback specified, configures kFp16.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT16));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp16);
  }
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::INT16));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp16);
  }
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::INT8));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp16);
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCommonGpuOptions_ExplicitActivationPrecisionOverride) {
  DummyExecutorSettings executor_settings;

  // Explicit FLOAT32 overrides fallback of FLOAT16/INT8.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    executor_settings.SetActivationDataType(ActivationDataType::FLOAT32);
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT16));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp32);
  }

  // Explicit INT8 or FLOAT16 overrides fallback of FLOAT32.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    executor_settings.SetActivationDataType(ActivationDataType::INT8);
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT32));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp16);
  }
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    executor_settings.SetActivationDataType(ActivationDataType::FLOAT16);
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT32));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp16);
  }
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetCommonGpuOptions_MixedPrecisionOverride) {
  DummyExecutorSettings executor_settings;
  executor_settings.SetEnableMixedPrecision(true);

  // Mixed precision overrides FLOAT16 fallback to FP32.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT16));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp32);
  }

  // Mixed precision overrides explicit FLOAT16 activation type to FP32.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
    executor_settings.SetActivationDataType(ActivationDataType::FLOAT16);
    EXPECT_OK(SetCommonGpuOptions(executor_settings, gpu_options,
                                  ActivationDataType::FLOAT16));
    EXPECT_EQ(GetGpuPrecision(gpu_options), kLiteRtDelegatePrecisionFp32);
  }
}

// A fake ModelResources whose weight-section lookup and backing file can be
// configured per test. Only the members exercised by GetExternalWeightResources
// are meaningful; all other lookups return errors.
class FakeExternalWeightModelResources : public ModelResources {
 public:
  FakeExternalWeightModelResources(
      absl::StatusOr<std::pair<size_t, size_t>> section_offset,
      std::optional<ScopedFile> scoped_file)
      : section_offset_(std::move(section_offset)),
        scoped_file_(std::move(scoped_file)) {}
  ~FakeExternalWeightModelResources() override = default;

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return section_offset_;
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    if (!scoped_file_.has_value()) {
      return absl::UnimplementedError("No scoped file.");
    }
    return std::ref(*scoped_file_);
  }

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError("Unimplemented");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("Unimplemented");
  }

 private:
  absl::StatusOr<std::pair<size_t, size_t>> section_offset_;
  std::optional<ScopedFile> scoped_file_;
};

// Returns the path to the shared test .task model.
std::string TestModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) /
          "litert_lm/runtime/testdata/test_lm.task")
      .string();
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetExternalWeightResources_SectionNotFound) {
  // A missing section means the model simply has no external weights, so an
  // empty result (not an error) is expected.
  FakeExternalWeightModelResources resources(
      absl::NotFoundError("No weights section."), /*scoped_file=*/std::nullopt);

  ASSERT_OK_AND_ASSIGN(
      auto external_weights,
      GetExternalWeightResources(resources, ModelType::kTfLitePrefillDecode));
  EXPECT_FALSE(external_weights.scoped_file.has_value());
  EXPECT_TRUE(external_weights.sections.empty());
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetExternalWeightResources_SectionUnimplemented) {
  // Resource formats that do not support external weight sections report
  // kUnimplemented, which is also treated as "no external weights".
  FakeExternalWeightModelResources resources(
      absl::UnimplementedError("Not supported."), /*scoped_file=*/std::nullopt);

  ASSERT_OK_AND_ASSIGN(
      auto external_weights,
      GetExternalWeightResources(resources, ModelType::kTfLitePrefillDecode));
  EXPECT_FALSE(external_weights.scoped_file.has_value());
  EXPECT_TRUE(external_weights.sections.empty());
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetExternalWeightResources_SectionErrorPropagated) {
  // Any other lookup error is a real failure and must be surfaced.
  FakeExternalWeightModelResources resources(
      absl::InvalidArgumentError("Bad section."), /*scoped_file=*/std::nullopt);

  EXPECT_THAT(
      GetExternalWeightResources(resources, ModelType::kTfLitePrefillDecode),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetExternalWeightResources_SectionPresent) {
  ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(TestModelPath()));

  // The section spans the byte range [100, 250), i.e. offset 100, length 150.
  FakeExternalWeightModelResources resources(
      std::pair<size_t, size_t>(100, 250), std::move(scoped_file));

  ASSERT_OK_AND_ASSIGN(
      auto external_weights,
      GetExternalWeightResources(resources, ModelType::kTfLitePrefillDecode));
  ASSERT_TRUE(external_weights.scoped_file.has_value());
  EXPECT_TRUE(external_weights.scoped_file->IsValid());
  ASSERT_TRUE(external_weights.sections.contains("tflite_weights"));
  const auto& section = external_weights.sections.at("tflite_weights");
  EXPECT_EQ(section.offset, 100);
  EXPECT_EQ(section.length, 150);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     SetExternalWeightOptions_NoExternalWeights) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options,
                              litert::Options::Create());
  FakeExternalWeightModelResources resources(
      absl::NotFoundError("No weights section."), /*scoped_file=*/std::nullopt);

  // With no external weights, the call is a no-op that still succeeds.
  EXPECT_OK(SetExternalWeightOptions(resources, ModelType::kTfLitePrefillDecode,
                                     compilation_options));
}

// NOTE: The "external weights present" branch of SetExternalWeightOptions is
// not unit-tested here. It forwards to
// litert::Options::SetExternalWeightScopedFile, which takes ownership of the
// weight source but only reclaims it when the Options object is built into a
// compiled model, something a unit test does not do. Exercising it therefore
// leaks the source and trips the heap checker. The substantive logic of that
// branch (locating the section and duplicating the backing file) is covered by
// GetExternalWeightResources_SectionPresent.

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetAttentionMaskParams_NullMetadata) {
  auto params = GetAttentionMaskParams(/*executor_metadata=*/nullptr);
  EXPECT_EQ(params.global_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
  EXPECT_EQ(params.local_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
  EXPECT_FALSE(params.sliding_window_size.has_value());
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetAttentionMaskParams_EmptyMetadata) {
  proto::ExecutorMetadata metadata;
  auto params = GetAttentionMaskParams(&metadata);
  EXPECT_EQ(params.global_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
  EXPECT_EQ(params.local_type, proto::ATTENTION_MASK_TYPE_CAUSAL);
  EXPECT_FALSE(params.sliding_window_size.has_value());
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetAttentionMaskParams_WithSettings) {
  proto::ExecutorMetadata metadata;
  auto* mask_settings = metadata.mutable_llm_executor_metadata()
                            ->mutable_attention_mask_settings();
  mask_settings->set_attention_mask_type(
      proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL);
  mask_settings->set_local_attention_mask_type(
      proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);
  mask_settings->set_sliding_window_size(512);

  auto params = GetAttentionMaskParams(&metadata);
  EXPECT_EQ(params.global_type,
            proto::ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL);
  EXPECT_EQ(params.local_type, proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);
  ASSERT_TRUE(params.sliding_window_size.has_value());
  EXPECT_EQ(*params.sliding_window_size, 512);
}

TEST(LlmLiteRTCompiledModelExecutorUtilsTest,
     GetAttentionMaskParams_LocalPolicyFallback) {
  proto::ExecutorMetadata metadata;
  auto* mask_settings = metadata.mutable_llm_executor_metadata()
                            ->mutable_attention_mask_settings();
  mask_settings->set_attention_mask_type(
      proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);

  auto params = GetAttentionMaskParams(&metadata);
  EXPECT_EQ(params.global_type, proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);
  EXPECT_EQ(params.local_type, proto::ATTENTION_MASK_TYPE_BIDIRECTIONAL);
  EXPECT_FALSE(params.sliding_window_size.has_value());
}

}  // namespace
}  // namespace litert::lm
