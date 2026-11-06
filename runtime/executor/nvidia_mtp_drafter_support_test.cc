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

#include "runtime/executor/nvidia_mtp_drafter_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_opaque_options.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_requirements.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/components/sampling_uniforms.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr auto kCudaBuffer =
    static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType);

TEST(NvidiaMtpDrafterSupportTest, RecognizesTransposedValueCache) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      kCudaBuffer, TensorBufferType::kHostMemory};
  const std::vector<uint32_t> strides = {2359296, 294912, 1, 256};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto requirements,
      TensorBufferRequirements::Create(buffer_types, 4718592, strides));

  ASSERT_OK_AND_ASSIGN(bool transposed,
                       UsesNvidiaTransposedValueCache(layout, requirements));
  EXPECT_TRUE(transposed);
}

TEST(NvidiaMtpDrafterSupportTest, LeavesDenseCudaValueCacheUnchanged) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {kCudaBuffer};
  for (const auto& strides :
       std::vector<std::vector<uint32_t>>{{}, {2359296, 294912, 1152, 1}}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto requirements,
        TensorBufferRequirements::Create(buffer_types, 4718592, strides));

    ASSERT_OK_AND_ASSIGN(bool transposed,
                         UsesNvidiaTransposedValueCache(layout, requirements));
    EXPECT_FALSE(transposed);
  }
}

TEST(NvidiaMtpDrafterSupportTest, LeavesOtherBackendStridesUnchanged) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      TensorBufferType::kOpenClBuffer};
  const std::vector<uint32_t> strides = {3000000, 375000, 1200, 1};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto requirements,
      TensorBufferRequirements::Create(buffer_types, 6000000, strides));

  ASSERT_OK_AND_ASSIGN(bool transposed,
                       UsesNvidiaTransposedValueCache(layout, requirements));
  EXPECT_FALSE(transposed);
}

TEST(NvidiaMtpDrafterSupportTest, RejectsUnsupportedCudaValueCacheStrides) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {kCudaBuffer};
  for (const auto& strides : std::vector<std::vector<uint32_t>>{
           {2359296, 294912, 2, 256}, {294912, 1, 256}}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto requirements,
        TensorBufferRequirements::Create(buffer_types, 4718592, strides));

    EXPECT_THAT(UsesNvidiaTransposedValueCache(layout, requirements),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(NvidiaMtpDrafterSupportTest, ReadOnlyValueCacheOptionsOwnSortedNames) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_2", "kv_cache_v_0"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto identifier, options.GetIdentifier());
  EXPECT_EQ(std::string(identifier.data(), identifier.size()),
            LITERT_NVIDIA_READ_ONLY_VALUE_CACHE_OPTIONS_ID);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto* payload, options.GetData<LiteRtNvidiaReadOnlyValueCacheOptions>());
  ASSERT_EQ(payload->num_inputs, 2u);
  EXPECT_STREQ(payload->input_names[0], "kv_cache_v_0");
  EXPECT_STREQ(payload->input_names[1], "kv_cache_v_2");
}

TEST(NvidiaMtpDrafterSupportTest, ReadOnlyValueCacheOptionsHashTheNameSet) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_2", "kv_cache_v_0"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto same_set,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_0", "kv_cache_v_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto other_set, CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_v_1"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto hash, options.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto same_hash, same_set.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto other_hash, other_set.Hash());
  EXPECT_EQ(hash, same_hash);
  EXPECT_NE(hash, other_hash);
}

TEST(NvidiaMtpDrafterSupportTest, SharedCacheInputsOwnSortedNames) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaSharedCacheInputs({"kv_cache_v_2", "kv_cache_k_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto identifier, options.GetIdentifier());
  EXPECT_EQ(std::string(identifier.data(), identifier.size()),
            LITERT_NVIDIA_SHARED_CACHE_INPUTS_ID);
  LITERT_ASSERT_OK_AND_ASSIGN(auto* payload,
                              options.GetData<LiteRtNvidiaSharedCacheInputs>());
  ASSERT_EQ(payload->num_inputs, 2u);
  EXPECT_STREQ(payload->input_names[0], "kv_cache_k_2");
  EXPECT_STREQ(payload->input_names[1], "kv_cache_v_2");
}

TEST(NvidiaMtpDrafterSupportTest, SharedCacheInputsHashTheNameSet) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateNvidiaSharedCacheInputs({"kv_cache_v_2", "kv_cache_k_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto same_set,
      CreateNvidiaSharedCacheInputs({"kv_cache_k_2", "kv_cache_v_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto read_only,
      CreateNvidiaReadOnlyValueCacheOptions({"kv_cache_k_2", "kv_cache_v_2"}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto hash, options.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto same_hash, same_set.Hash());
  LITERT_ASSERT_OK_AND_ASSIGN(auto read_only_hash, read_only.Hash());
  EXPECT_EQ(hash, same_hash);
  // The same names under the other identifier hash the same: the identifier
  // tells them apart in the options chain.
  EXPECT_EQ(hash, read_only_hash);
}

TEST(NvidiaMtpDrafterSupportTest, SharedValueCacheOptionsOnlyOnTheNpuRoute) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto drafter_model,
                              Model::CreateFromFile(model_path.string()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));
  // CPU and GPU never produce NVIDIA caches: the options are left alone
  // without looking at the models.
  for (Backend backend : {Backend::CPU, Backend::GPU}) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
    EXPECT_OK(AddNvidiaSharedValueCacheOptions(backend, drafter_model,
                                               base_model, options));
  }
}

TEST(NvidiaMtpDrafterSupportTest, GreedySamplerFallsBackOffNvidiaBuffers) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  // Host logits on the CPU backend, and host logits on the NPU route (whose
  // sampler decides on the first logits it sees), both sample on the host.
  for (Backend backend : {Backend::CPU, Backend::NPU}) {
    SCOPED_TRACE(static_cast<int>(backend));
    proto::SamplerParameters params;
    params.set_type(proto::SamplerParameters::TOP_P);
    params.set_k(1);
    params.set_p(0.0f);
    params.set_temperature(1.0f);
    params.set_seed(0);
    ASSERT_OK_AND_ASSIGN(auto sampler,
                         CreateMtpGreedySampler(
                             backend, /*batch_size=*/1, params, std::cref(env),
                             /*sequence_size=*/1, /*vocab_size=*/8,
                             ActivationDataType::FLOAT32));
    ASSERT_NE(sampler, nullptr);
    const std::vector<float> values = {0.1f, 0.2f, 0.7f, 0.3f,
                                       0.0f, 0.4f, 0.5f, 0.6f};
    LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                                CreateTensorBuffer<float>({1, 1, 8}));
    LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ids, CreateTensorBuffer<int32_t>({1, 1}));
    const std::vector<float> uniforms = {0.5f};
    for (int round = 0; round < 2; ++round) {
      ASSERT_OK(
          sampler->SampleToIdAndScoreBuffer(logits, ids, nullptr, uniforms));
      std::vector<int32_t> id(1);
      LITERT_ASSERT_OK(ids.Read<int32_t>(absl::MakeSpan(id)));
      EXPECT_EQ(id[0], 2);
    }
  }
}

TEST(NvidiaMtpDrafterSupportTest, GreedySamplerValidatesUniforms) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  for (Backend backend : {Backend::CPU, Backend::NPU}) {
    SCOPED_TRACE(static_cast<int>(backend));
    proto::SamplerParameters params;
    params.set_type(proto::SamplerParameters::TOP_P);
    params.set_k(1);
    params.set_p(0.0f);
    params.set_temperature(1.0f);
    params.set_seed(0);
    ASSERT_OK_AND_ASSIGN(auto sampler,
                         CreateMtpGreedySampler(
                             backend, /*batch_size=*/1, params, std::cref(env),
                             /*sequence_size=*/1, /*vocab_size=*/8,
                             ActivationDataType::FLOAT32));
    ASSERT_NE(sampler, nullptr);
    const std::vector<float> values = {0.1f, 0.2f, 0.7f, 0.3f,
                                       0.0f, 0.4f, 0.5f, 0.6f};
    LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                                CreateTensorBuffer<float>({1, 1, 8}));
    LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ids, CreateTensorBuffer<int32_t>({1, 1}));

    // Greedy sampling ignores the value of a valid uniform.
    const std::vector<float> uniforms = {0.5f};
    ASSERT_OK(
        sampler->SampleToIdAndScoreBuffer(logits, ids, nullptr, uniforms));
    std::vector<int32_t> id(1);
    LITERT_ASSERT_OK(ids.Read<int32_t>(absl::MakeSpan(id)));
    EXPECT_EQ(id[0], 2);

    for (const std::vector<float>& invalid :
         std::vector<std::vector<float>>{{}, {0.5f, 0.5f}, {1.0f}}) {
      EXPECT_THAT(
          sampler->SampleToIdAndScoreBuffer(logits, ids, nullptr, invalid),
          StatusIs(absl::StatusCode::kInvalidArgument));
    }
  }
}

proto::SamplerParameters GreedyParameters() {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(1);
  params.set_p(0.0f);
  params.set_temperature(1.0f);
  return params;
}

TEST(NvidiaMtpDrafterSupportTest, SupportsDecodeAndBatchedVerifierFloatLogits) {
  for (auto dtype : {ElementType::Float16, ElementType::Float32}) {
    for (int rows : {1, 4}) {
      ASSERT_OK_AND_ASSIGN(
          bool supported,
          SupportsNvidiaGreedySampler(
              Backend::NPU, GreedyParameters(), kCudaBuffer,
              RankedTensorType(dtype, Layout(Dimensions{1, rows, 262144}))));
      EXPECT_TRUE(supported);
    }
  }
}

TEST(NvidiaMtpDrafterSupportTest,
     LeavesOtherBackendsTypesAndConfigsOnExistingPath) {
  const RankedTensorType type(ElementType::Float32,
                              Layout(Dimensions{1, 4, 8}));
  for (Backend backend : {Backend::CPU, Backend::GPU}) {
    ASSERT_OK_AND_ASSIGN(
        bool supported, SupportsNvidiaGreedySampler(backend, GreedyParameters(),
                                                    kCudaBuffer, type));
    EXPECT_FALSE(supported);
  }
  ASSERT_OK_AND_ASSIGN(
      bool host_supported,
      SupportsNvidiaGreedySampler(Backend::NPU, GreedyParameters(),
                                  TensorBufferType::kHostMemory, type));
  EXPECT_FALSE(host_supported);
  ASSERT_OK_AND_ASSIGN(
      bool int_supported,
      SupportsNvidiaGreedySampler(
          Backend::NPU, GreedyParameters(), kCudaBuffer,
          RankedTensorType(ElementType::Int32, Layout(type.Layout()))));
  EXPECT_FALSE(int_supported);
  for (int variant = 0; variant < 3; ++variant) {
    auto params = GreedyParameters();
    if (variant == 0) {
      params.set_k(2);
    } else if (variant == 1) {
      params.set_p(0.9f);
    } else {
      params.set_type(proto::SamplerParameters::TOP_K);
    }
    ASSERT_OK_AND_ASSIGN(
        bool supported,
        SupportsNvidiaGreedySampler(Backend::NPU, params, kCudaBuffer, type));
    EXPECT_FALSE(supported);
  }
}

TEST(NvidiaMtpDrafterSupportTest, RejectsMalformedMatchingLogitsShape) {
  for (const Dimensions& dims :
       std::vector<Dimensions>{{4, 8}, {1, 0, 8}, {1, 4, -1}}) {
    EXPECT_THAT(SupportsNvidiaGreedySampler(
                    Backend::NPU, GreedyParameters(), kCudaBuffer,
                    RankedTensorType(ElementType::Float32, Layout(dims))),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // A batch above one is not malformed: the host samples it.
  ASSERT_OK_AND_ASSIGN(
      const bool batched,
      SupportsNvidiaGreedySampler(
          Backend::NPU, GreedyParameters(), kCudaBuffer,
          RankedTensorType(ElementType::Float32, Layout(Dimensions{2, 4, 8}))));
  EXPECT_FALSE(batched);
}

TEST(NvidiaMtpDrafterSupportTest, AcceptsDenseStridesAndRejectsPaddedLogits) {
  const Dimensions dims{1, 4, 8};
  const Strides dense{32, 8, 1};
  ASSERT_OK_AND_ASSIGN(
      bool supported,
      SupportsNvidiaGreedySampler(
          Backend::NPU, GreedyParameters(), kCudaBuffer,
          RankedTensorType(ElementType::Float16, Layout(dims, dense))));
  EXPECT_TRUE(supported);
  const Strides padded{36, 9, 1};
  EXPECT_THAT(SupportsNvidiaGreedySampler(
                  Backend::NPU, GreedyParameters(), kCudaBuffer,
                  RankedTensorType(ElementType::Float16, Layout(dims, padded))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

#if defined(__linux__)
class ScopedSamplingFlag {
 public:
  explicit ScopedSamplingFlag(const char* value) {
    if (const char* old = std::getenv(kName)) {
      old_ = old;
    }
    if (value) {
      setenv(kName, value, 1);
    } else {
      unsetenv(kName);
    }
  }
  ~ScopedSamplingFlag() {
    if (old_) {
      setenv(kName, old_->c_str(), 1);
    } else {
      unsetenv(kName);
    }
  }

 private:
  static constexpr char kName[] = "LITERT_NVIDIA_MTP_GPU_SAMPLING";
  std::optional<std::string> old_;
};

TEST(NvidiaMtpDrafterSupportTest, HostLogitsOrOptOutKeepExistingSampler) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float32, Layout(Dimensions{1, 1, 8})),
          8 * sizeof(float)));
  for (const char* value : std::vector<const char*>{nullptr, "0", "", "true"}) {
    ScopedSamplingFlag flag(value);
    ASSERT_OK_AND_ASSIGN(
        auto sampler, TryCreateNvidiaGreedySampler(env, Backend::NPU,
                                                   GreedyParameters(), logits));
    EXPECT_EQ(sampler, nullptr);
  }
}

TEST(NvidiaMtpDrafterSupportTest, StillRequiresActualCudaBuffer) {
  ScopedSamplingFlag flag("1");
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto logits,
      TensorBuffer::CreateManagedHostMemory(
          RankedTensorType(ElementType::Float32, Layout(Dimensions{1, 1, 8})),
          8 * sizeof(float)));
  ASSERT_OK_AND_ASSIGN(
      auto sampler, TryCreateNvidiaGreedySampler(env, Backend::NPU,
                                                 GreedyParameters(), logits));
  EXPECT_EQ(sampler, nullptr);
}
#endif  // defined(__linux__)

// The candidate path's policy, against the CPU sampler it reuses.

struct Candidate {
  int32_t id;
  float value;
};

// The k largest by logit then index: what NVIDIA's dispatch selects.
std::vector<Candidate> OrderedCandidates(const float* values, int vocab,
                                         int k) {
  std::vector<int32_t> order(vocab);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int32_t a, int32_t b) {
                      if (values[a] != values[b]) return values[a] > values[b];
                      return a < b;
                    });
  std::vector<Candidate> out;
  for (int i = 0; i < k; ++i) out.push_back({order[i], values[order[i]]});
  return out;
}

// The CPU sampler's own selection (TopKTokenIds in runtime/components/
// sampling_cpu_util.cc), as candidates.
std::vector<Candidate> HeapCandidates(const float* values, int vocab, int k) {
  auto ids = TopKTokenIds(absl::MakeConstSpan(values, vocab), k, 1, 1);
  EXPECT_TRUE(ids.ok()) << ids.status();
  std::vector<Candidate> out;
  for (int id : (*ids)[0]) out.push_back({id, values[id]});
  return out;
}

bool CutoffIsUnique(const float* values, int vocab, int k) {
  if (k >= vocab) return true;
  std::vector<float> sorted(values, values + vocab);
  std::partial_sort(sorted.begin(), sorted.begin() + k + 1, sorted.end(),
                    std::greater<float>());
  return sorted[k - 1] != sorted[k];
}

proto::SamplerParameters TopPParameters(int k, float p, float temperature,
                                        int seed) {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(k);
  params.set_p(p);
  params.set_temperature(temperature);
  params.set_seed(seed);
  return params;
}

std::vector<int32_t> ReadIds(TensorBuffer& ids, int rows) {
  std::vector<int32_t> out(rows);
  EXPECT_TRUE(ids.Read<int32_t>(absl::MakeSpan(out)).HasValue());
  return out;
}

std::vector<float> ReadScores(TensorBuffer& scores, int rows) {
  std::vector<float> out(rows);
  EXPECT_TRUE(scores.Read<float>(absl::MakeSpan(out)).HasValue());
  return out;
}

// On every configuration the drafter can be given, the policy on the
// device's candidates draws what TopPSampler draws on the full logits:
// the same ids, the same scores, with the same uniforms, twice over.
TEST(NvidiaMtpDrafterSupportTest, CandidatePolicyIsTheCpuSamplerBitForBit) {
  constexpr int kVocab = 4096;
  std::mt19937 gen(20261027);
  std::normal_distribution<float> dist(0.0f, 4.0f);
  int configurations = 0;
  for (int rows : {1, 4}) {
    for (int k : {1, 7, 64, 1024}) {
      for (float p : {0.0f, 0.3f, 0.95f, 1.0f}) {
        for (float temperature : {0.0f, 0.7f, 1.0f}) {
          SCOPED_TRACE(absl::StrCat("rows ", rows, " k ", k, " p ", p, " T ",
                                    temperature));
          std::vector<float> values(static_cast<size_t>(rows) * kVocab);
          for (float& value : values) value = dist(gen);
          for (int row = 0; row < rows; ++row) {
            ASSERT_TRUE(
                CutoffIsUnique(values.data() + row * kVocab, kVocab, k));
          }
          LITERT_ASSERT_OK_AND_ASSIGN(
              auto logits, CreateTensorBuffer<float>({1, rows, kVocab}));
          LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
          const proto::SamplerParameters params =
              TopPParameters(k, p, temperature, 7);
          ASSERT_OK_AND_ASSIGN(auto cpu,
                               TopPSampler::Create(k, p, temperature, 1, rows));
          ASSERT_OK(cpu->UpdateConfig(params, 1));
          std::vector<int32_t> candidate_ids;
          std::vector<float> candidate_values;
          for (int row = 0; row < rows; ++row) {
            for (const Candidate& candidate :
                 OrderedCandidates(values.data() + row * kVocab, kVocab, k)) {
              candidate_ids.push_back(candidate.id);
              candidate_values.push_back(candidate.value);
            }
          }
          std::default_random_engine generator(7);
          for (int draw = 0; draw < 2; ++draw) {
            // The same uniforms for both, as the executor draws them.
            std::vector<float> uniforms(rows);
            DrawSamplingUniforms(params, generator, absl::MakeSpan(uniforms));
            LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_ids,
                                        CreateTensorBuffer<int32_t>({1, rows}));
            LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_scores,
                                        CreateTensorBuffer<float>({1, rows}));
            ASSERT_OK(cpu->SampleToIdAndScoreBuffer(logits, cpu_ids,
                                                    &cpu_scores, uniforms));
            LITERT_ASSERT_OK_AND_ASSIGN(auto ids,
                                        CreateTensorBuffer<int32_t>({1, rows}));
            LITERT_ASSERT_OK_AND_ASSIGN(auto scores,
                                        CreateTensorBuffer<float>({1, rows}));
            ASSERT_OK(SampleNvidiaCandidates(candidate_ids, candidate_values,
                                             rows, k, p, temperature, uniforms,
                                             ids, &scores));
            EXPECT_EQ(ReadIds(ids, rows), ReadIds(cpu_ids, rows));
            EXPECT_EQ(ReadScores(scores, rows), ReadScores(cpu_scores, rows));
          }
          ++configurations;
        }
      }
    }
  }
  EXPECT_EQ(configurations, 96);
}

// Float16 logits tie often, and a constraint's mask leaves fewer tokens
// than k at a finite logit: the ties among the masked tokens at the cutoff
// are tokens of zero probability, and the draws still agree.
TEST(NvidiaMtpDrafterSupportTest, CandidatePolicyWithFp16ValuesAndMasks) {
  constexpr int kVocab = 8192;
  constexpr int kRows = 4;
  constexpr int kK = 64;
  std::mt19937 gen(5);
  std::normal_distribution<float> dist(0.0f, 4.0f);
  std::vector<float> values(static_cast<size_t>(kRows) * kVocab);
  for (float& value : values) {
    value = static_cast<float>(static_cast<tflite::half>(dist(gen)));
  }
  ASSERT_OK_AND_ASSIGN(auto mask_runner,
                       LiteRtLogitMaskRunner::CreateForHost());
  LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                              CreateTensorBuffer<float>({1, kRows, kVocab}));
  LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
  // Row 0: a bitmap that allows ten tokens. Row 1: one token. Row 2: a
  // sparse bias on three tokens. Row 3: nothing.
  std::vector<std::unique_ptr<LogitMask>> masks;
  {
    const std::vector<int> allowed = {3,    100,  101,  2000, 2001,
                                      2002, 5000, 7000, 8000, 8191};
    masks.push_back(BitmapLogitMask::CreateFromAllowedTokens(kVocab, allowed));
  }
  masks.push_back(BitmapLogitMask::CreateSingleAllowedToken(kVocab, 77));
  masks.push_back(std::make_unique<SparseLogitMask>(
      std::vector<SparseLogitMask::Entry>{{10, 1.0f, 30.0f, false},
                                          {11, 0.5f, 0.0f, false},
                                          {12, 1.0f, -100.0f, false}}));
  masks.push_back(nullptr);
  ASSERT_OK(mask_runner->ApplySequence(logits, masks));
  std::vector<float> masked(values.size());
  LITERT_ASSERT_OK(logits.Read<float>(absl::MakeSpan(masked)));
  for (float p : {0.95f, 1.0f}) {
    const proto::SamplerParameters params = TopPParameters(kK, p, 1.0f, 3);
    ASSERT_OK_AND_ASSIGN(auto cpu, TopPSampler::Create(kK, p, 1.0f, 1, kRows));
    ASSERT_OK(cpu->UpdateConfig(params, 1));
    std::vector<int32_t> candidate_ids;
    std::vector<float> candidate_values;
    for (int row = 0; row < kRows; ++row) {
      for (const Candidate& candidate :
           OrderedCandidates(masked.data() + row * kVocab, kVocab, kK)) {
        candidate_ids.push_back(candidate.id);
        candidate_values.push_back(candidate.value);
      }
    }
    std::default_random_engine generator(3);
    for (int draw = 0; draw < 8; ++draw) {
      std::vector<float> uniforms(kRows);
      DrawSamplingUniforms(params, generator, absl::MakeSpan(uniforms));
      LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_ids,
                                  CreateTensorBuffer<int32_t>({1, kRows}));
      LITERT_ASSERT_OK_AND_ASSIGN(auto cpu_scores,
                                  CreateTensorBuffer<float>({1, kRows}));
      ASSERT_OK(cpu->SampleToIdAndScoreBuffer(logits, cpu_ids, &cpu_scores,
                                              uniforms));
      LITERT_ASSERT_OK_AND_ASSIGN(auto ids,
                                  CreateTensorBuffer<int32_t>({1, kRows}));
      LITERT_ASSERT_OK_AND_ASSIGN(auto scores,
                                  CreateTensorBuffer<float>({1, kRows}));
      ASSERT_OK(SampleNvidiaCandidates(candidate_ids, candidate_values, kRows,
                                       kK, p, 1.0f, uniforms, ids, &scores));
      const std::vector<int32_t> got = ReadIds(ids, kRows);
      // The tokens may differ only where the uniform lands on tied
      // probabilities (see SampleNvidiaCandidates): such tokens share a logit.
      const std::vector<int32_t> cpu_got = ReadIds(cpu_ids, kRows);
      for (int row = 0; row < kRows; ++row) {
        if (got[row] != cpu_got[row]) {
          EXPECT_EQ(masked[row * kVocab + got[row]],
                    masked[row * kVocab + cpu_got[row]])
              << "p " << p << " draw " << draw << " row " << row;
        }
      }
      EXPECT_EQ(ReadScores(scores, kRows), ReadScores(cpu_scores, kRows));
      EXPECT_EQ(got[1], 77);
    }
  }
}

// Where the k-th value is shared, the CPU's heap keeps a subset of the tied
// tokens that its layout decides, and the device the lowest indices: the
// two selections hold the same logits, and differ only in which tied
// tokens, each with the smallest probability of the k, they carry.
TEST(NvidiaMtpDrafterSupportTest, CutoffTiesFollowTheIndexRuleNotTheHeap) {
  constexpr int kVocab = 16;
  constexpr int kK = 4;
  std::mt19937 gen(11);
  std::uniform_int_distribution<int> level(0, 2);
  int differing = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<float> values(kVocab);
    for (float& value : values) value = static_cast<float>(level(gen));
    const std::vector<Candidate> heap =
        HeapCandidates(values.data(), kVocab, kK);
    const std::vector<Candidate> device =
        OrderedCandidates(values.data(), kVocab, kK);
    ASSERT_EQ(heap.size(), device.size());
    std::vector<float> heap_values, device_values;
    for (int i = 0; i < kK; ++i) {
      heap_values.push_back(heap[i].value);
      device_values.push_back(device[i].value);
    }
    EXPECT_EQ(heap_values, device_values);
    bool same = true;
    for (int i = 0; i < kK; ++i) same = same && heap[i].id == device[i].id;
    if (!same) {
      ++differing;
      // Every token the two selections disagree on carries the k-th value.
      for (int i = 0; i < kK; ++i) {
        if (heap[i].id != device[i].id) {
          EXPECT_EQ(heap[i].value, device[kK - 1].value);
          EXPECT_EQ(device[i].value, device[kK - 1].value);
        }
      }
    }
    if (CutoffIsUnique(values.data(), kVocab, kK)) {
      EXPECT_TRUE(same);
    }
  }
  EXPECT_GT(differing, 0) << "the exception exists, and is documented";
}

// The NPU route's sampler, given host logits, keeps the factory's
// semantics across configuration updates: the recreation when k outgrows
// the sampler, and the same tokens for the same uniforms.
TEST(NvidiaMtpDrafterSupportTest, VerifierSamplerKeepsTheFactorySemantics) {
  constexpr int kVocab = 512;
  constexpr int kRows = 3;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  std::mt19937 gen(9);
  std::normal_distribution<float> dist(0.0f, 3.0f);
  std::vector<float> values(static_cast<size_t>(kRows) * kVocab);
  for (float& value : values) value = dist(gen);
  LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                              CreateTensorBuffer<float>({1, kRows, kVocab}));
  LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
  ASSERT_OK_AND_ASSIGN(
      auto reference,
      CreateSampler(Backend::CPU, 1, GreedyParameters(), std::cref(env), kRows,
                    kVocab, ActivationDataType::FLOAT32));
  ASSERT_OK_AND_ASSIGN(
      auto npu, CreateMtpGreedySampler(Backend::NPU, 1, GreedyParameters(),
                                       std::cref(env), kRows, kVocab,
                                       ActivationDataType::FLOAT32));
  int reference_max_k = 1;
  int npu_max_k = 1;
  std::default_random_engine generator(11);
  const auto compare = [&](const char* what,
                           const proto::SamplerParameters& params, int draws) {
    for (int draw = 0; draw < draws; ++draw) {
      std::vector<float> uniforms(kRows);
      DrawSamplingUniforms(params, generator, absl::MakeSpan(uniforms));
      LITERT_ASSERT_OK_AND_ASSIGN(auto ids_a,
                                  CreateTensorBuffer<int32_t>({1, kRows}));
      LITERT_ASSERT_OK_AND_ASSIGN(auto scores_a,
                                  CreateTensorBuffer<float>({1, kRows}));
      LITERT_ASSERT_OK_AND_ASSIGN(auto ids_b,
                                  CreateTensorBuffer<int32_t>({1, kRows}));
      LITERT_ASSERT_OK_AND_ASSIGN(auto scores_b,
                                  CreateTensorBuffer<float>({1, kRows}));
      ASSERT_OK(reference->SampleToIdAndScoreBuffer(logits, ids_a, &scores_a,
                                                    uniforms));
      ASSERT_OK(
          npu->SampleToIdAndScoreBuffer(logits, ids_b, &scores_b, uniforms));
      EXPECT_EQ(ReadIds(ids_b, kRows), ReadIds(ids_a, kRows)) << what;
      EXPECT_EQ(ReadScores(scores_b, kRows), ReadScores(scores_a, kRows))
          << what;
    }
  };
  const auto update = [&](const proto::SamplerParameters& params) {
    ASSERT_OK(UpdateOrRecreateSampler(reference, reference_max_k, Backend::CPU,
                                      1, params, std::cref(env), kRows, kVocab,
                                      ActivationDataType::FLOAT32));
    ASSERT_OK(UpdateOrRecreateMtpVerifierSampler(
        npu, npu_max_k, Backend::NPU, 1, params, std::cref(env), kRows, kVocab,
        ActivationDataType::FLOAT32));
    EXPECT_EQ(npu_max_k, reference_max_k);
  };
  compare("greedy", GreedyParameters(), 2);
  // k outgrows the sampler: recreated. The drafter names the CPU as its
  // session sampler's backend on the NPU route; the NPU route's sampler stays
  // what it is.
  Sampler* const npu_sampler = npu.get();
  ASSERT_OK(UpdateOrRecreateSampler(reference, reference_max_k, Backend::CPU, 1,
                                    TopPParameters(64, 0.95f, 1.0f, 3),
                                    std::cref(env), kRows, kVocab,
                                    ActivationDataType::FLOAT32));
  ASSERT_OK(UpdateOrRecreateMtpVerifierSampler(
      npu, npu_max_k, Backend::CPU, 1, TopPParameters(64, 0.95f, 1.0f, 3),
      std::cref(env), kRows, kVocab, ActivationDataType::FLOAT32));
  EXPECT_EQ(npu.get(), npu_sampler) << "the NPU route's sampler is kept";
  EXPECT_EQ(npu_max_k, reference_max_k);
  compare("k 64", TopPParameters(64, 0.95f, 1.0f, 3), 3);
  // A smaller k keeps the sampler.
  update(TopPParameters(8, 0.5f, 0.8f, 99));
  compare("k 8", TopPParameters(8, 0.5f, 0.8f, 99), 3);
  // k outgrows the sampler again.
  update(TopPParameters(100, 0.9f, 1.0f, 5));
  compare("k 100", TopPParameters(100, 0.9f, 1.0f, 5), 3);
  EXPECT_EQ(reference_max_k, 100);
}

// The generic factory returns no sampler for an unspecified type: the NPU
// route's sampler refuses the update, as UpdateOrRecreateSampler does, and
// keeps the sampler it had.
TEST(NvidiaMtpDrafterSupportTest, AFactoryRefusalLeavesTheKeptSamplerUsable) {
  constexpr int kVocab = 512;
  constexpr int kRows = 3;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  std::mt19937 gen(21);
  std::normal_distribution<float> dist(0.0f, 3.0f);
  std::vector<float> values(static_cast<size_t>(kRows) * kVocab);
  for (float& value : values) value = dist(gen);
  LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                              CreateTensorBuffer<float>({1, kRows, kVocab}));
  LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
  ASSERT_OK_AND_ASSIGN(
      auto reference,
      CreateSampler(Backend::CPU, 1, GreedyParameters(), std::cref(env), kRows,
                    kVocab, ActivationDataType::FLOAT32));
  ASSERT_OK_AND_ASSIGN(
      auto npu, CreateMtpGreedySampler(Backend::NPU, 1, GreedyParameters(),
                                       std::cref(env), kRows, kVocab,
                                       ActivationDataType::FLOAT32));
  int reference_max_k = 1;
  int npu_max_k = 1;
  proto::SamplerParameters unspecified;
  unspecified.set_k(8);
  EXPECT_THAT(UpdateOrRecreateSampler(reference, reference_max_k, Backend::CPU,
                                      1, unspecified, std::cref(env), kRows,
                                      kVocab, ActivationDataType::FLOAT32),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(UpdateOrRecreateMtpVerifierSampler(
                  npu, npu_max_k, Backend::CPU, 1, unspecified, std::cref(env),
                  kRows, kVocab, ActivationDataType::FLOAT32),
              StatusIs(absl::StatusCode::kInternal));
  ASSERT_NE(npu, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto ids_a,
                              CreateTensorBuffer<int32_t>({1, kRows}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto ids_b,
                              CreateTensorBuffer<int32_t>({1, kRows}));
  const std::vector<float> uniforms(kRows, 0.0f);
  ASSERT_OK(
      reference->SampleToIdAndScoreBuffer(logits, ids_a, nullptr, uniforms));
  ASSERT_OK(npu->SampleToIdAndScoreBuffer(logits, ids_b, nullptr, uniforms));
  EXPECT_EQ(ReadIds(ids_b, kRows), ReadIds(ids_a, kRows));
}

// The recreation arguments of an update reach the host sampler the NPU
// route's sampler makes, as the factory's do; a refused update leaves the
// caller's capacity as it was.
TEST(NvidiaMtpDrafterSupportTest, VerifierUpdateTakesTheRecreationArguments) {
  constexpr int kVocab = 512;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  std::mt19937 gen(31);
  std::normal_distribution<float> dist(0.0f, 3.0f);
  std::vector<float> values(static_cast<size_t>(3) * kVocab);
  for (float& value : values) value = dist(gen);
  LITERT_ASSERT_OK_AND_ASSIGN(auto logits,
                              CreateTensorBuffer<float>({1, 3, kVocab}));
  LITERT_ASSERT_OK(logits.Write<float>(absl::MakeConstSpan(values)));
  // Made for four rows, updated for three: the update's sequence size holds.
  ASSERT_OK_AND_ASSIGN(
      auto reference,
      CreateSampler(Backend::CPU, 1, GreedyParameters(), std::cref(env), 4,
                    kVocab, ActivationDataType::FLOAT32));
  ASSERT_OK_AND_ASSIGN(
      auto npu, CreateMtpGreedySampler(Backend::NPU, 1, GreedyParameters(),
                                       std::cref(env), 4, kVocab,
                                       ActivationDataType::FLOAT32));
  int reference_max_k = 1;
  int npu_max_k = 1;
  const proto::SamplerParameters params = TopPParameters(64, 0.95f, 1.0f, 3);
  ASSERT_OK(UpdateOrRecreateSampler(reference, reference_max_k, Backend::CPU, 1,
                                    params, std::cref(env), 3, kVocab,
                                    ActivationDataType::FLOAT32));
  ASSERT_OK(UpdateOrRecreateMtpVerifierSampler(
      npu, npu_max_k, Backend::CPU, 1, params, std::cref(env), 3, kVocab,
      ActivationDataType::FLOAT32));
  EXPECT_EQ(npu_max_k, 64);
  std::default_random_engine generator(3);
  for (int draw = 0; draw < 3; ++draw) {
    std::vector<float> uniforms(3);
    DrawSamplingUniforms(params, generator, absl::MakeSpan(uniforms));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ids_a,
                                CreateTensorBuffer<int32_t>({1, 3}));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ids_b,
                                CreateTensorBuffer<int32_t>({1, 3}));
    ASSERT_OK(
        reference->SampleToIdAndScoreBuffer(logits, ids_a, nullptr, uniforms));
    ASSERT_OK(npu->SampleToIdAndScoreBuffer(logits, ids_b, nullptr, uniforms));
    EXPECT_EQ(ReadIds(ids_b, 3), ReadIds(ids_a, 3)) << "draw " << draw;
  }
  // A refused update (no sampler for the type) leaves the capacity alone.
  proto::SamplerParameters unspecified;
  unspecified.set_k(100);
  EXPECT_THAT(UpdateOrRecreateMtpVerifierSampler(
                  npu, npu_max_k, Backend::CPU, 1, unspecified, std::cref(env),
                  3, kVocab, ActivationDataType::FLOAT32),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(npu_max_k, 64);
}

TEST(NvidiaMtpDrafterSupportTest, TopKCandidateEligibility) {
  const RankedTensorType fp16(ElementType::Float16,
                              Layout(Dimensions{1, 4, 262144}));
  ASSERT_OK_AND_ASSIGN(
      bool supported,
      SupportsNvidiaTopKCandidates(
          Backend::NPU, TopPParameters(64, 0.95f, 1.0f, 1), kCudaBuffer, fp16));
  EXPECT_TRUE(supported);
  ASSERT_OK_AND_ASSIGN(
      supported, SupportsNvidiaTopKCandidates(Backend::NPU, GreedyParameters(),
                                              kCudaBuffer, fp16));
  EXPECT_TRUE(supported) << "k=1 is a top-k configuration too";
  for (const proto::SamplerParameters& params :
       {TopPParameters(0, 0.95f, 1.0f, 1), TopPParameters(1025, 0.95f, 1.0f, 1),
        TopPParameters(64, 1.5f, 1.0f, 1),
        TopPParameters(64, 0.95f, -1.0f, 1)}) {
    ASSERT_OK_AND_ASSIGN(
        supported,
        SupportsNvidiaTopKCandidates(Backend::NPU, params, kCudaBuffer, fp16));
    EXPECT_FALSE(supported) << params.k() << " " << params.p();
  }
  proto::SamplerParameters top_k = TopPParameters(64, 0.95f, 1.0f, 1);
  top_k.set_type(proto::SamplerParameters::TOP_K);
  ASSERT_OK_AND_ASSIGN(supported, SupportsNvidiaTopKCandidates(
                                      Backend::NPU, top_k, kCudaBuffer, fp16));
  EXPECT_FALSE(supported);
  ASSERT_OK_AND_ASSIGN(
      supported,
      SupportsNvidiaTopKCandidates(
          Backend::GPU, TopPParameters(64, 0.95f, 1.0f, 1), kCudaBuffer, fp16));
  EXPECT_FALSE(supported);
  ASSERT_OK_AND_ASSIGN(supported,
                       SupportsNvidiaTopKCandidates(
                           Backend::NPU, TopPParameters(64, 0.95f, 1.0f, 1),
                           TensorBufferType::kHostMemory, fp16));
  EXPECT_FALSE(supported);
  // A batch above one is the host's, not an error.
  ASSERT_OK_AND_ASSIGN(
      supported,
      SupportsNvidiaTopKCandidates(
          Backend::NPU, TopPParameters(64, 0.95f, 1.0f, 1), kCudaBuffer,
          RankedTensorType(ElementType::Float16,
                           Layout(Dimensions{2, 4, 262144}))));
  EXPECT_FALSE(supported);
  EXPECT_THAT(SupportsNvidiaTopKCandidates(
                  Backend::NPU, TopPParameters(64, 0.95f, 1.0f, 1), kCudaBuffer,
                  RankedTensorType(ElementType::Float16,
                                   Layout(Dimensions{1, 0, 262144}))),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NvidiaMtpDrafterSupportTest, CandidatePolicyRejectsMismatchedInputs) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto ids, CreateTensorBuffer<int32_t>({1, 2}));
  const std::vector<float> uniforms = {0.25f, 0.75f};
  const std::vector<int32_t> candidate_ids = {5, 6, 7, 8};
  const std::vector<float> candidate_values = {3.0f, 2.0f, 1.0f, 0.0f};
  EXPECT_THAT(SampleNvidiaCandidates(candidate_ids, candidate_values, 2, 3,
                                     1.0f, 1.0f, uniforms, ids, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SampleNvidiaCandidates(candidate_ids, candidate_values, 2, 2,
                                     1.0f, 1.0f, {0.5f}, ids, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK(SampleNvidiaCandidates(candidate_ids, candidate_values, 2, 2, 0.0f,
                                   1.0f, uniforms, ids, nullptr));
  EXPECT_EQ(ReadIds(ids, 2), (std::vector<int32_t>{5, 7}))
      << "p = 0 draws each row's first candidate";
}

}  // namespace
}  // namespace litert::lm
