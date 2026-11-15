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

// The NPU route's sampler on real CUDA logits through NVIDIA's dispatch
// library, against the CPU sampler it reuses: the same tokens, scores and
// generator, on every configuration the drafter can be given. Needs a GPU
// and NVIDIA_TEST_DISPATCH_LIBRARY, the path of libLiteRtDispatch_Nvidia.so.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/log/log_sink.h"  // from @com_google_absl
#include "absl/log/log_sink_registry.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"  // from @litert
#include "litert/c/internal/litert_runtime_context.h"  // from @litert
#include "litert/c/internal/litert_tensor_buffer_registry.h"  // from @litert
#include "litert/cc/internal/litert_shared_library.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "litert/vendors/c/litert_dispatch_api.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampling_uniforms.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/nvidia_mtp_drafter_support.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

constexpr auto kCudaBuffer =
    static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType);

// The INFO lines of this process, so a test can see which path sampled.
class LogCapture : public absl::LogSink {
 public:
  LogCapture() { absl::AddLogSink(this); }
  ~LogCapture() override { absl::RemoveLogSink(this); }
  void Send(const absl::LogEntry& entry) override {
    lines_ += std::string(entry.text_message());
    lines_ += '\n';
  }
  bool Has(const std::string& text) const {
    return lines_.find(text) != std::string::npos;
  }
  void Clear() { lines_.clear(); }

 private:
  std::string lines_;
};

struct Candidate {
  int32_t id;
  float value;
};

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

bool CutoffIsUnique(const float* values, int vocab, int k) {
  if (k >= vocab) return true;
  std::vector<float> sorted(values, values + vocab);
  std::partial_sort(sorted.begin(), sorted.begin() + k + 1, sorted.end(),
                    std::greater<float>());
  return sorted[k - 1] != sorted[k];
}

proto::SamplerParameters TopPParameters(int k, float p, float temperature,
                                        int seed, bool exact = false) {
  proto::SamplerParameters params;
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(k);
  params.set_p(p);
  params.set_temperature(temperature);
  params.set_seed(seed);
  params.set_compute_exact_log_probs(exact);
  return params;
}

proto::SamplerParameters GreedyParameters() {
  return TopPParameters(1, 0.0f, 1.0f, 0);
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

class NvidiaMtpCandidatesIntegrationTest : public testing::Test {
 protected:
  void SetUp() override {
    const char* path = std::getenv("NVIDIA_TEST_DISPATCH_LIBRARY");
    if (path == nullptr) {
      GTEST_SKIP() << "NVIDIA_TEST_DISPATCH_LIBRARY names the dispatch DSO";
    }
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    directory_ = std::filesystem::absolute(path).parent_path().string();
    LITERT_ASSERT_OK_AND_ASSIGN(
        library_, SharedLibrary::Load(path, RtldFlags::Now().Local()));
    using QueryInterface = decltype(&LiteRtDispatchQueryInterface);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto query,
        library_->LookupSymbol<QueryInterface>("LiteRtDispatchQueryInterface"));
    LITERT_ASSERT_OK_AND_ASSIGN(
        env_, Environment::Create(
                  EnvironmentOptions(std::vector<EnvironmentOptions::Option>{
                      {EnvironmentOptions::Tag::kDispatchLibraryDir,
                       directory_.c_str()}})));
    LiteRtInterface raw_interface = nullptr;
    ASSERT_EQ(query(kLiteRtInterfaceBasic,
                    {LITERT_DISPATCH_ABI_VERSION_MAJOR,
                     LITERT_DISPATCH_ABI_VERSION_MINOR,
                     LITERT_DISPATCH_ABI_VERSION_PATCH},
                    &raw_interface),
              kLiteRtStatusOk);
    interface_ = static_cast<const LiteRtDispatchInterface_V1*>(raw_interface);
    ASSERT_NE(interface_, nullptr);
    ASSERT_EQ(
        interface_->initialize(LrtGetRuntimeContext(), env_->Get(), nullptr),
        kLiteRtStatusOk);
    LiteRtInterface raw_handlers = nullptr;
    ASSERT_EQ(query(kLiteRtInterfaceCustomTensorBufferHandlers,
                    {LITERT_DISPATCH_ABI_VERSION_MAJOR,
                     LITERT_DISPATCH_ABI_VERSION_MINOR,
                     LITERT_DISPATCH_ABI_VERSION_PATCH},
                    &raw_handlers),
              kLiteRtStatusOk);
    const auto* handlers =
        static_cast<const LiteRtCustomTensorBufferHandlersDef_V1*>(
            raw_handlers);
    ASSERT_NE(handlers, nullptr);
    ASSERT_EQ(
        LiteRtRegisterTensorBufferHandlers(
            env_->Get(), nvidia::kNvidiaCudaTensorBufferType,
            handlers->create_func, handlers->destroy_func, handlers->lock_func,
            handlers->unlock_func, handlers->clear_func, handlers->import_func,
            handlers->device_tag, handlers->queue_tag),
        kLiteRtStatusOk);
    unsetenv("LITERT_NVIDIA_MTP_GPU_SAMPLING");
  }

  void TearDown() override {
    unsetenv("LITERT_NVIDIA_MTP_GPU_SAMPLING");
    if (interface_ != nullptr) {
      EXPECT_EQ(interface_->initialize(nullptr, nullptr, nullptr),
                kLiteRtStatusOk);
    }
  }

  // The same values in a CUDA buffer and a host buffer; the values are
  // rounded to float16 first when the buffers are.
  struct Logits {
    TensorBuffer device;
    TensorBuffer host;
    std::vector<float> values;
    int rows;
    int vocab;
  };

  Logits MakeLogits(std::vector<float> values, int rows, bool fp16) {
    const int vocab = values.size() / rows;
    const RankedTensorType type(
        fp16 ? ElementType::Float16 : ElementType::Float32,
        Layout(Dimensions{1, rows, vocab}));
    const size_t bytes = values.size() * (fp16 ? 2 : 4);
    auto device = TensorBuffer::CreateManaged(*env_, kCudaBuffer, type, bytes);
    auto host = TensorBuffer::CreateManagedHostMemory(type, bytes);
    EXPECT_TRUE(device.HasValue());
    EXPECT_TRUE(host.HasValue());
    if (fp16) {
      std::vector<tflite::half> half(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        half[i] = static_cast<tflite::half>(values[i]);
        values[i] = static_cast<float>(half[i]);
      }
      EXPECT_TRUE(device->Write<tflite::half>(half).HasValue());
      EXPECT_TRUE(host->Write<tflite::half>(half).HasValue());
    } else {
      EXPECT_TRUE(device->Write<float>(values).HasValue());
      EXPECT_TRUE(host->Write<float>(values).HasValue());
    }
    return Logits{std::move(*device), std::move(*host), std::move(values), rows,
                  vocab};
  }

  // The NPU route's sampler as the drafter makes it: greedy at creation, the
  // session's parameters through the verifier's update.
  std::unique_ptr<Sampler> MakeDeviceSampler(
      const Logits& logits, const proto::SamplerParameters& params, bool fp16) {
    const ActivationDataType dtype =
        fp16 ? ActivationDataType::FLOAT16 : ActivationDataType::FLOAT32;
    auto sampler = CreateMtpGreedySampler(Backend::NPU, 1, GreedyParameters(),
                                          std::cref(*env_), logits.rows,
                                          logits.vocab, dtype);
    EXPECT_TRUE(sampler.ok()) << sampler.status();
    if (!sampler.ok()) return nullptr;
    int max_top_k = 1;
    const absl::Status status = UpdateOrRecreateMtpVerifierSampler(
        *sampler, max_top_k, Backend::NPU, 1, params, std::cref(*env_),
        logits.rows, logits.vocab, dtype);
    EXPECT_TRUE(status.ok()) << status;
    return std::move(*sampler);
  }

  std::unique_ptr<TopPSampler> MakeCpuSampler(
      const Logits& logits, const proto::SamplerParameters& params) {
    auto sampler =
        TopPSampler::Create(params.k(), params.p(), params.temperature(), 1,
                            logits.rows, params.compute_exact_log_probs());
    EXPECT_TRUE(sampler.ok()) << sampler.status();
    if (!sampler.ok()) return nullptr;
    const absl::Status status = (*sampler)->UpdateConfig(params, 1);
    EXPECT_TRUE(status.ok()) << status;
    return std::move(*sampler);
  }

  struct Draw {
    std::vector<int32_t> ids;
    std::vector<float> scores;
  };

  // The uniforms of a sampling call, drawn from `generator` as the executor
  // draws them (DrawSamplingUniforms).
  std::vector<float> Uniforms(const proto::SamplerParameters& params, int rows,
                              std::default_random_engine& generator) {
    std::vector<float> uniforms(rows);
    DrawSamplingUniforms(params, generator, absl::MakeSpan(uniforms));
    return uniforms;
  }

  Draw Sample(Sampler& sampler, const TensorBuffer& logits, int rows,
              bool with_scores, absl::Span<const float> uniforms) {
    auto ids = CreateTensorBuffer<int32_t>({1, rows});
    auto scores = CreateTensorBuffer<float>({1, rows});
    EXPECT_TRUE(ids.HasValue());
    EXPECT_TRUE(scores.HasValue());
    const absl::Status status = sampler.SampleToIdAndScoreBuffer(
        logits, *ids, with_scores ? &*scores : nullptr, uniforms);
    EXPECT_TRUE(status.ok()) << status;
    Draw draw;
    draw.ids = ReadIds(*ids, rows);
    if (with_scores) draw.scores = ReadScores(*scores, rows);
    return draw;
  }

  // The policy on the host's own ordered candidates of the same values: what
  // the device path must equal whatever the ties.
  Draw ReferenceDraw(const Logits& logits,
                     const proto::SamplerParameters& params,
                     absl::Span<const float> uniforms, bool with_scores) {
    const int k = std::min(params.k(), logits.vocab);
    std::vector<int32_t> candidate_ids;
    std::vector<float> candidate_values;
    for (int row = 0; row < logits.rows; ++row) {
      for (const Candidate& candidate : OrderedCandidates(
               logits.values.data() + row * logits.vocab, logits.vocab, k)) {
        candidate_ids.push_back(candidate.id);
        candidate_values.push_back(candidate.value);
      }
    }
    auto ids = CreateTensorBuffer<int32_t>({1, logits.rows});
    auto scores = CreateTensorBuffer<float>({1, logits.rows});
    EXPECT_TRUE(ids.HasValue());
    EXPECT_TRUE(scores.HasValue());
    const absl::Status status = SampleNvidiaCandidates(
        candidate_ids, candidate_values, logits.rows, k, params.p(),
        params.temperature(), uniforms, *ids, with_scores ? &*scores : nullptr);
    EXPECT_TRUE(status.ok()) << status;
    Draw draw;
    draw.ids = ReadIds(*ids, logits.rows);
    if (with_scores) draw.scores = ReadScores(*scores, logits.rows);
    return draw;
  }

  std::vector<float> RandomValues(size_t n, float scale = 4.0f) {
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> values(n);
    for (float& value : values) value = dist(gen_);
    return values;
  }

  // Makes every row's k-th largest value unique, and representable in
  // float16: k + 1 distinct integers above the rest at random positions.
  void ForceUniqueCutoffs(std::vector<float>& values, int rows, int k) {
    const int vocab = values.size() / rows;
    std::uniform_int_distribution<int> position(0, vocab - 1);
    for (int row = 0; row < rows; ++row) {
      std::set<int> taken;
      while (static_cast<int>(taken.size()) < k + 1)
        taken.insert(position(gen_));
      float value = 50.0f + static_cast<float>(taken.size());
      for (int index : taken) {
        values[static_cast<size_t>(row) * vocab + index] = value;
        value -= 1.0f;
      }
    }
  }

  std::string directory_;
  std::optional<SharedLibrary> library_;
  std::optional<Environment> env_;
  const LiteRtDispatchInterface_V1* interface_ = nullptr;
  std::mt19937 gen_{20261027};
};

constexpr int kVocab = 262144;

TEST_F(NvidiaMtpCandidatesIntegrationTest,
       MatchesTheCpuSamplerOnUniqueCutoffs) {
  LogCapture log;
  std::map<std::pair<bool, int>, int> configurations;
  for (bool fp16 : {false, true}) {
    for (int rows : {1, 4}) {
      for (int k : {1, 7, 64, 1024}) {
        for (float p : {0.0f, 0.3f, 0.95f, 1.0f}) {
          for (float temperature : {0.0f, 0.7f, 1.0f}) {
            SCOPED_TRACE(absl::StrCat("fp16 ", fp16, " rows ", rows, " k ", k,
                                      " p ", p, " T ", temperature));
            // Float16 values tie: the cutoffs are made unique by construction.
            std::vector<float> values = RandomValues(rows * kVocab);
            ForceUniqueCutoffs(values, rows, k);
            Logits logits = MakeLogits(std::move(values), rows, fp16);
            ASSERT_FALSE(HasFatalFailure());
            for (int row = 0; row < rows; ++row) {
              ASSERT_TRUE(CutoffIsUnique(logits.values.data() + row * kVocab,
                                         kVocab, k));
            }
            const proto::SamplerParameters params =
                TopPParameters(k, p, temperature, 7);
            std::default_random_engine generator(7);
            auto device = MakeDeviceSampler(logits, params, fp16);
            auto cpu = MakeCpuSampler(logits, params);
            ASSERT_NE(device, nullptr);
            ASSERT_NE(cpu, nullptr);
            for (int draw = 0; draw < 3; ++draw) {
              const std::vector<float> uniforms =
                  Uniforms(params, rows, generator);
              const Draw got =
                  Sample(*device, logits.device, rows, true, uniforms);
              const Draw want = Sample(*cpu, logits.host, rows, true, uniforms);
              EXPECT_EQ(got.ids, want.ids) << "draw " << draw;
              EXPECT_EQ(got.scores, want.scores) << "draw " << draw;
            }
            ++configurations[std::make_pair(fp16, k)];
          }
        }
      }
    }
  }
  for (bool fp16 : {false, true}) {
    for (int k : {1, 7, 64, 1024}) {
      const int count = configurations[std::make_pair(fp16, k)];
      EXPECT_EQ(count, 24) << "fp16 " << fp16 << " k " << k;
    }
  }
  EXPECT_TRUE(log.Has("candidate selection enabled")) << "the device sampled";
}

TEST_F(NvidiaMtpCandidatesIntegrationTest,
       Fp16TiesFollowTheHostPolicyOnOrderedCandidates) {
  for (int rows : {1, 4}) {
    for (int k : {7, 64}) {
      SCOPED_TRACE(absl::StrCat("rows ", rows, " k ", k));
      Logits logits =
          MakeLogits(RandomValues(rows * kVocab), rows, /*fp16=*/true);
      ASSERT_FALSE(HasFatalFailure());
      const proto::SamplerParameters params = TopPParameters(k, 0.95f, 1.0f, 3);
      std::default_random_engine generator(3);
      auto device = MakeDeviceSampler(logits, params, true);
      auto cpu = MakeCpuSampler(logits, params);
      ASSERT_NE(device, nullptr);
      ASSERT_NE(cpu, nullptr);
      for (int draw = 0; draw < 4; ++draw) {
        const std::vector<float> uniforms = Uniforms(params, rows, generator);
        const Draw got = Sample(*device, logits.device, rows, true, uniforms);
        const Draw reference = ReferenceDraw(logits, params, uniforms, true);
        const Draw want = Sample(*cpu, logits.host, rows, true, uniforms);
        EXPECT_EQ(got.ids, reference.ids) << "draw " << draw;
        EXPECT_EQ(got.scores, reference.scores) << "draw " << draw;
        // The CPU's heap and the device agree wherever the cutoff is unique.
        for (int row = 0; row < rows; ++row) {
          if (CutoffIsUnique(logits.values.data() + row * kVocab, kVocab, k)) {
            EXPECT_EQ(got.ids[row], want.ids[row]) << "row " << row;
          }
        }
      }
    }
  }
}

TEST_F(NvidiaMtpCandidatesIntegrationTest,
       HostWrittenMasksReachTheDeviceAndTheDrawsAgree) {
  constexpr int kRows = 4;
  for (bool fp16 : {false, true}) {
    SCOPED_TRACE(absl::StrCat("fp16 ", fp16));
    // The unmasked row compares with the CPU sampler: its cutoff is unique.
    std::vector<float> values = RandomValues(kRows * kVocab);
    ForceUniqueCutoffs(values, kRows, 64);
    Logits logits = MakeLogits(std::move(values), kRows, fp16);
    ASSERT_FALSE(HasFatalFailure());
    const proto::SamplerParameters params = TopPParameters(64, 0.95f, 1.0f, 5);
    std::default_random_engine generator(5);
    auto device = MakeDeviceSampler(logits, params, fp16);
    auto cpu = MakeCpuSampler(logits, params);
    ASSERT_NE(device, nullptr);
    ASSERT_NE(cpu, nullptr);
    // A draw before any mask, so the device has the logits already.
    {
      const std::vector<float> uniforms = Uniforms(params, kRows, generator);
      EXPECT_EQ(Sample(*device, logits.device, kRows, false, uniforms).ids,
                Sample(*cpu, logits.host, kRows, false, uniforms).ids);
    }
    // Then the masks a constraint and a penalty leave, through the host lock:
    // the runner's host path for CUDA buffers. A mask is destructive, so each
    // round starts from the logits again, written through the lock as well.
    ASSERT_OK_AND_ASSIGN(auto runner, LiteRtLogitMaskRunner::CreateForHost());
    for (int round = 0; round < 3; ++round) {
      if (fp16) {
        std::vector<tflite::half> half(logits.values.size());
        for (size_t i = 0; i < half.size(); ++i) {
          half[i] = static_cast<tflite::half>(logits.values[i]);
        }
        LITERT_ASSERT_OK(logits.device.Write<tflite::half>(half));
        LITERT_ASSERT_OK(logits.host.Write<tflite::half>(half));
      } else {
        LITERT_ASSERT_OK(logits.device.Write<float>(logits.values));
        LITERT_ASSERT_OK(logits.host.Write<float>(logits.values));
      }
      std::vector<std::unique_ptr<LogitMask>> masks;
      const std::vector<int> allowed = {3 + round, 100,    2000,   5000 + round,
                                        70000,     123456, 200000, 262143};
      masks.push_back(
          BitmapLogitMask::CreateFromAllowedTokens(kVocab, allowed));
      masks.push_back(
          BitmapLogitMask::CreateSingleAllowedToken(kVocab, 77 + round));
      masks.push_back(std::make_unique<SparseLogitMask>(
          std::vector<SparseLogitMask::Entry>{{10, 1.0f, 30.0f, false},
                                              {11, 0.5f, 0.0f, false},
                                              {12, 1.0f, -100.0f, false}}));
      masks.push_back(nullptr);
      ASSERT_OK(runner->ApplySequence(logits.device, masks));
      ASSERT_OK(runner->ApplySequence(logits.host, masks));
      for (int draw = 0; draw < 2; ++draw) {
        const std::vector<float> uniforms = Uniforms(params, kRows, generator);
        const Draw got = Sample(*device, logits.device, kRows, true, uniforms);
        const Draw want = Sample(*cpu, logits.host, kRows, true, uniforms);
        EXPECT_EQ(got.ids, want.ids) << "round " << round << " draw " << draw;
        EXPECT_EQ(got.scores, want.scores);
        EXPECT_EQ(got.ids[1], 77 + round);
        EXPECT_NE(std::find(allowed.begin(), allowed.end(), got.ids[0]),
                  allowed.end());
      }
    }
  }
}

TEST_F(NvidiaMtpCandidatesIntegrationTest, FallsBackToTheHostWhereItMust) {
  constexpr int kRows = 2;
  LogCapture log;
  // A NaN: the row's candidates are unspecified, the host samples.
  {
    std::vector<float> values = RandomValues(kRows * kVocab);
    values[kVocab + 1234] = std::numeric_limits<float>::quiet_NaN();
    Logits logits = MakeLogits(values, kRows, /*fp16=*/false);
    ASSERT_FALSE(HasFatalFailure());
    const proto::SamplerParameters params = TopPParameters(64, 0.95f, 1.0f, 9);
    std::default_random_engine generator(9);
    auto device = MakeDeviceSampler(logits, params, false);
    auto cpu = MakeCpuSampler(logits, params);
    ASSERT_NE(device, nullptr);
    ASSERT_NE(cpu, nullptr);
    for (int draw = 0; draw < 2; ++draw) {
      const std::vector<float> uniforms = Uniforms(params, kRows, generator);
      EXPECT_EQ(Sample(*device, logits.device, kRows, false, uniforms).ids,
                Sample(*cpu, logits.host, kRows, false, uniforms).ids);
    }
    EXPECT_TRUE(log.Has("candidate selection enabled"));
  }
  // Exact log probabilities need the whole vocabulary: the host's scores.
  {
    Logits logits = MakeLogits(RandomValues(kRows * kVocab), kRows, false);
    ASSERT_FALSE(HasFatalFailure());
    const proto::SamplerParameters params =
        TopPParameters(64, 0.95f, 1.0f, 9, /*exact=*/true);
    std::default_random_engine generator(9);
    auto device = MakeDeviceSampler(logits, params, false);
    auto cpu = MakeCpuSampler(logits, params);
    ASSERT_NE(device, nullptr);
    ASSERT_NE(cpu, nullptr);
    std::vector<float> uniforms = Uniforms(params, kRows, generator);
    const Draw got = Sample(*device, logits.device, kRows, true, uniforms);
    const Draw want = Sample(*cpu, logits.host, kRows, true, uniforms);
    EXPECT_EQ(got.ids, want.ids);
    EXPECT_EQ(got.scores, want.scores);
    // Without scores the device path runs, with the same uniforms.
    uniforms = Uniforms(params, kRows, generator);
    EXPECT_EQ(Sample(*device, logits.device, kRows, false, uniforms).ids,
              Sample(*cpu, logits.host, kRows, false, uniforms).ids);
  }
  // k beyond the device's reach: the host, from the start.
  {
    Logits logits = MakeLogits(RandomValues(kRows * kVocab), kRows, false);
    ASSERT_FALSE(HasFatalFailure());
    log.Clear();
    const proto::SamplerParameters params =
        TopPParameters(2048, 0.95f, 1.0f, 9);
    std::default_random_engine generator(9);
    auto device = MakeDeviceSampler(logits, params, false);
    auto cpu = MakeCpuSampler(logits, params);
    ASSERT_NE(device, nullptr);
    ASSERT_NE(cpu, nullptr);
    const std::vector<float> uniforms = Uniforms(params, kRows, generator);
    EXPECT_EQ(Sample(*device, logits.device, kRows, true, uniforms).ids,
              Sample(*cpu, logits.host, kRows, true, uniforms).ids);
    EXPECT_FALSE(log.Has("candidate selection enabled"));
  }
  // Opted out: the host, from the start.
  {
    Logits logits = MakeLogits(RandomValues(kRows * kVocab), kRows, true);
    ASSERT_FALSE(HasFatalFailure());
    log.Clear();
    setenv("LITERT_NVIDIA_MTP_GPU_SAMPLING", "0", 1);
    const proto::SamplerParameters params = TopPParameters(64, 0.95f, 1.0f, 9);
    std::default_random_engine generator(9);
    auto device = MakeDeviceSampler(logits, params, true);
    auto cpu = MakeCpuSampler(logits, params);
    ASSERT_NE(device, nullptr);
    ASSERT_NE(cpu, nullptr);
    const std::vector<float> uniforms = Uniforms(params, kRows, generator);
    EXPECT_EQ(Sample(*device, logits.device, kRows, true, uniforms).ids,
              Sample(*cpu, logits.host, kRows, true, uniforms).ids);
    EXPECT_FALSE(log.Has("candidate selection enabled"));
    EXPECT_FALSE(log.Has("greedy sampling enabled"));
    unsetenv("LITERT_NVIDIA_MTP_GPU_SAMPLING");
  }
}

TEST_F(NvidiaMtpCandidatesIntegrationTest, GreedyIsTheFirstMaximumOnTies) {
  constexpr int kRows = 4;
  LogCapture log;
  std::vector<float> values = RandomValues(kRows * kVocab, 2.0f);
  // Float16 rounding ties the maxima often enough; force a few.
  values[5] = 50.0f;
  values[kVocab - 1] = 50.0f;
  values[kVocab + 9] = 50.0f;
  values[kVocab + 8] = 50.0f;
  Logits logits = MakeLogits(values, kRows, /*fp16=*/true);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_OK_AND_ASSIGN(
      auto device, CreateMtpGreedySampler(Backend::NPU, 1, GreedyParameters(),
                                          std::cref(*env_), kRows, kVocab,
                                          ActivationDataType::FLOAT16));
  auto cpu = MakeCpuSampler(logits, GreedyParameters());
  ASSERT_NE(cpu, nullptr);
  const std::vector<float> uniforms(kRows, 0.0f);
  const Draw got = Sample(*device, logits.device, kRows, false, uniforms);
  const Draw want = Sample(*cpu, logits.host, kRows, false, uniforms);
  EXPECT_EQ(got.ids, want.ids);
  EXPECT_EQ(got.ids[0], 5);
  EXPECT_EQ(got.ids[1], 8);
  EXPECT_TRUE(log.Has("greedy sampling enabled"));
  // With a scores tensor the argmax has nothing to give: the host scores.
  const Draw scored = Sample(*device, logits.device, kRows, true, uniforms);
  const Draw scored_want = Sample(*cpu, logits.host, kRows, true, uniforms);
  EXPECT_EQ(scored.ids, scored_want.ids);
  EXPECT_EQ(scored.scores, scored_want.scores);
  EXPECT_EQ(scored.scores, std::vector<float>(kRows, 0.0f));
}

}  // namespace
}  // namespace litert::lm
