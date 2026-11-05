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

#include "runtime/components/sampler_factory.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <atomic>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <string>
#include <system_error>  // NOLINT

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#endif  // defined(__linux__) && !defined(__ANDROID__)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

TEST(SamplerFactoryTest, CreateSamplerForCpuWorksCorrectly) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  // Make sure the factory creates the correct sampler.
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
  EXPECT_FALSE(top_p_sampler->ComputeExactLogProbs());
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithComputeExactLogProbs) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_compute_exact_log_probs(true);
  ASSERT_OK_AND_ASSIGN(
      auto sampler, CreateSampler(Backend::CPU,
                                  /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  ASSERT_NE(top_p_sampler, nullptr);
  EXPECT_TRUE(top_p_sampler->ComputeExactLogProbs());
}

TEST(SamplerFactoryTest, CreateSamplerForCpuWithUnsupportedSamplerTypeFails) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_K);
  auto result = CreateSampler(Backend::CPU,
                              /*batch_size=*/1, std::move(sampler_params));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(result.status().message(), HasSubstr("not implemented yet."));
}

TEST(SamplerFactoryTest,
     CreateSamplerForCpuWithUnspecifiedSamplerTypeReturnsNullptr) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  ASSERT_OK_AND_ASSIGN(
      auto sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, std::move(sampler_params)));
  EXPECT_EQ(sampler, nullptr);
}

proto::SamplerParameters TopPParams(int k, float p, float temperature) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(k);
  sampler_params.set_p(p);
  sampler_params.set_temperature(temperature);
  sampler_params.set_seed(0);
  return sampler_params;
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerCreatesMissingSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                                    /*rand_gen=*/nullptr));
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  EXPECT_EQ(max_top_k, 1);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerRecreatesOnlyWhenTopKGrows) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f)));
  int max_top_k = 1;
  auto rand_gen = std::make_shared<std::default_random_engine>(42);

  // A larger top-k recreates the sampler.
  const Sampler* original = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(40, 0.95f, 1.0f), rand_gen));
  EXPECT_NE(sampler.get(), original);
  EXPECT_EQ(max_top_k, 40);

  // A smaller or equal top-k reconfigures the sampler in place.
  const Sampler* recreated = sampler.get();
  ASSERT_OK(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                    /*batch_size=*/1,
                                    TopPParams(10, 0.9f, 0.7f), rand_gen));
  EXPECT_EQ(sampler.get(), recreated);
  EXPECT_EQ(max_top_k, 40);
}

absl::StatusOr<int> SampleOne(Sampler& sampler,
                              const std::vector<float>& logits) {
  auto logits_tensor =
      CopyToTensorBuffer<float>(logits, {1, static_cast<int>(logits.size())});
  if (!logits_tensor) {
    return absl::InternalError("Failed to create the logits tensor.");
  }
  std::vector<int> ids_vector(1);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1});
  if (!ids_tensor) {
    return absl::InternalError("Failed to create the ids tensor.");
  }
  ABSL_RETURN_IF_ERROR(sampler.SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, /*scores_tensor=*/nullptr));
  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  if (!ids) {
    return absl::InternalError("Failed to read the ids tensor.");
  }
  return (*ids)[0];
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerSharesRandomGenerator) {
  // Samplers `a` and `b` share one generator, so alternating between them
  // yields the same draws as a single sampler `c` with its own generator
  // seeded identically.
  const auto params = TopPParams(8, 1.0f, 1.0f);
  auto shared = std::make_shared<std::default_random_engine>(7);
  auto reference = std::make_shared<std::default_random_engine>(7);
  std::unique_ptr<Sampler> a;
  std::unique_ptr<Sampler> b;
  std::unique_ptr<Sampler> c;
  int max_top_k_a = 0;
  int max_top_k_b = 0;
  int max_top_k_c = 0;
  ASSERT_OK(UpdateOrRecreateSampler(a, max_top_k_a, Backend::CPU,
                                    /*batch_size=*/1, params, shared));
  ASSERT_OK(UpdateOrRecreateSampler(b, max_top_k_b, Backend::CPU,
                                    /*batch_size=*/1, params, shared));
  ASSERT_OK(UpdateOrRecreateSampler(c, max_top_k_c, Backend::CPU,
                                    /*batch_size=*/1, params, reference));

  const std::vector<float> logits(8, 0.0f);  // Uniform over 8 tokens.
  std::vector<int> alternating;
  std::vector<int> single;
  for (int i = 0; i < 64; ++i) {
    ASSERT_OK_AND_ASSIGN(int id, SampleOne(i % 2 == 0 ? *a : *b, logits));
    alternating.push_back(id);
    ASSERT_OK_AND_ASSIGN(int ref_id, SampleOne(*c, logits));
    single.push_back(ref_id);
  }
  EXPECT_EQ(alternating, single);
}

TEST(SamplerFactoryTest, UpdateOrRecreateSamplerFailsWithoutSampler) {
  std::unique_ptr<Sampler> sampler;
  int max_top_k = 0;
  proto::SamplerParameters unspecified;
  unspecified.set_k(1);
  EXPECT_THAT(UpdateOrRecreateSampler(sampler, max_top_k, Backend::CPU,
                                      /*batch_size=*/1, unspecified,
                                      /*rand_gen=*/nullptr),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(max_top_k, 0);
}

TEST(SamplerFactoryTest, GreedySamplerParamsAreGreedyTopP) {
  const proto::SamplerParameters params = GreedySamplerParams();
  EXPECT_EQ(params.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(params.k(), 1);
  EXPECT_EQ(params.temperature(), 1.0f);
  EXPECT_EQ(params.seed(), 0);
}

TEST(SamplerFactoryTest, SamplerParamsEquivalentTreatsUnspecifiedAsDefault) {
  // Unspecified parameters select the default sampler like unset ones, whatever
  // the other fields hold.
  proto::SamplerParameters unspecified;
  unspecified.set_k(40);
  unspecified.set_temperature(0.7f);
  EXPECT_TRUE(SamplerParamsEquivalent(std::nullopt, std::nullopt));
  EXPECT_TRUE(SamplerParamsEquivalent(std::nullopt, unspecified));
  EXPECT_TRUE(SamplerParamsEquivalent(unspecified, std::nullopt));
  EXPECT_TRUE(SamplerParamsEquivalent(unspecified, proto::SamplerParameters()));

  EXPECT_FALSE(SamplerParamsEquivalent(std::nullopt, GreedySamplerParams()));
  EXPECT_FALSE(SamplerParamsEquivalent(GreedySamplerParams(), unspecified));
}

TEST(SamplerFactoryTest, SamplerParamsEquivalentComparesEveryField) {
  proto::SamplerParameters base;
  base.set_type(proto::SamplerParameters::TOP_P);
  base.set_k(40);
  base.set_p(0.95f);
  base.set_temperature(1.0f);
  base.set_seed(7);
  EXPECT_TRUE(SamplerParamsEquivalent(base, base));

  proto::SamplerParameters unset_seed = base;
  unset_seed.clear_seed();
  proto::SamplerParameters zero_seed = base;
  zero_seed.set_seed(0);
  EXPECT_TRUE(SamplerParamsEquivalent(unset_seed, zero_seed));

  proto::SamplerParameters changed = base;
  changed.set_type(proto::SamplerParameters::TOP_K);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_k(10);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_p(0.5f);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_temperature(0.7f);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_seed(8);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_backend(proto::SamplerParameters::CPU);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
  changed = base;
  changed.set_compute_exact_log_probs(true);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base));
}

TEST(SamplerFactoryTest, SamplerParamsEquivalentCanIgnoreSeed) {
  proto::SamplerParameters base;
  base.set_type(proto::SamplerParameters::TOP_P);
  base.set_k(40);
  base.set_p(0.95f);
  base.set_temperature(1.0f);
  base.set_seed(7);

  proto::SamplerParameters changed = base;
  changed.set_seed(8);
  EXPECT_TRUE(SamplerParamsEquivalent(changed, base, /*ignore_seed=*/true));
  changed.clear_seed();
  EXPECT_TRUE(SamplerParamsEquivalent(changed, base, /*ignore_seed=*/true));

  // The other fields are still compared.
  changed.set_k(10);
  EXPECT_FALSE(SamplerParamsEquivalent(changed, base, /*ignore_seed=*/true));
}

#if defined(__linux__) && !defined(__ANDROID__)
#define SO_EXT ".so"

using ::testing::status::IsOkAndHolds;

class DynamicSamplerFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<int> counter{0};
    test_dir_ = std::filesystem::path(::testing::TempDir()) /
                absl::StrCat("dynamic_sampler_factory_test_",
                             ::testing::UnitTest::GetInstance()->random_seed(),
                             "_", counter++);
    std::error_code ec;
    std::filesystem::create_directories(test_dir_, ec);

    webgpu_fake_runfile_ = FindRunfile("fake_top_k_webgpu_sampler.so");
    opencl_fake_runfile_ = FindRunfile("fake_top_k_opencl_sampler.so");
  }

  void TearDown() override {
    if (!test_dir_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(test_dir_, ec);
    }
  }

  const std::filesystem::path& test_dir() const { return test_dir_; }

  static std::filesystem::path FindRunfile(absl::string_view filename) {
    constexpr absl::string_view kTestComponentsDir =
        "litert_lm/runtime/components/";
    std::filesystem::path internal_runfile_path =
        std::filesystem::path(::testing::SrcDir()) / kTestComponentsDir /
        filename;
    if (std::filesystem::exists(internal_runfile_path)) {
      return internal_runfile_path;
    }

    std::filesystem::path workspace_relative_path =
        std::filesystem::path(::testing::SrcDir()) / "runtime/components" /
        filename;
    if (std::filesystem::exists(workspace_relative_path)) {
      return workspace_relative_path;
    }

    std::filesystem::path oss_runfile_path =
        std::filesystem::path(::testing::SrcDir()) /
        "litert_lm/runtime/components" / filename;
    if (std::filesystem::exists(oss_runfile_path)) {
      return oss_runfile_path;
    }

    return "";
  }

  absl::Status InstallFake(const std::filesystem::path& fake_runfile,
                           absl::string_view backend_label,
                           absl::string_view target_filename) {
    if (test_dir_.empty()) {
      return absl::InternalError("test_dir_ was not initialized");
    }
    std::error_code ec;
    if (fake_runfile.empty() || !std::filesystem::exists(fake_runfile, ec) ||
        ec) {
      return absl::NotFoundError(absl::StrCat(
          backend_label, " fake runfile not found: ", fake_runfile.string()));
    }
    std::filesystem::copy_file(
        fake_runfile, test_dir() / std::string(target_filename),
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      return absl::InternalError(ec.message());
    }
    return absl::OkStatus();
  }

  absl::Status InstallWebGpuFake() {
    return InstallFake(webgpu_fake_runfile_, "WebGPU",
                       "libLiteRtTopKWebGpuSampler" SO_EXT);
  }

  absl::Status InstallOpenClFake() {
    return InstallFake(opencl_fake_runfile_, "OpenCL",
                       "libLiteRtTopKOpenClSampler" SO_EXT);
  }

  std::filesystem::path test_dir_;
  std::filesystem::path webgpu_fake_runfile_;
  std::filesystem::path opencl_fake_runfile_;
};

TEST_F(DynamicSamplerFactoryTest,
       LoadsWebGpuSamplerFromRuntimeLibraryDirAndSamples) {
  ASSERT_OK(InstallWebGpuFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(litert::EnvironmentOptions({opt})));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  EXPECT_TRUE(sampler->CanHandleInput());
  EXPECT_TRUE(sampler->HandlesInput());
  EXPECT_OK(sampler->UpdateConfig(TopPParams(10, 0.9f, 0.7f), /*batch_size=*/1,
                                  /*rand_gen=*/nullptr));

  // Test basic SampleOne (token ID 42 from fake WebGPU sampler).
  const std::vector<float> logits(201, 0.0f);
  EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(42));

  // Test SampleToIdAndScoreBuffer with uniforms.
  auto logits_tensor =
      CopyToTensorBuffer<float>(logits, {1, static_cast<int>(logits.size())});
  ASSERT_TRUE(logits_tensor.HasValue());
  std::vector<int> ids_vector(1);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {1});
  ASSERT_TRUE(ids_tensor.HasValue());
  const float uniforms[] = {0.5f};
  EXPECT_OK(sampler->SampleToIdAndScoreBuffer(
      *logits_tensor, *ids_tensor, /*scores_tensor=*/nullptr, uniforms));
  litert::Expected<std::vector<int>> ids =
      CopyFromTensorBuffer<int>(*ids_tensor);
  ASSERT_TRUE(ids.HasValue());
  EXPECT_EQ((*ids)[0], 42);
}

TEST_F(DynamicSamplerFactoryTest,
       SelectsBackendBasedOnEnvironmentDeviceOptions) {
  ASSERT_OK(InstallWebGpuFake());
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());

  const std::vector<float> logits(201, 0.0f);

  // 1. Explicit kWebGpuDevice selects WebGPU (id 42).
  {
    litert::EnvironmentOptions::Option dev_opt(
        litert::EnvironmentOptions::Tag::kWebGpuDevice,
        static_cast<int64_t>(1));
    LITERT_ASSERT_OK_AND_ASSIGN(
        litert::Environment env,
        litert::Environment::Create(
            litert::EnvironmentOptions({dir_opt, dev_opt})));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(42));
  }

  // 2. Explicit kOpenClDeviceId selects OpenCL (id 99).
#if !defined(LITERT_DISABLE_OPENCL_SUPPORT)
  {
    litert::EnvironmentOptions::Option dev_opt(
        litert::EnvironmentOptions::Tag::kOpenClDeviceId,
        static_cast<int64_t>(1));
    LITERT_ASSERT_OK_AND_ASSIGN(
        litert::Environment env,
        litert::Environment::Create(
            litert::EnvironmentOptions({dir_opt, dev_opt})));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(99));
  }
#endif

  // 3. Blank device option prefers WebGPU (id 42) by default.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        litert::Environment env,
        litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1, /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(42));
  }
}

TEST_F(
    DynamicSamplerFactoryTest,
    FallsBackToCpuWhenRequestedBackendIsUnavailableEvenIfOtherBackendPresent) {
  // Only install OpenCL fake; WebGPU is absent.
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option dev_opt(
      litert::EnvironmentOptions::Tag::kWebGpuDevice, static_cast<int64_t>(1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(
          litert::EnvironmentOptions({dir_opt, dev_opt})));

  // Requested WebGPU explicitly, which is absent. Should fall back to CPU
  // rather than using mismatched OpenCL!
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
}

TEST_F(DynamicSamplerFactoryTest, LoadsWebGpuSamplerUsingStringViewOption) {
  ASSERT_OK(InstallWebGpuFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
      litert::StringView(dir_str.data(), dir_str.size()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(litert::EnvironmentOptions({opt})));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  const std::vector<float> logits(201, 0.0f);
  EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(42));
}

TEST_F(DynamicSamplerFactoryTest,
       FallsBackToCpuWhenOpenClRequestedAndUnavailable) {
  ASSERT_FALSE(test_dir_.empty());
  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option dev_opt(
      litert::EnvironmentOptions::Tag::kOpenClDeviceId,
      static_cast<int64_t>(1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(
          litert::EnvironmentOptions({dir_opt, dev_opt})));

  // Requested OpenCL explicitly, which is absent in test_dir_. Should fall back
  // to CPU.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
}

TEST_F(DynamicSamplerFactoryTest,
       FallsThroughToOpenClWhenWebGpuReturnsNotFoundAndNoExplicitBackend) {
  ASSERT_OK(InstallWebGpuFake());
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));

  // vocab_size = 404 triggers NotFoundError in fake WebGPU sampler.
  // With no explicit backend requested, it falls through to OpenCL (id 99).
#if !defined(LITERT_DISABLE_OPENCL_SUPPORT)
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/404,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  const std::vector<float> logits(404, 0.0f);
  EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(99));
#endif
}

TEST_F(DynamicSamplerFactoryTest,
       PropagatesNotFoundErrorWhenWebGpuExplicitlyRequested) {
  ASSERT_OK(InstallWebGpuFake());
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option dev_opt(
      litert::EnvironmentOptions::Tag::kWebGpuDevice, static_cast<int64_t>(1));

  litert::EnvironmentOptions env_options({dir_opt, dev_opt});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::Environment env,
                              litert::Environment::Create(env_options));

  // vocab_size = 404 triggers NotFound in fake WebGPU sampler.
  // Since WebGPU was explicitly requested, NotFound must be propagated.
  EXPECT_THAT(
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/404,
                    /*activation_data_type=*/std::nullopt),
      StatusIs(absl::StatusCode::kNotFound));
}

#if !defined(LITERT_DISABLE_OPENCL_SUPPORT)
TEST_F(DynamicSamplerFactoryTest,
       PropagatesNotFoundErrorWhenOpenClExplicitlyRequested) {
  ASSERT_OK(InstallWebGpuFake());
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option dev_opt(
      litert::EnvironmentOptions::Tag::kOpenClDeviceId,
      static_cast<int64_t>(1));

  litert::EnvironmentOptions env_options({dir_opt, dev_opt});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::Environment env,
                              litert::Environment::Create(env_options));

  // vocab_size = 405 triggers NotFound in fake OpenCL sampler.
  // Since OpenCL was explicitly requested, NotFound must be propagated.
  EXPECT_THAT(
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/405,
                    /*activation_data_type=*/std::nullopt),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DynamicSamplerFactoryTest,
       FallsBackToCpuWhenOpenClReturnsNotFoundAndNoExplicitBackend) {
  // Only install OpenCL fake; WebGPU is absent.
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));

  // vocab_size = 405 triggers NotFound in fake OpenCL sampler.
  // With no explicit backend requested and WebGPU absent, NotFound from
  // OpenCL falls back cleanly to CPU sampling.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/405,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
}

TEST_F(DynamicSamplerFactoryTest,
       PropagatesOpenClNonFallbackErrorWithoutFallingThrough) {
  // Only install OpenCL fake; WebGPU is absent.
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));

  // vocab_size = 401 triggers InvalidArgument in fake OpenCL sampler.
  // Even with no explicit backend requested, a non-fallback error (like
  // InvalidArgument) must be returned immediately rather than falling back to
  // CPU.
  EXPECT_THAT(
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/401,
                    /*activation_data_type=*/std::nullopt),
      StatusIs(absl::StatusCode::kInvalidArgument));
}
#endif  // !defined(LITERT_DISABLE_OPENCL_SUPPORT)

TEST_F(DynamicSamplerFactoryTest,
       SelectsWebGpuWhenWebGpuAndOpenClCoPopulatedInOptions) {
  ASSERT_OK(InstallWebGpuFake());
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option webgpu_opt(
      litert::EnvironmentOptions::Tag::kWebGpuDevice, static_cast<int64_t>(1));
  litert::EnvironmentOptions::Option opencl_opt(
      litert::EnvironmentOptions::Tag::kOpenClDeviceId,
      static_cast<int64_t>(1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(
          litert::EnvironmentOptions({dir_opt, webgpu_opt, opencl_opt})));

  // When both WebGPU and OpenCL device options are populated (e.g. default
  // devices created by GpuEnvironment), WebGPU is selected.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  const std::vector<float> logits(201, 0.0f);
  EXPECT_THAT(SampleOne(*sampler, logits), IsOkAndHolds(42));
}

TEST_F(DynamicSamplerFactoryTest,
       FallsBackToCpuWhenWebGpuAndOpenClCoPopulatedAndWebGpuPluginAbsent) {
  // Only install OpenCL fake; WebGPU is absent.
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option webgpu_opt(
      litert::EnvironmentOptions::Tag::kWebGpuDevice, static_cast<int64_t>(1));
  litert::EnvironmentOptions::Option opencl_opt(
      litert::EnvironmentOptions::Tag::kOpenClDeviceId,
      static_cast<int64_t>(1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(
          litert::EnvironmentOptions({dir_opt, webgpu_opt, opencl_opt})));

  // WebGPU is active, but its plugin is absent. Even though OpenCL is also
  // present in options and on disk, it must fall back to CPU rather than
  // selecting the mismatched OpenCL backend.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
}

TEST_F(DynamicSamplerFactoryTest,
       FallsBackWhenPluginInRuntimeLibraryDirFailsToLoad) {
  ASSERT_FALSE(test_dir_.empty());
  // Write invalid non-ELF bytes to libLiteRtTopKWebGpuSampler.so so that
  // std::filesystem::exists(full_path) is true, but SharedLibrary::Load fails.
  std::filesystem::path corrupt_lib =
      test_dir() / ("libLiteRtTopKWebGpuSampler" SO_EXT);
  {
    std::ofstream ofs(corrupt_lib, std::ios::binary);
    ofs << "NOT_AN_ELF_BINARY";
  }

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());
  litert::EnvironmentOptions::Option dev_opt(
      litert::EnvironmentOptions::Tag::kWebGpuDevice, static_cast<int64_t>(1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(
          litert::EnvironmentOptions({dir_opt, dev_opt})));

  // The corrupt plugin fails to load, logs a warning, falls back to loading
  // by library name (which is also absent), and finally falls back to CPU
  // TopPSampler.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt));
  ASSERT_NE(sampler, nullptr);
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler.get());
  EXPECT_NE(top_p_sampler, nullptr);
}

TEST_F(DynamicSamplerFactoryTest,
       PropagatesNonFallbackErrorWithoutFallingThrough) {
  ASSERT_OK(InstallWebGpuFake());
  ASSERT_OK(InstallOpenClFake());

  std::string dir_str = test_dir().string();
  litert::EnvironmentOptions::Option dir_opt(
      litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, dir_str.c_str());

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::Environment env,
      litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));

  // vocab_size = 400 triggers InvalidArgumentError in fake WebGPU sampler.
  // Even without an explicit backend requested, a non-fallback error (like
  // InvalidArgument) must be returned immediately rather than falling
  // through to OpenCL.
  EXPECT_THAT(
      CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                    env, /*sequence_size=*/1, /*vocab_size=*/400,
                    /*activation_data_type=*/std::nullopt),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(DynamicSamplerFactoryTest,
       FallsBackCleanlyWhenRuntimeLibraryDirIsEmptyOrNonExistent) {
  // 1. Empty kRuntimeLibraryDir string.
  {
    litert::EnvironmentOptions::Option dir_opt(
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir, "");
    LITERT_ASSERT_OK_AND_ASSIGN(
        litert::Environment env,
        litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1,
                      /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    ASSERT_NE(sampler, nullptr);
    EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  }

  // 2. Non-existent kRuntimeLibraryDir path.
  {
    litert::EnvironmentOptions::Option dir_opt(
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
        "/nonexistent/litert_lm_runtime_dir");
    LITERT_ASSERT_OK_AND_ASSIGN(
        litert::Environment env,
        litert::Environment::Create(litert::EnvironmentOptions({dir_opt})));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Sampler> sampler,
        CreateSampler(Backend::GPU, /*batch_size=*/1, TopPParams(1, 0.0f, 1.0f),
                      env, /*sequence_size=*/1,
                      /*vocab_size=*/201,
                      /*activation_data_type=*/std::nullopt));
    ASSERT_NE(sampler, nullptr);
    EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
  }
}

#undef SO_EXT
#endif  // defined(__linux__) && !defined(__ANDROID__)

}  // namespace
}  // namespace litert::lm
