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

#include "runtime/executor/llm_litert_mtp_drafter.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_requirements.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "litert/vendors/nvidia/cache_layout.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

class TfLiteModelResources : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<TfLiteModelResources>> Create(
      const ModelAssets& model_assets, bool with_mtp_drafter = false) {
    LITERT_ASSIGN_OR_RETURN(auto path, model_assets.GetPath());
    LITERT_ASSIGN_OR_RETURN(auto model,
                            Model::CreateFromFile(std::string(path)));
    return absl::WrapUnique(
        new TfLiteModelResources(std::move(model), with_mtp_drafter));
  }

  absl::StatusOr<const Model*> GetTFLiteModel(ModelType model_type) override {
    if (model_type == ModelType::kTfLitePrefillDecode) {
      return &model_;
    }
    if (model_type == ModelType::kTfLiteMtpDrafter) {
      if (with_mtp_drafter_) {
        return &model_;
      } else {
        return absl::NotFoundError("MTP Drafter model not found");
      }
    }
    return absl::UnimplementedError("Unsupported model type");
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    return absl::UnimplementedError("GetTFLiteModelBuffer not implemented.");
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("GetTokenizer not implemented.");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("GetLlmMetadata not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("GetExecutorMetadata not implemented.");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

 private:
  explicit TfLiteModelResources(Model model, bool with_mtp_drafter = false)
      : model_(std::move(model)), with_mtp_drafter_(with_mtp_drafter) {}

  Model model_;
  bool with_mtp_drafter_;
};

TEST(LlmLiteRtMtpDrafterTest, CreateFromModelResources_MissingVerifySignature) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto executor_settings,
      LlmExecutorSettings::CreateDefault(model_assets, Backend::CPU));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       TfLiteModelResources::Create(model_assets,
                                                    /*with_mtp_drafter=*/true));

  const std::filesystem::path embedder_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/dummy_embedder.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(auto embedder_model,
                              Model::CreateFromFile(embedder_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_lookup,
      EmbeddingLookupManager::Create(env, &embedder_model,
                                     /*fully_supports_multi_modal=*/false));

  for (bool select_main_signature : {false, true}) {
    SCOPED_TRACE(select_main_signature);
    const std::vector<std::string> selected_signatures =
        select_main_signature ? std::vector<std::string>{"main_only_signature"}
                              : std::vector<std::string>{};
    executor_settings.SetSelectedSignatures(selected_signatures);
    auto drafter_or = LlmLiteRtMtpDrafter::Create(
        env, *model_resources, executor_settings, base_model, *embedding_lookup,
        /*ple_manager=*/std::nullopt);
    // The drafter must compile independently of the main model's selection and
    // reach the missing verifier signature in both cases.
    EXPECT_THAT(drafter_or, StatusIs(absl::StatusCode::kNotFound));
    EXPECT_EQ(executor_settings.GetSelectedSignatures(), selected_signatures);
  }
}

TEST(LlmLiteRtMtpDrafterTest,
     CreateFromPreCompiledModel_MissingVerifySignature) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/magic_test_decode_batch.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto executor_settings,
                       LlmExecutorSettings::CreateDefault(model_assets));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto mtp_model,
                              Model::CreateFromFile(model_path.string()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_mtp_model,
      CompiledModel::Create(env, model_path.string(),
                            litert::HwAccelerators::kCpu));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto base_model, CompiledModel::Create(env, model_path.string(),
                                             litert::HwAccelerators::kCpu));

  const std::filesystem::path embedder_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/dummy_embedder.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(auto embedder_model,
                              Model::CreateFromFile(embedder_path.string()));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_lookup,
      EmbeddingLookupManager::Create(env, &embedder_model,
                                     /*fully_supports_multi_modal=*/false));

  auto drafter_or = LlmLiteRtMtpDrafter::Create(
      env, std::move(compiled_mtp_model), executor_settings, base_model,
      mtp_model, *embedding_lookup,
      /*ple_manager=*/std::nullopt);
  EXPECT_THAT(drafter_or, StatusIs(absl::StatusCode::kNotFound));
}

TEST(LlmLiteRtMtpDrafterTest, UpdateCompilationOptions) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto settings_gpu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::GPU));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  EXPECT_OK(UpdateCompilationOptions(settings_gpu, options));

  ASSERT_OK_AND_ASSIGN(auto settings_cpu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::CPU));
  EXPECT_OK(UpdateCompilationOptions(settings_cpu, options));
}

TEST(LlmLiteRtMtpDrafterTest,
     UpdateCompilationOptionsAcceptsNpuWithoutChangingAccelerators) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto settings_npu, LlmExecutorSettings::CreateDefault(
                                              model_assets, Backend::NPU));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  LITERT_ASSERT_OK(options.SetHardwareAccelerators(HwAccelerators::kNpu |
                                                   HwAccelerators::kCpu));
  LITERT_ASSERT_OK_AND_ASSIGN(auto accelerators,
                              options.GetHardwareAccelerators());

  EXPECT_OK(UpdateCompilationOptions(settings_npu, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto updated_accelerators,
                              options.GetHardwareAccelerators());
  EXPECT_EQ(updated_accelerators, accelerators);
}

TEST(LlmLiteRtMtpDrafterTest, RecognizesNvidiaTransposedValueCache) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType),
      TensorBufferType::kHostMemory};
  const std::vector<uint32_t> strides = {2359296, 294912, 1, 256};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto requirements,
      TensorBufferRequirements::Create(buffer_types, 4718592, strides));

  ASSERT_OK_AND_ASSIGN(bool transposed,
                       UsesNvidiaTransposedValueCache(layout, requirements));
  EXPECT_TRUE(transposed);
}

TEST(LlmLiteRtMtpDrafterTest, LeavesNativeNvidiaValueCacheUnchanged) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType)};
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

TEST(LlmLiteRtMtpDrafterTest, LeavesOtherBackendStridesUnchanged) {
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

TEST(LlmLiteRtMtpDrafterTest, RejectsUnsupportedNvidiaValueCacheStrides) {
  const Layout layout(Dimensions{1, 8, 256, 1152});
  const std::vector<TensorBufferType> buffer_types = {
      static_cast<TensorBufferType>(nvidia::kNvidiaCudaTensorBufferType)};
  for (const auto& strides : std::vector<std::vector<uint32_t>>{
           {2359296, 294912, 2, 256}, {294912, 1, 256}}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto requirements,
        TensorBufferRequirements::Create(buffer_types, 4718592, strides));

    EXPECT_THAT(UsesNvidiaTransposedValueCache(layout, requirements),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(LlmLiteRtMtpDrafterTest, VerificationCannotCrossLocalCacheRingBoundary) {
  const MtpVerificationState state(/*num_verify_tokens=*/4,
                                   /*global_capacity=*/131071,
                                   /*local_ring_capacity=*/1152);
  for (int ring_base : {0, 1152, 32256}) {
    EXPECT_TRUE(state.CanVerify(ring_base + 1148));
    for (int tail = 1149; tail < 1152; ++tail) {
      EXPECT_FALSE(state.CanVerify(ring_base + tail));
    }
    EXPECT_TRUE(state.CanVerify(ring_base + 1152));
  }
}

TEST(LlmLiteRtMtpDrafterTest, VerificationRespectsActualGlobalCapacity) {
  const MtpVerificationState state(/*num_verify_tokens=*/4,
                                   /*global_capacity=*/32,
                                   /*local_ring_capacity=*/std::nullopt);
  EXPECT_TRUE(state.CanVerify(28));
  for (int position : {-1, 0, 29, 30, 31, 32}) {
    EXPECT_FALSE(state.CanVerify(position));
  }
  EXPECT_TRUE(state.CanDecode(0));
  EXPECT_TRUE(state.CanDecode(31));
  EXPECT_FALSE(state.CanDecode(-1));
  EXPECT_FALSE(state.CanDecode(32));
}

TEST(LlmLiteRtMtpDrafterTest, VerificationUsesConfiguredWidthAndRingCapacity) {
  const MtpVerificationState state(/*num_verify_tokens=*/2,
                                   /*global_capacity=*/std::nullopt,
                                   /*local_ring_capacity=*/8);
  EXPECT_TRUE(state.CanVerify(6));
  EXPECT_FALSE(state.CanVerify(7));
  EXPECT_TRUE(state.CanVerify(8));
  EXPECT_TRUE(state.CanVerify(14));
  EXPECT_FALSE(state.CanVerify(15));
}

TEST(LlmLiteRtMtpDrafterTest, VerifiedActivationMatchesAcceptedPrefixPosition) {
  for (int accepted = 0; accepted <= 3; ++accepted) {
    MtpVerificationState state(/*num_verify_tokens=*/4,
                               /*global_capacity=*/128,
                               /*local_ring_capacity=*/std::nullopt);
    EXPECT_FALSE(state.ActivationIndex(50).has_value());
    state.RecordVerification(/*position=*/50, accepted);
    EXPECT_EQ(state.ActivationIndex(51 + accepted), accepted);
    EXPECT_FALSE(state.ActivationIndex(50 + accepted).has_value());
    EXPECT_FALSE(state.ActivationIndex(52 + accepted).has_value());
  }
}

TEST(LlmLiteRtMtpDrafterTest, OrdinaryDecodeInvalidatesVerifierActivations) {
  MtpVerificationState state(/*num_verify_tokens=*/4,
                             /*global_capacity=*/128,
                             /*local_ring_capacity=*/std::nullopt);
  // A new drafter cannot reuse a verifier row after ordinary decoding.
  EXPECT_FALSE(state.ActivationIndex(50).has_value());
  state.RecordVerification(/*position=*/50, /*num_accepted=*/2);
  ASSERT_EQ(state.ActivationIndex(53), 2);
  state.InvalidateActivations();
  EXPECT_FALSE(state.ActivationIndex(53).has_value());
  // After a fresh decode supplies activations, verification establishes a new
  // accepted-prefix row. The old row stays invalid even if positions repeat.
  state.RecordVerification(/*position=*/55, /*num_accepted=*/0);
  EXPECT_EQ(state.ActivationIndex(56), 0);
  EXPECT_FALSE(state.ActivationIndex(53).has_value());
}

TEST(LlmLiteRtMtpDrafterTest, VerificationCanCommitLessThanItsPhysicalWidth) {
  const MtpVerificationState state(/*num_verify_tokens=*/4,
                                   /*global_capacity=*/128,
                                   /*local_ring_capacity=*/std::nullopt);
  EXPECT_TRUE(state.CanVerify(50));
  for (int budget : {-1, 0}) {
    EXPECT_FALSE(state.CanVerify(50, budget));
  }
  for (int budget = 1; budget <= 5; ++budget) {
    EXPECT_TRUE(state.CanVerify(50, budget));
    // A small output budget does not shrink the physical verifier write.
    EXPECT_FALSE(state.CanVerify(125, budget));
  }
  const MtpVerificationState ring_state(/*num_verify_tokens=*/4,
                                        /*global_capacity=*/128,
                                        /*local_ring_capacity=*/8);
  EXPECT_TRUE(ring_state.CanVerify(4, /*output_budget=*/1));
  EXPECT_FALSE(ring_state.CanVerify(5, /*output_budget=*/1));
}

TEST(LlmLiteRtMtpDrafterTest,
     BoundedAcceptanceChoosesMatchingBonusAndActivation) {
  const std::vector<int> drafted = {10, 11, 12};
  // Each row rejects at a different position; the last accepts every draft.
  const std::vector<std::vector<int>> verified = {
      {20, 21, 22, 23}, {10, 21, 22, 23}, {10, 11, 22, 23}, {10, 11, 12, 23}};
  const int expected_accepted[4][5] = {
      {0, 0, 0, 0, 0}, {0, 1, 1, 1, 1}, {0, 1, 2, 2, 2}, {0, 1, 2, 3, 3}};
  for (int reject_at = 0; reject_at <= 3; ++reject_at) {
    for (int budget = 1; budget <= 5; ++budget) {
      SCOPED_TRACE(::testing::Message()
                   << "reject_at=" << reject_at << " budget=" << budget);
      MtpVerificationState state(/*num_verify_tokens=*/4,
                                 /*global_capacity=*/128,
                                 /*local_ring_capacity=*/std::nullopt);
      ASSERT_OK_AND_ASSIGN(
          int accepted,
          state.CountAcceptedDrafts(drafted, verified[reject_at], budget));
      ASSERT_EQ(accepted, expected_accepted[reject_at][budget - 1]);
      std::vector<int> output(drafted.begin(), drafted.begin() + accepted);
      output.push_back(verified[reject_at][accepted]);
      EXPECT_LE(output.size(), budget);
      EXPECT_EQ(output.back(),
                accepted == reject_at ? 20 + accepted : 10 + accepted);
      state.RecordVerification(/*position=*/50, accepted);
      const int next_pending_position = 50 + output.size();
      EXPECT_EQ(state.ActivationIndex(next_pending_position), accepted);
      EXPECT_FALSE(
          state.ActivationIndex(next_pending_position + 1).has_value());
    }
  }
}

TEST(LlmLiteRtMtpDrafterTest, BoundedAcceptanceValidatesInputs) {
  const MtpVerificationState state(/*num_verify_tokens=*/4,
                                   /*global_capacity=*/128,
                                   /*local_ring_capacity=*/std::nullopt);
  const std::vector<int> drafted = {10, 11, 12};
  const std::vector<int> verified = {10, 11, 12, 13};
  ASSERT_OK_AND_ASSIGN(int accepted,
                       state.CountAcceptedDrafts(drafted, verified));
  EXPECT_EQ(accepted, 3);
  for (int budget : {-1, 0}) {
    EXPECT_THAT(state.CountAcceptedDrafts(drafted, verified, budget),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  EXPECT_THAT(state.CountAcceptedDrafts(drafted, drafted, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(state.CountAcceptedDrafts(verified, verified, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmLiteRtMtpDrafterTest, OneTokenBudgetNeedsNoReloadForWarmVerifier) {
  MtpVerificationState state(/*num_verify_tokens=*/4,
                             /*global_capacity=*/128,
                             /*local_ring_capacity=*/std::nullopt);
  // The executor reserves an ordinary decode output when it must bootstrap
  // activations. With one output left, it must do only that ordinary decode.
  EXPECT_FALSE(state.ActivationIndex(50).has_value());
  EXPECT_FALSE(state.CanVerify(51, /*output_budget=*/1 - 1));
  EXPECT_TRUE(state.CanVerify(51, /*output_budget=*/2 - 1));
  // A warm verifier needs no bootstrap and can commit just its row-0 token.
  state.RecordVerification(/*position=*/47, /*num_accepted=*/2);
  ASSERT_EQ(state.ActivationIndex(50), 2);
  EXPECT_TRUE(state.CanVerify(50, /*output_budget=*/1));
  // An intervening ordinary decode requires a fresh bootstrap again.
  state.InvalidateActivations();
  EXPECT_FALSE(state.ActivationIndex(50).has_value());
  EXPECT_FALSE(state.CanVerify(51, /*output_budget=*/1 - 1));
}

}  // namespace
}  // namespace litert::lm
