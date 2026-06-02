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

#include "runtime/executor/magic_number_configs_helper.h"

#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

// Model without magic numbers.
//   prefill: context_length = 1280, prefill_length = 1024
//   decode: context_length = 1280
constexpr absl::string_view kTestModelPathNone = "magic_test_none.tflite";

// Model with magic numbers for context length.
//   prefill: context_length = 8209, prefill_length = 1024
//   decode: context_length = 8209
//   test_prefill_1280: context_length = 1280, prefill_length = 1024
//   test_decode_1280: context_length = 1280
constexpr absl::string_view kTestModelPathContextLength =
    "magic_test_context_length.tflite";

// Model with magic numbers both for context length and prefill length.
//   prefill: context_length = 8209, prefill_length = 4099
//   decode: context_length = 8209
//   test_prefill_1280: context_length = 1280, prefill_length = 1024
//   test_decode_1280: context_length = 1280
constexpr absl::string_view kTestModelPathBoth = "magic_test_both.tflite";

// Model with magic numbers for context length, prefill length, and decode
// batch size.
//   prefill: context_length = 8209, prefill_length = 4099
//   decode: context_length = 8209, decode_batch_size = 11
//   test_prefill_1280: context_length = 1280, prefill_length = 1024
//   test_decode_1280: context_length = 1280, decode_batch_size = 3
constexpr absl::string_view kTestModelPathDecodeBatch =
    "magic_test_decode_batch.tflite";

// Model with magic numbers with multiple prefill lengths.
//   prefill_4099: context_length = 8209, prefill_length = 4099
//   prefill_1031: context_length = 8209, prefill_length = 1031
//   prefill_257: context_length = 8209, prefill_length = 257
//   prefill_67: context_length = 8209, prefill_length = 67
//   decode: context_length = 8209
//   test_prefill_1024: context_length = 1280, prefill_length = 1024
//   test_prefill_256: context_length = 1280, prefill_length = 256
//   test_decode_1280: context_length = 1280
constexpr absl::string_view kTestModelPathMulti = "magic_test_multi.tflite";

// Helper to get the idx-th element of the array. It's to avoid static array
// bounds check on Windows.
template <typename T>
T& Get(T* ptr, int idx) {
  return ptr[idx];
}

absl::StatusOr<Model> LoadModelFromFile(absl::string_view model_path) {
  auto model_path_in_srcdir =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/" /
      std::string(model_path);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(model_path_in_srcdir.string()));
  return model;
}

class ModelResourcesMock : public ModelResources {
 public:
  MOCK_METHOD(absl::StatusOr<const proto::LlmMetadata*>, GetLlmMetadata, (),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Tokenizer>>, GetTokenizer, (),
              (override));
  MOCK_METHOD(absl::StatusOr<absl::string_view>, GetTFLiteModelBuffer,
              (ModelType model_type), (override));
  MOCK_METHOD(std::optional<std::string>, GetTFLiteModelBackendConstraint,
              (ModelType model_type), (override));
  MOCK_METHOD(std::optional<std::string>, GetTFLiteModelPreferActivationType,
              (ModelType model_type), (override));
  MOCK_METHOD(absl::StatusOr<std::reference_wrapper<ScopedFile>>, GetScopedFile,
              (), (override));
  MOCK_METHOD((absl::StatusOr<std::pair<size_t, size_t>>),
              GetWeightsSectionOffset, (ModelType model_type), (override));

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    return &model_;
  }

  explicit ModelResourcesMock(const Model& model) : model_(model) {}

 private:
  const Model& model_;
};

absl::StatusOr<LlmExecutorSettings> GetLlmExecutorSettings() {
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create("dont_care_path"));
  return LlmExecutorSettings::CreateDefault(std::move(model_assets));
}

TEST(MagicNumberConfigsHelperTest, None_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathNone);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  // No magic number configs and verifications.
  EXPECT_TRUE(env_options.empty());
  EXPECT_EQ(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, None_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathNone);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  // No magic number configs and verifications.
  EXPECT_TRUE(env_options.empty());
  EXPECT_EQ(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, ContextLength_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, ContextLength_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest,
     ContextLength_ExplictSettingsLargerThanMagicNumbers) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(9000);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest,
     ContextLength_ExplictSettingsWithVerifications) {
  auto model = LoadModelFromFile(kTestModelPathContextLength);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 1);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 2);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      Get(helper.magic_number_verifications()->verifications, 1);
  EXPECT_EQ(std::string(verification1.signature), "prefill");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1280");
  EXPECT_EQ(verification1.is_superset, true);
}

TEST(MagicNumberConfigsHelperTest, Both_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 4096);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Both_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 1024);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Both_ExplictSettingsLargerThanMagicNumbers) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(9000);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {5000}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 4096);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Both_ExplictSettingsWithVerifications) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024},
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 1024);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 2);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      Get(helper.magic_number_verifications()->verifications, 1);
  EXPECT_EQ(std::string(verification1.signature), "prefill");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1280");
  EXPECT_EQ(verification1.is_superset, true);
}

TEST(MagicNumberConfigsHelperTest,
     Both_ExplictSettingsWithVerifications_MatchedPartially) {
  auto model = LoadModelFromFile(kTestModelPathBoth);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  // prefill_batch_sizes is not matched with test_prefill_1280.
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {512},
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 2);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 4099);
  EXPECT_EQ(config1.target_number, 512);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 1);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  // prefill won't be verified because prefill_batch_size is not matched with
  // test_prefill_1280.
}

TEST(MagicNumberConfigsHelperTest, DecodeBatch_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathDecodeBatch);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 3);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 11);
  EXPECT_EQ(config1.target_number, 1);
  EXPECT_EQ(std::string(config1.signature_prefix), "decode");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 4099);
  EXPECT_EQ(config2.target_number, 4096);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, DecodeBatch_ExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathDecodeBatch);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024},
                                     .num_output_candidates = 3};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 3);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 11);
  EXPECT_EQ(config1.target_number, 3);
  EXPECT_EQ(std::string(config1.signature_prefix), "decode");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 4099);
  EXPECT_EQ(config2.target_number, 1024);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest,
     DecodeBatch_ExplictSettingsLargerThanMagicNumbers) {
  auto model = LoadModelFromFile(kTestModelPathDecodeBatch);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(9000);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {5000},
                                     .num_output_candidates = 20};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 3);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 11);
  EXPECT_EQ(config1.target_number, 8);
  EXPECT_EQ(std::string(config1.signature_prefix), "decode");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 4099);
  EXPECT_EQ(config2.target_number, 4096);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest,
     DecodeBatch_ExplictSettingsWithVerifications) {
  auto model = LoadModelFromFile(kTestModelPathDecodeBatch);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024},
                                     .num_output_candidates = 3,
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 3);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 11);
  EXPECT_EQ(config1.target_number, 3);
  EXPECT_EQ(std::string(config1.signature_prefix), "decode");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 4099);
  EXPECT_EQ(config2.target_number, 1024);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 2);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      Get(helper.magic_number_verifications()->verifications, 1);
  EXPECT_EQ(std::string(verification1.signature), "prefill");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1280");
  EXPECT_EQ(verification1.is_superset, true);
}

TEST(MagicNumberConfigsHelperTest,
     DecodeBatch_ExplictSettingsWithVerifications_MatchedPartially) {
  auto model = LoadModelFromFile(kTestModelPathDecodeBatch);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  // prefill_batch_sizes is not matched with test_prefill_1280.
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {512},
                                     .num_output_candidates = 3,
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 3);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 11);
  EXPECT_EQ(config1.target_number, 3);
  EXPECT_EQ(std::string(config1.signature_prefix), "decode");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 4099);
  EXPECT_EQ(config2.target_number, 512);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 1);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  // prefill won't be verified because prefill_batch_size is not matched with
  // test_prefill_1280.
}

TEST(MagicNumberConfigsHelperTest, Multi_DefaultSettings) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 256);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 1024);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 4096);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Multi_LessExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024, 128}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 128);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 1024);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  // Last prefill_batch_size is capped by the context length, 1280.
  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 1280);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest,
     Multi_LessExplictSettings_LargerThanMagicNumbers) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(8192);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {512, 6144}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 8192);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 256);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 512);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 4096);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Multi_MoreExplictSettings) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(3072);
  AdvancedSettings advanced_settings{
      .prefill_batch_sizes = {1024, 128, 2048, 32, 8}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  // Since 5 batch sizes are given and 4 magic numbers are available, the first
  // small batch size, 8, is skipped.
  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 3072);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 32);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 128);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 1024);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 2048);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Multi_MoreExplictSettings_SkipLast) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(3072);
  AdvancedSettings advanced_settings{
      .prefill_batch_sizes = {1024, 128, 256, 512, 2048, 32}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 3072);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Since 6 batch sizes are given and 4 magic numbers are available, the first
  // 2 small batch sizes, 32, 128, are skipped and the first 256 doesn't fit in
  // 67, the default target number 64 is used. The last 2048 is also skipped
  // because no more magic numbers are available.
  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 256);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 512);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 1024);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest,
     Multi_MoreExplictSettings_LargePrefillLengths) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(256);
  AdvancedSettings advanced_settings{
      .prefill_batch_sizes = {1024, 64, 256, 512, 128}};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 1);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 256);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  // Since 6 batch sizes are given and 4 magic numbers are available, the first
  // small batch size, 64 is skipped and the first 256 doesn't fit in 67, the
  // default target number 64 is used. The last 2 lengths, 512, 1024 are larger
  // than the context length 256, so they are also capped to 256.
  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 128);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 256);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 256);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  // Verifications are disabled by default.
  EXPECT_EQ(helper.magic_number_verifications(), nullptr);
}

TEST(MagicNumberConfigsHelperTest, Multi_LessExplictSettingsWithVerifications) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024, 256},
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 256);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 1024);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  // Last prefill_batch_size is capped by the context length, 1280.
  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 1280);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 3);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      Get(helper.magic_number_verifications()->verifications, 1);
  EXPECT_EQ(std::string(verification1.signature), "prefill_1031");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1024");
  EXPECT_EQ(verification1.is_superset, true);

  const auto& verification2 =
      Get(helper.magic_number_verifications()->verifications, 2);
  EXPECT_EQ(std::string(verification2.signature), "prefill_257");
  EXPECT_EQ(std::string(verification2.test_signature), "test_prefill_256");
  EXPECT_EQ(verification2.is_superset, true);
}

TEST(MagicNumberConfigsHelperTest,
     Multi_LessExplictSettingsWithVerifications_MatchedPartially) {
  auto model = LoadModelFromFile(kTestModelPathMulti);
  EXPECT_OK(model);
  auto executor_settings = GetLlmExecutorSettings();
  EXPECT_OK(executor_settings);
  executor_settings->SetMaxNumTokens(1280);
  AdvancedSettings advanced_settings{.prefill_batch_sizes = {1024, 128, 512},
                                     .verify_magic_numbers = true};
  executor_settings->SetAdvancedSettings(advanced_settings);

  ModelResourcesMock model_resources(*model);
  MagicNumberConfigsHelper helper;
  auto env_options =
      helper.GetLiteRtEnvOptions(model_resources, *executor_settings);
  EXPECT_EQ(env_options.size(), 2);
  EXPECT_NE(helper.magic_number_configs(), nullptr);
  EXPECT_EQ(helper.magic_number_configs()->num_configs, 5);

  const auto& config0 = Get(helper.magic_number_configs()->configs, 0);
  EXPECT_EQ(config0.magic_number, 8209);
  EXPECT_EQ(config0.target_number, 1280);
  EXPECT_EQ(config0.signature_prefix, nullptr);

  const auto& config1 = Get(helper.magic_number_configs()->configs, 1);
  EXPECT_EQ(config1.magic_number, 67);
  EXPECT_EQ(config1.target_number, 64);
  EXPECT_EQ(std::string(config1.signature_prefix), "prefill");

  const auto& config2 = Get(helper.magic_number_configs()->configs, 2);
  EXPECT_EQ(config2.magic_number, 257);
  EXPECT_EQ(config2.target_number, 128);
  EXPECT_EQ(std::string(config2.signature_prefix), "prefill");

  const auto& config3 = Get(helper.magic_number_configs()->configs, 3);
  EXPECT_EQ(config3.magic_number, 1031);
  EXPECT_EQ(config3.target_number, 512);
  EXPECT_EQ(std::string(config3.signature_prefix), "prefill");

  const auto& config4 = Get(helper.magic_number_configs()->configs, 4);
  EXPECT_EQ(config4.magic_number, 4099);
  EXPECT_EQ(config4.target_number, 1024);
  EXPECT_EQ(std::string(config4.signature_prefix), "prefill");

  EXPECT_NE(helper.magic_number_verifications(), nullptr);
  EXPECT_EQ(helper.magic_number_verifications()->num_verifications, 2);

  const auto& verification0 =
      Get(helper.magic_number_verifications()->verifications, 0);
  EXPECT_EQ(std::string(verification0.signature), "decode");
  EXPECT_EQ(std::string(verification0.test_signature), "test_decode_1280");
  EXPECT_EQ(verification0.is_superset, true);

  const auto& verification1 =
      Get(helper.magic_number_verifications()->verifications, 1);
  EXPECT_EQ(std::string(verification1.signature), "prefill_4099");
  EXPECT_EQ(std::string(verification1.test_signature), "test_prefill_1024");
  EXPECT_EQ(verification1.is_superset, true);
}

}  // namespace
}  // namespace litert::lm
