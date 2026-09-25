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

#include "omni/tts/kokoro/kokoro_factory.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/espeak_assets.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/text_chunk_utils.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/scoped_file.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
namespace {

using ::testing::ElementsAre;

// ModelResources stub that only reports GenericBinaryData section names.
class FakeSectionNames : public lm::ModelResources {
 public:
  explicit FakeSectionNames(std::vector<std::string> names)
      : names_(std::move(names)) {}

  std::vector<std::string> GetGenericBinaryDataNames() const override {
    return names_;
  }

  // Unused parts of the interface.
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::ModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::ModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

 private:
  std::vector<std::string> names_;
};

TEST(KokoroFactoryTest, InitKokoroResourcesFailsWithNonexistentModels) {
  KokoroModelConfig config;
  config.acoustic_file = "nonexistent_acoustic.tflite";
  config.vocoder_file = "nonexistent_vocoder.tflite";

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  ModelResources resources(shared_env);

  auto status =
      InitKokoroResources(config, "/tmp/invalid_path", "", lm::Backend::CPU, 1,
                          *shared_env, resources);
  EXPECT_FALSE(status.ok());
}

TEST(KokoroFactoryTest, CreateKokoroComponentsRejectsMissingModels) {
  KokoroModelConfig config;
  config.target_bucket = 128;

  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<::litert::Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  TextChunkConfig chunk_config;
  chunk_config.max_buffer_size = 0;

  // Since ModelResources does not have compiled models loaded,
  // CreateKokoroComponents should propagate the error.
  auto components = CreateKokoroComponents(
      config, "/tmp", std::make_unique<StreamTextSource>(chunk_config),
      resources);
  EXPECT_FALSE(components.ok());
}

TEST(KokoroFactoryTest, GetAvailableKokoroVoicesSkipsNonVoiceSections) {
  FakeSectionNames lm_resources(
      {"af_heart", std::string(kokoro::kEspeakNgSectionName), "ja-lexicon",
       "zh-lexicon", "zh-textnorm", "ef_dora.bin"});

  EXPECT_THAT(GetAvailableKokoroVoices(/*model_folder=*/"", &lm_resources),
              ElementsAre("af_heart", "ef_dora"));
}

}  // namespace
}  // namespace litert::omni::tts
