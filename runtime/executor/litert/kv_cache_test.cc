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

#include "runtime/executor/litert/kv_cache.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr char kTestStaticModelPath[] =
    "litert_lm/runtime/testdata/test_lm.litertlm";

// Test model with dynamic sequence and context length dimensions.
constexpr char kTestDynamicModelPath[] =
    "litert_lm/runtime/testdata/test_lm_dynamic.litertlm";

constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr absl::string_view kPrefillSignatureRunner = "prefill";

class CompiledModelTest : public CompiledModel {
 public:
  using CompiledModel::CompiledModel;
  using CompiledModel::Create;
};

class LitertKVCacheTest : public ::testing::Test {
 protected:
  void SetUpKV(const std::string& model_path, bool inplace_update) {
    auto path = std::filesystem::path(::testing::SrcDir()) /
                std::filesystem::path(model_path);
    ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(path.string()));
    ASSERT_OK_AND_ASSIGN(auto loader,
                         LitertLmLoader::Create(std::move(scoped_file)));
    ASSERT_OK_AND_ASSIGN(resources_,
                         ModelResourcesLitertLm::Create(std::move(loader)));
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto env, Environment::Create(std::vector<Environment::Option>()));
    env_ = std::move(env);
    ASSERT_OK_AND_ASSIGN(litert_model_, resources_->GetTFLiteModel(
                                            ModelType::kTfLitePrefillDecode));
    LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options, Options::Create());
    compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto compiled_model,
        CompiledModelTest::Create(*env_, litert_model_->Get(),
                                  compilation_options));
    compiled_model_ = std::move(compiled_model);
    ASSERT_OK_AND_ASSIGN(
        kv_cache_,
        LitertKVCache::Create(*env_, *litert_model_, kDecodeSignatureRunner,
                              *compiled_model_, inplace_update));
  }

  std::unique_ptr<ModelResources> resources_;
  const litert::Model* litert_model_;
  std::optional<Environment> env_;
  std::optional<CompiledModel> compiled_model_;
  std::unique_ptr<LitertKVCache> kv_cache_;
};

#ifndef _WIN32
TEST_F(LitertKVCacheTest, CanCreateKVWithDynamicModel) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestDynamicModelPath, /*inplace_update=*/false));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 1);
}
#endif  // !_WIN32

TEST_F(LitertKVCacheTest, CanCreateKVWithStaticModelOutOfPlace) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestStaticModelPath, /*inplace_update=*/false));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 160);
}

TEST_F(LitertKVCacheTest, CanCreateKVWithStaticModelInPlace) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestStaticModelPath, /*inplace_update=*/true));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 160);
}

TEST_F(LitertKVCacheTest, SerializeNotSupported) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestStaticModelPath, /*inplace_update=*/true));
  EXPECT_THAT(kv_cache_->Serialize(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LitertKVCacheTest, LoadNotSupported) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestStaticModelPath, /*inplace_update=*/true));
  EXPECT_THAT(kv_cache_->Load(""), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LitertKVCacheTest, StaticKVNotResizeable) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestStaticModelPath, /*inplace_update=*/true));
  EXPECT_THAT(kv_cache_->Resize(100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LitertKVCacheTest, InplaceDynamicKVResizeable) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestDynamicModelPath, /*inplace_update=*/true));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 1);
  EXPECT_OK(kv_cache_->Resize(100));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 100);
}

TEST_F(LitertKVCacheTest, OutOfPlaceDynamicKVNotResizeable) {
  ASSERT_NO_FATAL_FAILURE(
      SetUpKV(kTestDynamicModelPath, /*inplace_update=*/false));
  EXPECT_THAT(kv_cache_->Resize(100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
