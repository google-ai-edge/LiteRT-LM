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

#include "runtime/executor/litert/state.h"

#include <filesystem>  // NOLINT
#include <limits>
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
#include "runtime/proto/executor_metadata.pb.h"
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

class LitertStateTest : public ::testing::Test {
 protected:
  void SetUpKV(const std::string& model_path,
               LitertState::AllocationPolicy allocation_policy, int batch_size,
               const proto::ExecutorMetadata* executor_metadata = nullptr,
               bool clear_kv_cache_before_prefill = true) {
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
        LitertState::Create(*env_, *compiled_model_, kDecodeSignatureRunner,
                            executor_metadata, allocation_policy, batch_size,
                            clear_kv_cache_before_prefill));
  }

  std::unique_ptr<ModelResources> resources_;
  const litert::Model* litert_model_;
  std::optional<Environment> env_;
  std::optional<CompiledModel> compiled_model_;
  std::unique_ptr<LitertState> kv_cache_;
};

#ifndef _WIN32
TEST_F(LitertStateTest, CanCreateKVWithDynamicModel) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 1);
}
#endif  // !_WIN32

TEST_F(LitertStateTest, CanCreateKVWithStaticModelOutOfPlace) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 128);
}

TEST_F(LitertStateTest, CanCreateKVWithStaticModelInPlace) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 128);
}

TEST_F(LitertStateTest, CanCreateKVWithClearFlagFalse) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1,
                                  /*executor_metadata=*/nullptr,
                                  /*clear_kv_cache_before_prefill=*/false));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 128);
}

TEST_F(LitertStateTest, SerializeNotSupported) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Serialize(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LitertStateTest, LoadNotSupported) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Load(""), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LitertStateTest, StaticKVNotResizeable) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Resize(*compiled_model_, kDecodeSignatureRunner, 100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LitertStateTest, InplaceDynamicKVResizeable) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 1);
  EXPECT_OK(kv_cache_->Resize(*compiled_model_, kDecodeSignatureRunner, 100));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 100);
}

TEST_F(LitertStateTest, OutOfPlaceDynamicKVNotResizeable) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Resize(*compiled_model_, kDecodeSignatureRunner, 100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
