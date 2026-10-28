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

#include "runtime/components/model_resources_litert_lm.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::lm::proto::LlmMetadata;

constexpr char kTestLmPath[] =
    "litert_lm/runtime/testdata/test_lm.litertlm";

// Loads a TFLite model from a .litertlm file with file-backed loading enabled.
// With the dynamically linked LiteRT runtime on Windows this used to pass a C
// runtime file descriptor into libLiteRt.dll and crash with 0xC0000409
// (https://github.com/google-ai-edge/LiteRT-LM/issues/4023); it must load the
// model (falling back to buffer-backed loading where needed) on every platform
// and link mode.
TEST(ModelResourcesLitertLmTest, GetTFLiteModelWithFileBackedLoading) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) / kTestLmPath;
  ASSERT_OK_AND_ASSIGN(auto model_file, ScopedFile::Open(model_path.string()));
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(model_file)));
  ASSERT_GT(loader->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE).Size(),
            0);

  ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResourcesLitertLm::Create(
          std::move(loader), /*enable_file_backed_model_loading=*/true));
  ASSERT_OK_AND_ASSIGN(
      const auto* tflite_model,
      model_resources->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
  EXPECT_GT(tflite_model->GetNumSignatures(), 0);
}

}  // namespace
}  // namespace litert::lm
