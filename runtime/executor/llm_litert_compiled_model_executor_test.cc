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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {
constexpr char kPrefillDecodeModelNameInTaskBundle[] = "TF_LITE_PREFILL_DECODE";
const int kMaxNumTokens = 32;
const int kNumThreads = 4;

using ::litert::Expected;
using ::litert::Model;
using ::litert::lm::ModelAssetBundleResources;
using ::litert::lm::proto::ExternalFile;
using ::testing::_;  // NOLINT: Required by ASSERT_OK_AND_ASSIGN().

absl::StatusOr<std::unique_ptr<ExecutorModelResources>>
CreateExecutorModelResources(absl::string_view model_path) {
  auto executor_model_resources = std::make_unique<ExecutorModelResources>();
  litert::Expected<Model> litert_model;
  std::unique_ptr<ModelAssetBundleResources> resources;
  auto external_file = std::make_unique<ExternalFile>();
  external_file->set_file_name(std::string(model_path));
  ASSIGN_OR_RETURN(resources,  // NOLINT: wrongly detected.
                   ModelAssetBundleResources::Create(
                       /*tag=*/"", std::move(external_file)));

  ASSIGN_OR_RETURN(absl::string_view buffer,  // NOLINT: wrongly detected.
                   resources->GetFile(kPrefillDecodeModelNameInTaskBundle));
  litert::BufferRef<uint8_t> buffer_ref(buffer.data(), buffer.size());
  litert_model = Model::CreateFromBuffer(buffer_ref);
  RET_CHECK(litert_model) << "Failed to build "
                          << kPrefillDecodeModelNameInTaskBundle << " model.";
  executor_model_resources->model_asset_bundle_resources = std::move(resources);
  executor_model_resources->litert_model = std::move(*litert_model);
  return executor_model_resources;
}

TEST(LlmLiteRTCompiledModelExecutorTest, CreateExecutorTest) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.task";
  ASSERT_OK_AND_ASSIGN(auto model_resources,
                       CreateExecutorModelResources(model_path.string()));
  ModelAssets model_assets;
  model_assets.model_paths.push_back(model_path);
  LlmExecutorSettings executor_settings(model_assets);
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  ::litert::lm::CpuConfig config;
  config.number_of_threads = kNumThreads;
  executor_settings.SetBackendConfig(config);
  executor_settings.SetBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(auto executor,
                       LlmLiteRtCompiledModelExecutor::Create(
                           executor_settings, model_resources->litert_model));
  ASSERT_NE(executor, nullptr);
}

}  // namespace
}  // namespace litert::lm
