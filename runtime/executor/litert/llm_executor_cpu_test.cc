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

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

const int kNumThreads = 4;
const int kMaxNumTokens = 32;

absl::StatusOr<std::unique_ptr<ModelResources>>
CreateExecutorModelResourcesLitertLm(absl::string_view model_path) {
  ASSIGN_OR_RETURN(auto scoped_file, ScopedFile::Open(model_path));
  return ModelResourcesLitertLm::Create(
      std::make_unique<LitertLmLoader>(std::move(scoped_file)));
}

absl::StatusOr<std::pair<std::unique_ptr<ModelResources>,
                         std::unique_ptr<LlmExecutorExternalSamplerDynamic>>>
CreateExternalSamplerDynamicExecutor(Environment& env,
                                     const std::string& model_path,
                                     uint32_t kv_increment_size = 8,
                                     int prefill_chunk_size = -1) {
  auto path = std::filesystem::path(::testing::SrcDir()) / model_path;
  ASSIGN_OR_RETURN(auto model_resources,
                   CreateExecutorModelResourcesLitertLm(path.string()));
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(path.string()));
  ASSIGN_OR_RETURN(auto executor_settings, LlmExecutorSettings::CreateDefault(
                                               model_assets, Backend::CPU));
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetMaxNumTokens(kMaxNumTokens);
  CpuConfig config;
  config.number_of_threads = kNumThreads;
  config.kv_increment_size = kv_increment_size;
  config.prefill_chunk_size = prefill_chunk_size;
  executor_settings.SetBackendConfig(config);
  ASSIGN_OR_RETURN(auto executor,
                   LlmExecutorExternalSamplerDynamic::Create(
                       executor_settings, env, *model_resources));
  return std::make_pair(std::move(model_resources), std::move(executor));
}

}  // namespace
}  // namespace litert::lm
