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

#include "runtime/core/session_factory.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

class FakeTokenizer : public Tokenizer {
 public:
  FakeTokenizer() = default;

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    return std::vector<int>{1, 2, 3};
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    return token == "BOS" ? 2 : 1;
  }

  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override {
    return "fake_text";
  }

  TokenizerType GetTokenizerType() const override {
    return TokenizerType::kUnspecified;
  }
};

TEST(SessionFactoryTest, InitializeSession) {
  FakeTokenizer tokenizer;
  std::vector<std::vector<int>> stop_token_ids = {{1}, {2}};
  std::vector<std::vector<int>> dummy_tokens = {{0}};
  FakeLlmExecutor executor(256, dummy_tokens, dummy_tokens);
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetSamplerBackend(Backend::CPU);
  ThreadPool worker_thread_pool("testpool", /*max_num_threads=*/1);
  auto session =
      InitializeSession(&executor, &tokenizer,
                        /*vision_executor=*/nullptr,
                        /*audio_executor=*/nullptr, session_config,
                        /*benchmark_info=*/std::nullopt, &worker_thread_pool);
  EXPECT_OK(session);
}

}  // namespace
}  // namespace litert::lm
