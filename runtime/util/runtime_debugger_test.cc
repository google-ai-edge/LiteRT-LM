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

#include "runtime/util/runtime_debugger.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/engine/io_types.h"

namespace litert::lm {
namespace {

TEST(RuntimeDebuggerTest, ConstructorStripsTrailingSlash) {
  RuntimeDebugger debugger_with_slash("/tmp/test_dir/");
  EXPECT_EQ(debugger_with_slash.capture_dir(), "/tmp/test_dir");

  RuntimeDebugger debugger_clean("/tmp/test_dir");
  EXPECT_EQ(debugger_clean.capture_dir(), "/tmp/test_dir");
}

TEST(RuntimeDebuggerTest, CreateHandlesValidAndInvalidCacheDirectories) {
  std::string test_dir = testing::TempDir();
  auto valid_debugger = RuntimeDebugger::Create(test_dir);
  ASSERT_NE(valid_debugger, nullptr);
  EXPECT_EQ(valid_debugger->capture_dir(),
            (std::filesystem::path(test_dir) / "litert_lm_debugger").string());

  EXPECT_EQ(RuntimeDebugger::Create(""), nullptr);
  EXPECT_EQ(RuntimeDebugger::Create(":memory"), nullptr);
  EXPECT_EQ(RuntimeDebugger::Create(":nocache"), nullptr);
}

TEST(RuntimeDebuggerTest, SessionLifecycleAndSwitching) {
  std::string test_dir = testing::TempDir();
  auto debugger = RuntimeDebugger::Create(test_dir);
  ASSERT_NE(debugger, nullptr);

  debugger->RegisterDebugSession(/*session_id=*/0);
  debugger->RegisterDebugSession(/*session_id=*/1);

  // Switch to session 0 and verify tokens are captured to session 0 dir.
  debugger->SetActiveDebugSession(/*session_id=*/0);
  Responses responses_s0(TaskState::kDone, {"s0"}, {1.0f}, {1}, {{101}});
  debugger->ObserveTokens(/*session_id=*/0, responses_s0);

  std::string s0_capture_path =
      (std::filesystem::path(test_dir) / "litert_lm_debugger" / "0" /
       "generated_tokens.jsonl")
          .string();
  EXPECT_TRUE(std::filesystem::exists(s0_capture_path));

  // Setting non-existent session safely falls back without crash.
  debugger->SetActiveDebugSession(/*session_id=*/999);

  // Unregister session 0 and verify cleanup.
  debugger->SetActiveDebugSession(/*session_id=*/0);
  debugger->UnregisterDebugSession(/*session_id=*/0);
}

TEST(RuntimeDebuggerTest, GraphRunCallbacksReturnCallables) {
  RuntimeDebugger debugger("/tmp/capture");
  auto pre_callback = debugger.CreatePreGraphRunCallback();
  EXPECT_TRUE(pre_callback != nullptr);

  auto post_callback = debugger.CreatePostGraphRunCallback();
  EXPECT_TRUE(post_callback != nullptr);
}

TEST(RuntimeDebuggerTest, ObserveTokensWritesJsonlAndHandlesEmpty) {
  std::string test_dir = testing::TempDir();
  RuntimeDebugger debugger(test_dir);
  debugger.RegisterDebugSession(/*session_id=*/0);

  // 1. Valid responses are captured with structured JSONL fields.
  Responses responses(TaskState::kDone, {"hello"}, {1.0f}, {2}, {{123, 456}});
  debugger.ObserveTokens(/*session_id=*/0, responses);

  std::string captured_path =
      (std::filesystem::path(test_dir) / "0" / "generated_tokens.jsonl")
          .string();
  std::ifstream infile(captured_path);
  ASSERT_TRUE(infile.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(infile, line));
  EXPECT_THAT(line, testing::HasSubstr("\"token_ids\":[[123,456]]"));
  EXPECT_THAT(line, testing::HasSubstr("\"texts\":[\"hello\"]"));
  EXPECT_THAT(line, testing::HasSubstr("\"scores\":[1.0]"));
  infile.close();

  // 2. Empty tokens cleanly return without crashing.
  Responses empty_responses(TaskState::kProcessing);
  debugger.ObserveTokens(/*session_id=*/0, empty_responses);
}

TEST(RuntimeDebuggerTest, DualCheckpointKvAndPerStepProbeCapturing) {
  // Declare environment before debugger to guarantee correct destruction order
  // in dynamic linking environments.
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  std::string test_dir = testing::TempDir();
  RuntimeDebugger debugger(test_dir);
  debugger.RegisterDebugSession(/*session_id=*/0);
  debugger.SetActiveDebugSession(/*session_id=*/0);

  auto callback = debugger.CreatePostGraphRunCallback();
  ASSERT_NE(callback, nullptr);

  auto ranked_type =
      ::litert::RankedTensorType(::litert::ElementType::Float32,
                                 ::litert::Layout(::litert::Dimensions{2, 2}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto kv_buffer,
      ::litert::TensorBuffer::CreateManaged(
          *env, ::litert::TensorBufferType::kHostMemory, ranked_type, 16));
  LITERT_ASSERT_OK(kv_buffer.Clear());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto probe_buffer,
      ::litert::TensorBuffer::CreateManaged(
          *env, ::litert::TensorBufferType::kHostMemory, ranked_type, 16));
  LITERT_ASSERT_OK(probe_buffer.Clear());

  // 1. Prefill Step: Captures BOTH probe and KV cache immediately.
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_outputs;
  LITERT_ASSERT_OK_AND_ASSIGN(auto probe_dup_1, probe_buffer.Duplicate());
  prefill_outputs.emplace("probe_prefill_out_0", std::move(probe_dup_1));

  LITERT_ASSERT_OK_AND_ASSIGN(auto kv_dup_1, kv_buffer.Duplicate());
  prefill_outputs.emplace("kv_cache_k_0", std::move(kv_dup_1));

  callback("prefill_128", /*current_step=*/0, prefill_outputs);

  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(test_dir) / "0" /
      "prefill_128_post_probe_prefill_out_0_step_0.safetensors"));
  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(test_dir) / "0" /
      "prefill_128_post_kv_cache_k_0_step_0.safetensors"));

  // 2. Decode Step 0: Captures probe, but does NOT save decode KV cache yet.
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> decode_outputs;
  LITERT_ASSERT_OK_AND_ASSIGN(auto probe_dup_2, probe_buffer.Duplicate());
  decode_outputs.emplace("probe_decode_out_0", std::move(probe_dup_2));

  LITERT_ASSERT_OK_AND_ASSIGN(auto kv_dup_2, kv_buffer.Duplicate());
  decode_outputs.emplace("kv_cache_k_0", std::move(kv_dup_2));

  callback("decode", /*current_step=*/0, decode_outputs);

  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(test_dir) / "0" /
      "decode_post_probe_decode_out_0_step_0.safetensors"));
  EXPECT_FALSE(
      std::filesystem::exists(std::filesystem::path(test_dir) / "0" /
                              "decode_post_kv_cache_k_0_step_0.safetensors"));

  // 3. Decode Step 1: Captures probe.
  callback("decode", /*current_step=*/1, decode_outputs);
  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(test_dir) / "0" /
      "decode_post_probe_decode_out_0_step_1.safetensors"));

  // 4. Turn Completion (kDone): ObserveTokens flushes final decode KV cache!
  Responses done_response(TaskState::kDone, {"done"}, {1.0f}, {1}, {{123}});
  debugger.ObserveTokens(/*session_id=*/0, done_response);

  EXPECT_TRUE(
      std::filesystem::exists(std::filesystem::path(test_dir) / "0" /
                              "decode_post_kv_cache_k_0_step_1.safetensors"));
}

}  // namespace
}  // namespace litert::lm
