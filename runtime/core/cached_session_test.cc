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

#include "runtime/core/cached_session.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/prefix_cache.h"
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"

namespace litert::lm {
namespace {

class NoOpTaskController : public SessionInterface::TaskController {
 public:
  NoOpTaskController() = default;
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return absl::OkStatus();
  }
  absl::Status Cancel() override { return absl::OkStatus(); }
};

class MockSession : public SessionInterface {
 public:
  MockSession() {
    ON_CALL(*this, GetSessionConfig).WillByDefault(testing::ReturnRef(config_));
  }

  void SetSessionConfig(SessionConfig config) { config_ = std::move(config); }

  MOCK_METHOD(absl::StatusOr<Responses>, GenerateContent,
              (const std::vector<InputData>& contents), (override));
  MOCK_METHOD(absl::Status, GenerateContentStream,
              (const std::vector<InputData>& contents,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback),
              (override));
  MOCK_METHOD(absl::Status, GenerateContentStream,
              (const std::vector<InputData>& contents,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
               const DecodeConfig& decode_config),
              (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunTextScoring,
              (const std::vector<absl::string_view>& target_text,
               bool store_token_lengths),
              (override));
  MOCK_METHOD(absl::Status, RunPrefill,
              (const std::vector<InputData>& contents), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>, RunPrefillAsync,
              (const std::vector<InputData>& contents,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>,
              PrefillPreprocessedContents,
              (std::vector<InputData> preprocessed_contents,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback),
              (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode, (), (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode,
              (const DecodeConfig& decode_config), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>, RunDecodeAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>, RunDecodeAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
               const DecodeConfig& decode_config),
              (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo>, GetBenchmarkInfo, (), (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo*>, GetMutableBenchmarkInfo, (),
              (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (), (override));
  MOCK_METHOD(const SessionConfig&, GetSessionConfig, (), (const, override));
  MOCK_METHOD(absl::Status, RewindToStep, (int step), (override));

 private:
  SessionConfig config_ = SessionConfig::CreateDefault();
};

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

std::string GetSentencePieceModelPath() {
  return absl::StrCat(::testing::SrcDir(), "/", kTestdataDir,
                      "sentencepiece.model");
}

class CachedSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto sentencepiece_tokenizer =
        SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
    ASSERT_OK(sentencepiece_tokenizer.status());
    tokenizer_ = std::move(*sentencepiece_tokenizer);
    auto env = litert::Environment::Create({});
    env_ = std::move(*env);
  }

  litert::TensorBuffer CreateDummyTensorBuffer(int num_elements,
                                               float value = 1.0f) {
    RankedTensorType tensor_type(GetElementType<float>(),
                                 Layout(Dimensions({1, num_elements})));
    auto buffer =
        TensorBuffer::CreateManaged(*env_, TensorBufferType::kHostMemory,
                                    tensor_type, num_elements * sizeof(float));
    std::vector<float> dummy_data(num_elements, value);
    (void)buffer->Write<float>(dummy_data);
    return std::move(*buffer);
  }

  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
  std::optional<litert::Environment> env_;
};

TEST_F(CachedSessionTest, BasicPrefillCacheHit) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  CachedSession cached_session(std::move(mock_session), tokenizer_.get());

  // First run: prefill "hello" (33, 547, 58).
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 1);
            const auto& input_text = std::get<InputText>(contents[0]);
            EXPECT_TRUE(input_text.IsTensorBuffer());

            auto tensor = input_text.GetPreprocessedTextTensor();
            EXPECT_TRUE(tensor.ok());
            if (tensor.ok()) {
              auto ids = Tokenizer::TensorBufferToTokenIds(**tensor);
              EXPECT_TRUE(ids.ok());
              if (ids.ok()) {
                EXPECT_EQ(ids->size(), 1);
                EXPECT_THAT((*ids)[0], testing::ElementsAre(33, 547, 58));
              }
            }

            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs1;
  inputs1.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs1));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58}));

  // Second run: prefill "hello world" -> incoming [33, 547, 58, 359].
  // Cache has [33, 547, 58]. Longest match is 3.
  EXPECT_CALL(*mock_session_ptr, RewindToStep(3))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 1);
            EXPECT_TRUE(std::get<InputText>(contents[0]).IsTensorBuffer());
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs2;
  inputs2.push_back(InputText("hello"));
  inputs2.push_back(InputText("world"));
  EXPECT_OK(cached_session.RunPrefill(inputs2));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58, 359}));
}

TEST_F(CachedSessionTest, PrefillCacheHitWithText) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  CachedSession cached_session(std::move(mock_session), tokenizer_.get());

  // Prime the cache first with "hello" (33, 547, 58).
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs1;
  inputs1.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs1));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58}));

  // Second run: "hello" + preprocessed "world" (359).
  EXPECT_CALL(*mock_session_ptr, RewindToStep(3))
      .WillOnce(testing::Return(absl::OkStatus()));

  auto world_tensor = Tokenizer::TokenIdsToTensorBuffer(std::vector<int>{359});
  ASSERT_OK(world_tensor.status());

  LiteRtTensorBuffer original_handle = world_tensor->Get();

  std::vector<InputData> inputs2;
  inputs2.push_back(InputText("hello"));
  inputs2.push_back(InputText(std::move(*world_tensor)));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [original_handle](
              std::vector<InputData> contents,
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
              -> absl::StatusOr<
                  std::unique_ptr<SessionInterface::TaskController>> {
            EXPECT_EQ(contents.size(), 1);
            const auto& sliced_input = std::get<InputText>(contents[0]);
            EXPECT_TRUE(sliced_input.IsTensorBuffer());

            auto sliced_tensor = sliced_input.GetPreprocessedTextTensor();
            if (!sliced_tensor.ok()) {
              ADD_FAILURE()
                  << "sliced_tensor is not OK: " << sliced_tensor.status();
              return sliced_tensor.status();
            }
            EXPECT_EQ((*sliced_tensor)->Get(), original_handle);

            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(inputs2));
}

TEST_F(CachedSessionTest, PartialTextCacheHit) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  CachedSession cached_session(std::move(mock_session), tokenizer_.get());

  // First run: prefill "hello" (33, 547, 58) to prime the cache.
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 1);
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs1;
  inputs1.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs1));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58}));

  // Second run: prefill a SINGLE InputText containing "hello world" ->
  // [33, 547, 58, 359].
  // The cache matches the prefix "hello" (3 tokens).
  // The remaining token is "world" (359).
  // The system should:
  // 1. Call RewindToStep(3).
  // 2. Slice the single InputText and convert "world" (359) into a
  //    preprocessed TensorBuffer.
  // 3. Call RunPrefillAsync with a single preprocessed InputText.

  EXPECT_CALL(*mock_session_ptr, RewindToStep(3))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce([](std::vector<InputData> contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
                    -> absl::StatusOr<
                        std::unique_ptr<SessionInterface::TaskController>> {
        EXPECT_EQ(contents.size(), 1);
        const auto& sliced_input = std::get<InputText>(contents[0]);
        EXPECT_TRUE(sliced_input.IsTensorBuffer());

        auto sliced_tensor = sliced_input.GetPreprocessedTextTensor();
        if (!sliced_tensor.ok()) {
          ADD_FAILURE() << "sliced_tensor is not OK: "
                        << sliced_tensor.status();
          return sliced_tensor.status();
        }

        // Verify that the sliced buffer contains exactly the token ID 359
        // ("world").
        auto ids = Tokenizer::TensorBufferToTokenIds(**sliced_tensor);
        if (!ids.ok()) {
          ADD_FAILURE() << "ids is not OK: " << ids.status();
          return ids.status();
        }
        EXPECT_EQ(ids->size(), 1);
        EXPECT_THAT((*ids)[0], testing::ElementsAre(359));

        callback(Responses(TaskState::kDone));
        return std::make_unique<NoOpTaskController>();
      });

  std::vector<InputData> inputs2;
  inputs2.push_back(InputText("hello world"));
  EXPECT_OK(cached_session.RunPrefill(inputs2));

  // Cache should now contain the merged tokens: [33, 547, 58, 359]
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58, 359}));
}

TEST_F(CachedSessionTest, MultimodalCacheMiss) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 100;

  CachedSessionOptions options;
  options.vision_properties = vision_props;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  auto image_buffer1 = CreateDummyTensorBuffer(100, 1.0f);
  std::vector<InputData> contents1;
  contents1.push_back(InputImage(std::move(image_buffer1)));
  contents1.push_back(InputText("hello"));

  // First run: cache is empty, match is 0.
  // It should:
  // 1. Call RewindToStep(0).
  // 2. Call RunPrefillAsync with original contents.
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 2);
            EXPECT_TRUE(std::holds_alternative<InputImage>(contents[0]));
            EXPECT_TRUE(std::holds_alternative<InputText>(contents[1]));
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(contents1));

  // Cache should now contain MediaHash (100 tokens) + "hello" (33, 547, 58) = 4
  // elements.
  std::string first_run_hash;
  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 4);
    EXPECT_TRUE(std::holds_alternative<MediaHash>(elements[0]));
    if (std::holds_alternative<MediaHash>(elements[0])) {
      first_run_hash = std::get<MediaHash>(elements[0]).hash;
      EXPECT_EQ(std::get<MediaHash>(elements[0]).token_length, 100);
    }
    EXPECT_EQ(elements[1], CacheElement(33));
    EXPECT_EQ(elements[2], CacheElement(547));
    EXPECT_EQ(elements[3], CacheElement(58));
    EXPECT_EQ(cached_session.GetPrefixCache().TokenLength(), 103);
  }

  // Second run: prefill a DIFFERENT image + "hello".
  // Since the image buffer contents are filled with 2.0f instead of 1.0f,
  // its cryptographic hash will differ from the cached one.
  // This results in a complete cache miss (matched_elements = 0).
  // It should fallback to:
  // 1. Call RewindToStep(0).
  // 2. Clear the cache.
  // 3. Call RunPrefillAsync with the new original contents.

  auto image_buffer2 = CreateDummyTensorBuffer(100, 2.0f);
  std::vector<InputData> contents2;
  contents2.push_back(InputImage(std::move(image_buffer2)));
  contents2.push_back(InputText("hello"));

  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 2);
            EXPECT_TRUE(std::holds_alternative<InputImage>(contents[0]));
            EXPECT_TRUE(std::holds_alternative<InputText>(contents[1]));
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(contents2));

  // The old cache should be completely cleared, and the new MediaHash + "hello"
  // should be populated.
  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 4);
    EXPECT_TRUE(std::holds_alternative<MediaHash>(elements[0]));
    if (std::holds_alternative<MediaHash>(elements[0])) {
      EXPECT_EQ(std::get<MediaHash>(elements[0]).token_length, 100);
      EXPECT_NE(std::get<MediaHash>(elements[0]).hash, first_run_hash);
    }
    EXPECT_EQ(elements[1], CacheElement(33));
    EXPECT_EQ(elements[2], CacheElement(547));
    EXPECT_EQ(elements[3], CacheElement(58));
    EXPECT_EQ(cached_session.GetPrefixCache().TokenLength(), 103);
  }
}

TEST_F(CachedSessionTest, MultimodalCacheHit) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 100;

  CachedSessionOptions options;
  options.vision_properties = vision_props;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  // Prime the cache first.
  // We can do this by running a prefill which we mock.
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  auto image_buffer1 = CreateDummyTensorBuffer(100);
  std::vector<InputData> contents1;
  contents1.push_back(InputImage(std::move(image_buffer1)));
  contents1.push_back(InputText("hello"));

  EXPECT_OK(cached_session.RunPrefill(contents1));
  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 4);
    EXPECT_TRUE(std::holds_alternative<MediaHash>(elements[0]));
    if (std::holds_alternative<MediaHash>(elements[0])) {
      EXPECT_EQ(std::get<MediaHash>(elements[0]).token_length, 100);
    }
    EXPECT_EQ(elements[1], CacheElement(33));
    EXPECT_EQ(elements[2], CacheElement(547));
    EXPECT_EQ(elements[3], CacheElement(58));
  }

  // Second run: prefill same Image + "hello" + "world".
  // We use a new image buffer but with SAME content so it hashes to same value.
  auto image_buffer2 = CreateDummyTensorBuffer(100);
  std::vector<InputData> contents2;
  contents2.push_back(InputImage(std::move(image_buffer2)));
  contents2.push_back(InputText("hello"));
  contents2.push_back(InputText("world"));

  // Incoming elements: [MediaHash, 33, 547, 58, 359]
  // Cache: [MediaHash, 33, 547, 58]
  // Match: 4 elements, 103 tokens (100 for image + 3 for "hello").
  // It should:
  // 1. Call RewindToStep(103).
  // 2. Call RunPrefillAsync with sliced "world" (359).

  EXPECT_CALL(*mock_session_ptr, RewindToStep(103))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 1);
            EXPECT_TRUE(std::get<InputText>(contents[0]).IsTensorBuffer());
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(contents2));
  // Cache should now contain: [MediaHash, 33, 547, 58, 359]
  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 5);
    EXPECT_TRUE(std::holds_alternative<MediaHash>(elements[0]));
    if (std::holds_alternative<MediaHash>(elements[0])) {
      EXPECT_EQ(std::get<MediaHash>(elements[0]).token_length, 100);
    }
    EXPECT_EQ(elements[1], CacheElement(33));
    EXPECT_EQ(elements[2], CacheElement(547));
    EXPECT_EQ(elements[3], CacheElement(58));
    EXPECT_EQ(elements[4], CacheElement(359));
  }
}

TEST_F(CachedSessionTest, MultimodalCacheHitWithRemainingMedia) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 100;

  CachedSessionOptions options;
  options.vision_properties = vision_props;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  // Prime cache with "hello" (33, 547, 58)
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });
  std::vector<InputData> inputs1;
  inputs1.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs1));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58}));

  // Second run: "hello" + Image + "world".
  // Cache: [33, 547, 58] (3 tokens)
  // Incoming: [33, 547, 58, MediaHash, 359]
  // Match: 3 elements ("hello"), 3 tokens.
  // Remaining: [MediaHash, 359]
  // With dynamic media prefilling, it should perform a cached prefill:
  // 1. Call RewindToStep(3).
  // 2. Call RunPrefillAsync with sliced contents:
  //    [InputImage, InputText("world")] (size = 2).
  // 3. Update cache to contain all elements:
  //    [33, 547, 58, MediaHash, 359] (size = 5).

  EXPECT_CALL(*mock_session_ptr, RewindToStep(3))
      .WillOnce(testing::Return(absl::OkStatus()));

  auto image_buffer = CreateDummyTensorBuffer(100);
  std::vector<InputData> contents;
  contents.push_back(InputText("hello"));
  contents.push_back(InputImage(std::move(image_buffer)));
  contents.push_back(InputText("world"));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 2);
            EXPECT_TRUE(std::holds_alternative<InputImage>(contents[0]));
            EXPECT_TRUE(std::get<InputText>(contents[1]).IsTensorBuffer());
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(contents));
  // Cache should now contain all elements: [33, 547, 58, MediaHash, 359]
  // (size = 5, token_length = 3 + 100 + 1 = 104)
  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 5);
    EXPECT_EQ(elements[0], CacheElement(33));
    EXPECT_EQ(elements[1], CacheElement(547));
    EXPECT_EQ(elements[2], CacheElement(58));
    EXPECT_TRUE(std::holds_alternative<MediaHash>(elements[3]));
    if (std::holds_alternative<MediaHash>(elements[3])) {
      EXPECT_EQ(std::get<MediaHash>(elements[3]).token_length, 100);
    }
    EXPECT_EQ(elements[4], CacheElement(359));
    EXPECT_EQ(cached_session.GetPrefixCache().TokenLength(), 104);
  }
}

TEST_F(CachedSessionTest, CacheTruncationOnMismatch) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  CachedSession cached_session(std::move(mock_session), tokenizer_.get());

  // Prime cache with "hello world" (33, 547, 58, 359).
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs1;
  inputs1.push_back(InputText("hello"));
  inputs1.push_back(InputText("world"));
  EXPECT_OK(cached_session.RunPrefill(inputs1));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58, 359}));

  // Second prefill: "hello friends" -> "hello" (33, 547, 58) is matched,
  // but "friends" (70, 2037, 150, 8) is not.
  // So matched_elements = 3 ("hello"), matched_tokens = 3.
  // It should:
  // 1. Call RewindToStep(3).
  // 2. Truncate cache to size 3.
  // 3. Call RunPrefillAsync with sliced "friends".
  std::vector<int> expected_tokens = {70, 2037, 150, 8};

  EXPECT_CALL(*mock_session_ptr, RewindToStep(3))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [&expected_tokens](
              std::vector<InputData> contents,
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
              -> absl::StatusOr<
                  std::unique_ptr<SessionInterface::TaskController>> {
            EXPECT_EQ(contents.size(), 1);
            const auto& sliced_input = std::get<InputText>(contents[0]);
            EXPECT_TRUE(sliced_input.IsTensorBuffer());
            auto sliced_tensor = sliced_input.GetPreprocessedTextTensor();
            if (!sliced_tensor.ok()) {
              ADD_FAILURE()
                  << "sliced_tensor is not OK: " << sliced_tensor.status();
              return sliced_tensor.status();
            }
            auto ids = Tokenizer::TensorBufferToTokenIds(**sliced_tensor);
            if (!ids.ok()) {
              ADD_FAILURE() << "ids is not OK: " << ids.status();
              return ids.status();
            }
            EXPECT_EQ(ids->size(), 1);
            EXPECT_EQ((*ids)[0], expected_tokens);

            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs2;
  inputs2.push_back(InputText("hello friends"));
  EXPECT_OK(cached_session.RunPrefill(inputs2));

  std::vector<CacheElement> expected_cache = {33, 547, 58, 70, 2037, 150, 8};
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(), expected_cache);
}

TEST_F(CachedSessionTest, CompleteCacheHitZeroPrefill) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  CachedSession cached_session(std::move(mock_session), tokenizer_.get());

  // Prime the cache with "hello" (33, 547, 58).
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs1;
  inputs1.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs1));

  // Second prefill: exact same "hello" (33, 547, 58).
  // Since matched_elements == incoming_elements.size(),
  // it should NOT call mock_session's RewindToStep or RunPrefillAsync.
  std::vector<InputData> inputs2;
  inputs2.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs2));

  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({33, 547, 58}));
}

TEST_F(CachedSessionTest, MultimodalPartialMatchWithMultipleMedia) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 100;

  AudioExecutorProperties audio_props;
  audio_props.audio_shrink_factor = 4;

  CachedSessionOptions options;
  options.vision_properties = vision_props;
  options.audio_properties = audio_props;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  // Prime cache with: Image1 + "hello"
  auto image_buffer1 = CreateDummyTensorBuffer(100);
  std::vector<InputData> contents1;
  contents1.push_back(InputImage(std::move(image_buffer1)));
  contents1.push_back(InputText("hello"));

  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(contents1));

  std::string image_hash;
  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 4);
    image_hash = std::get<MediaHash>(elements[0]).hash;
    EXPECT_EQ(std::get<MediaHash>(elements[0]).token_length, 100);
    EXPECT_EQ(elements[1], CacheElement(33));
    EXPECT_EQ(elements[2], CacheElement(547));
    EXPECT_EQ(elements[3], CacheElement(58));
  }

  // Second run: Image1 + "hello" + Audio1 + "world".
  // Cache: [MediaHash(image), 33, 547, 58] (103 tokens)
  // Incoming: [MediaHash(image), 33, 547, 58, MediaHash(audio), 359]
  EXPECT_CALL(*mock_session_ptr, RewindToStep(103))
      .WillOnce(testing::Return(absl::OkStatus()));

  RankedTensorType audio_tensor_type(GetElementType<float>(),
                                     Layout(Dimensions({1, 12, 1})));
  auto audio_buffer =
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kHostMemory,
                                  audio_tensor_type, 12 * sizeof(float));
  ASSERT_TRUE(audio_buffer.HasValue());
  LiteRtTensorBuffer original_audio_handle = audio_buffer->Get();

  std::vector<InputData> contents2;
  auto image_buffer2 = CreateDummyTensorBuffer(100);
  contents2.push_back(InputImage(std::move(image_buffer2)));
  contents2.push_back(InputText("hello"));
  contents2.push_back(InputAudio(std::move(*audio_buffer)));
  contents2.push_back(InputText("world"));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [original_audio_handle](
              std::vector<InputData> contents,
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
              -> absl::StatusOr<
                  std::unique_ptr<SessionInterface::TaskController>> {
            EXPECT_EQ(contents.size(), 2);
            EXPECT_TRUE(std::holds_alternative<InputAudio>(contents[0]));
            EXPECT_TRUE(std::get<InputText>(contents[1]).IsTensorBuffer());

            const auto& sliced_audio = std::get<InputAudio>(contents[0]);
            auto sliced_audio_tensor =
                sliced_audio.GetPreprocessedAudioTensor();
            if (!sliced_audio_tensor.ok()) {
              ADD_FAILURE() << "sliced_audio_tensor is not OK: "
                            << sliced_audio_tensor.status();
              return sliced_audio_tensor.status();
            }
            EXPECT_EQ((*sliced_audio_tensor)->Get(), original_audio_handle);

            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  EXPECT_OK(cached_session.RunPrefill(contents2));

  {
    const auto& elements = cached_session.GetPrefixCache().GetElements();
    ASSERT_EQ(elements.size(), 6);
    EXPECT_EQ(elements[0], CacheElement(MediaHash{image_hash, 100}));
    EXPECT_EQ(elements[1], CacheElement(33));
    EXPECT_EQ(elements[2], CacheElement(547));
    EXPECT_EQ(elements[3], CacheElement(58));
    EXPECT_TRUE(std::holds_alternative<MediaHash>(elements[4]));
    if (std::holds_alternative<MediaHash>(elements[4])) {
      EXPECT_EQ(std::get<MediaHash>(elements[4]).token_length, 3);
    }
    EXPECT_EQ(elements[5], CacheElement(359));
    EXPECT_EQ(cached_session.GetPrefixCache().TokenLength(), 107);
  }
}

TEST_F(CachedSessionTest, ImageWithoutVisionPropertiesError) {
  auto mock_session = std::make_unique<MockSession>();
  CachedSession cached_session(std::move(mock_session), tokenizer_.get());

  auto image_buffer = CreateDummyTensorBuffer(100);
  std::vector<InputData> contents;
  contents.push_back(InputImage(std::move(image_buffer)));

  auto status = cached_session.RunPrefill(contents);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Vision properties not set"));
}

TEST_F(CachedSessionTest, RawAudioInputError) {
  auto mock_session = std::make_unique<MockSession>();
  AudioExecutorProperties audio_props;
  audio_props.audio_shrink_factor = 4;

  CachedSessionOptions options;
  options.audio_properties = audio_props;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  std::vector<InputData> contents;
  contents.push_back(InputAudio("raw_audio_data_pcm"));

  auto status = cached_session.RunPrefill(contents);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Audio must be preprocessed before being used in "
                         "SessionAdvanced."));
}

TEST_F(CachedSessionTest, RawImageInputError) {
  auto mock_session = std::make_unique<MockSession>();
  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 100;

  CachedSessionOptions options;
  options.vision_properties = vision_props;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  std::vector<InputData> contents;
  contents.push_back(InputImage("raw_image_bytes_abc"));

  auto status = cached_session.RunPrefill(contents);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Image must be preprocessed before being used in "
                         "SessionAdvanced."));
}

TEST_F(CachedSessionTest, SessionAdvancedMockExecutor) {
  // 1. Tokenize inputs dynamically to find exact expected IDs.
  ASSERT_OK_AND_ASSIGN(std::vector<int> hello_ids,
                       tokenizer_->TextToTokenIds("hello"));
  ASSERT_OK_AND_ASSIGN(std::vector<int> hello_world_ids,
                       tokenizer_->TextToTokenIds("hello world"));

  ASSERT_GE(hello_world_ids.size(), hello_ids.size());
  std::vector<int> world_sliced_ids(hello_world_ids.begin() + hello_ids.size(),
                                    hello_world_ids.end());

  // 2. Setup Fake Executor and ThreadedExecutionManager.
  std::vector<std::vector<int>> prefill_tokens_set = {hello_ids,
                                                      world_sliced_ids};
  std::vector<std::vector<int>> decode_tokens_set = {{999}};

  auto executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/32000, prefill_tokens_set, decode_tokens_set,
      /*batch_size=*/1);

  ASSERT_OK_AND_ASSIGN(
      auto execution_manager,
      ThreadedExecutionManager::Create(
          tokenizer_.get(), /*model_resources=*/nullptr, std::move(executor),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, &*env_));

  std::shared_ptr<ThreadedExecutionManager> execution_manager_shared =
      std::move(execution_manager);

  // 3. Setup SessionAdvanced.
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.SetApplyPromptTemplateInSession(false);

  ASSERT_OK_AND_ASSIGN(
      auto session_advanced,
      SessionAdvanced::Create(execution_manager_shared, tokenizer_.get(),
                              session_config,
                              /*benchmark_info=*/std::nullopt));

  // 4. Setup CachedSession.
  CachedSession cached_session(std::move(session_advanced), tokenizer_.get());

  // 5. First prefill run (prompt: "hello").
  std::vector<InputData> contents1;
  contents1.push_back(InputText("hello"));

  EXPECT_OK(cached_session.RunPrefill(contents1));

  // 6. Second prefill run (prompt: "hello world" sharing "hello" prefix).
  std::vector<InputData> contents2;
  contents2.push_back(InputText("hello world"));

  EXPECT_OK(cached_session.RunPrefill(contents2));
}

TEST_F(CachedSessionTest, InsertBosTokenIdFromConstructor) {
  auto mock_session = std::make_unique<MockSession>();
  auto* mock_session_ptr = mock_session.get();

  // Set the start token ID to 2 (BOS token) in the mock config.
  SessionConfig config = SessionConfig::CreateDefault();
  config.SetStartTokenId(2);
  mock_session_ptr->SetSessionConfig(config);

  // Initialize CachedSession with insert_bos_token_id = true.
  CachedSessionOptions options;
  options.insert_bos_token_id = true;
  CachedSession cached_session(std::move(mock_session), tokenizer_.get(),
                               options);

  // Prefill "hello" (33, 547, 58).
  // With insert_bos_token_id = true, it should prefill [2, 33, 547, 58] (4
  // tokens).
  EXPECT_CALL(*mock_session_ptr, RewindToStep(0))
      .WillOnce(testing::Return(absl::OkStatus()));

  EXPECT_CALL(*mock_session_ptr,
              PrefillPreprocessedContents(testing::_, testing::_))
      .WillOnce(
          [](std::vector<InputData> contents,
             absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
            EXPECT_EQ(contents.size(), 2);  // BOS token + "hello" preprocessed

            // First item: BOS token ID (2)
            const auto& first_input = std::get<InputText>(contents[0]);
            EXPECT_TRUE(first_input.IsTensorBuffer());
            auto first_tensor = first_input.GetPreprocessedTextTensor();
            EXPECT_TRUE(first_tensor.ok());
            auto first_ids = Tokenizer::TensorBufferToTokenIds(**first_tensor);
            EXPECT_TRUE(first_ids.ok());
            EXPECT_EQ((*first_ids)[0], std::vector<int>{2});

            // Second item: "hello" token IDs (33, 547, 58)
            const auto& second_input = std::get<InputText>(contents[1]);
            EXPECT_TRUE(second_input.IsTensorBuffer());
            auto second_tensor = second_input.GetPreprocessedTextTensor();
            EXPECT_TRUE(second_tensor.ok());
            auto second_ids =
                Tokenizer::TensorBufferToTokenIds(**second_tensor);
            EXPECT_TRUE(second_ids.ok());
            EXPECT_EQ((*second_ids)[0], std::vector<int>({33, 547, 58}));

            callback(Responses(TaskState::kDone));
            return std::make_unique<NoOpTaskController>();
          });

  std::vector<InputData> inputs;
  inputs.push_back(InputText("hello"));
  EXPECT_OK(cached_session.RunPrefill(inputs));
  EXPECT_EQ(cached_session.GetPrefixCache().GetElements(),
            std::vector<CacheElement>({2, 33, 547, 58}));
}

}  // namespace
}  // namespace litert::lm
