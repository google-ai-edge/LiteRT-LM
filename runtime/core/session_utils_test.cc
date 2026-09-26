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

#include "runtime/core/session_utils.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/sentencepiece_tokenizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using SentencePieceTokenizer = ::litert::support::SentencePieceTokenizer;
using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;

namespace {

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/components/testdata/";

class ExtendedTokenizer : public Tokenizer {
 public:
  static absl::StatusOr<std::unique_ptr<ExtendedTokenizer>> CreateFromFile(
      absl::string_view model_path) {
    ABSL_ASSIGN_OR_RETURN(auto tokenizer,
                          SentencePieceTokenizer::CreateFromFile(model_path));
    return absl::WrapUnique(new ExtendedTokenizer(std::move(tokenizer)));
  }

  void SetExtendedToken(int token_id, absl::string_view token_str) {
    extended_tokens_to_id_[token_str] = token_id;
    id_to_extended_tokens_[token_id] = token_str;
  }

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    std::vector<int> token_ids;
    bool is_extended_token_found = false;
    do {
      is_extended_token_found = false;
      for (const auto& [extended_token_str, extended_token_id] :
           extended_tokens_to_id_) {
        auto extended_token_pos = text.find(extended_token_str);
        if (extended_token_pos != std::string::npos) {
          // The text before the extended token.
          ABSL_ASSIGN_OR_RETURN(
              auto text_ids,
              tokenizer_->TextToTokenIds(text.substr(0, extended_token_pos)));
          token_ids.insert(token_ids.end(), text_ids.begin(), text_ids.end());
          token_ids.push_back(extended_token_id);
          text = text.substr(extended_token_pos + extended_token_str.size());
          is_extended_token_found = true;
        }
      }
    } while (is_extended_token_found);
    if (!text.empty()) {
      ABSL_ASSIGN_OR_RETURN(auto text_ids, tokenizer_->TextToTokenIds(text));
      token_ids.insert(token_ids.end(), text_ids.begin(), text_ids.end());
    }
    return token_ids;
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) override {
    std::vector<std::string> token_strs;
    for (int token_id : token_ids) {
      if (id_to_extended_tokens_.contains(token_id)) {
        token_strs.push_back(id_to_extended_tokens_[token_id]);
      } else {
        token_strs.push_back(tokenizer_->TokenIdsToText({token_id}).value());
      }
    }
    return absl::StrJoin(token_strs, "");
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (extended_tokens_to_id_.contains(token)) {
      return extended_tokens_to_id_[token];
    }
    return tokenizer_->TokenToId(token);
  }

  TokenizerType GetTokenizerType() const override {
    return tokenizer_->GetTokenizerType();
  }

  std::vector<std::string> GetTokens() const override {
    return tokenizer_->GetTokens();
  }

  int GetVocabSize() const override { return tokenizer_->GetVocabSize(); }

 private:
  explicit ExtendedTokenizer(std::unique_ptr<SentencePieceTokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {};

  absl::flat_hash_map<int, std::string> id_to_extended_tokens_;
  absl::flat_hash_map<std::string, int> extended_tokens_to_id_;
  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
};

class MockEngine : public Engine {
 public:
  MOCK_METHOD(const EngineSettings&, GetEngineSettings, (), (const, override));
  MOCK_METHOD(const support::Tokenizer&, GetTokenizer, (), (const, override));
  MOCK_METHOD(absl::StatusOr<AudioExecutorProperties>,
              GetAudioExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<VisionExecutorProperties>,
              GetVisionExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<SessionInterface>>, CreateSession,
              (const SessionConfig& session_config), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (absl::Duration timeout),
              (override));
};

class SessionUtilsTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = ExtendedTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) /
         std::string(kTestdataDir) / "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer.value()->SetExtendedToken(256000, " Букмекерлер");
    tokenizer_ = std::move(*tokenizer);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
};

TEST_F(SessionUtilsTest, MaybeGetBosString) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);  // Corresponds to "</s>"
  ASSERT_OK_AND_ASSIGN(auto bos_string,
                       MaybeGetBosString(session_config, *tokenizer_));
  EXPECT_EQ(bos_string, "</s>");
}

TEST_F(SessionUtilsTest, StringToProcessedInputText) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);  // Corresponds to "</s>"
  std::optional<BenchmarkInfo> benchmark_info;
  ASSERT_OK_AND_ASSIGN(auto input_text, StringToProcessedInputText(
                                            "</s>Hello World!", session_config,
                                            *tokenizer_, benchmark_info));
  ASSERT_TRUE(input_text.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto text_tensor,
                       input_text.GetPreprocessedTextTensor());
  ASSERT_NE(text_tensor, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto token_ids_span,
                              ReferTensorBufferAsSpan<int>(*text_tensor));
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(2, 90, 547, 58, 735, 210, 466, 2294));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesFails) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);  // Corresponds to "</s>"

  std::vector<InputData> inputs_with_bos;
  inputs_with_bos.emplace_back(InputText("</s>Hello World!"));
  EXPECT_THAT(
      ApplyPromptTemplates(inputs_with_bos, ContentType::kFirst, session_config,
                           *tokenizer_, /*is_first_turn=*/true),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                "Input contains bos control token. Control "
                                "token should not be included in the input."));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesCanHandleEmptyContent) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);  // Corresponds to "</s>"
  {
    std::vector<InputData> empty_inputs;
    ASSERT_OK_AND_ASSIGN(
        auto templated_single,
        ApplyPromptTemplates(empty_inputs, ContentType::kFirst, session_config,
                             *tokenizer_, /*is_first_turn=*/true));
    ASSERT_EQ(templated_single.size(), 1);
    EXPECT_THAT(std::get<InputText>(templated_single[0]).GetRawTextString(),
                testing::status::IsOkAndHolds("</s>"));
  }

  for (const auto& content_type :
       {ContentType::kFirst, ContentType::kLast, ContentType::kMiddle}) {
    std::vector<InputData> empty_inputs;
    ASSERT_OK_AND_ASSIGN(
        auto templated_empty,
        ApplyPromptTemplates(empty_inputs, content_type, session_config,
                             *tokenizer_, /*is_first_turn=*/false));
    EXPECT_TRUE(templated_empty.empty());
  }
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesWithSingleTextChunk) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  {
    std::vector<InputData> single_chunk;
    single_chunk.emplace_back(InputText("Hello "));
    ASSERT_OK_AND_ASSIGN(
        auto templated_single,
        ApplyPromptTemplates(single_chunk, ContentType::kFirst, session_config,
                             *tokenizer_, /*is_first_turn=*/true));
    ASSERT_EQ(templated_single.size(), 2);
    EXPECT_THAT(std::get<InputText>(templated_single[0]).GetRawTextString(),
                testing::status::IsOkAndHolds("</s>"));
    EXPECT_THAT(std::get<InputText>(templated_single[1]).GetRawTextString(),
                testing::status::IsOkAndHolds("<test>User\nHello "));
  }
  {
    std::vector<InputData> single_chunk;
    single_chunk.emplace_back(InputText("world!"));
    ASSERT_OK_AND_ASSIGN(
        auto templated_single,
        ApplyPromptTemplates(single_chunk, ContentType::kMiddle, session_config,
                             *tokenizer_, /*is_first_turn=*/false));
    ASSERT_EQ(templated_single.size(), 1);
    EXPECT_THAT(std::get<InputText>(templated_single[0]).GetRawTextString(),
                testing::status::IsOkAndHolds("world!"));
  }
  {
    std::vector<InputData> single_chunk;
    single_chunk.emplace_back(InputText(""));
    ASSERT_OK_AND_ASSIGN(
        auto templated_single,
        ApplyPromptTemplates(single_chunk, ContentType::kLast, session_config,
                             *tokenizer_, /*is_first_turn=*/false));
    ASSERT_EQ(templated_single.size(), 1);
    EXPECT_THAT(std::get<InputText>(templated_single[0]).GetRawTextString(),
                testing::status::IsOkAndHolds("<end>\n<test>Model\n"));
  }
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesDisabled) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");
  session_config.SetApplyPromptTemplateInSession(false);

  // Single text chunk. (is_first_chunk=true, is_last_chunk=true)
  std::vector<InputData> single_chunk;
  single_chunk.emplace_back(InputText("Hello World!"));
  ASSERT_OK_AND_ASSIGN(
      auto templated_single,
      ApplyPromptTemplates(single_chunk, ContentType::kNA, session_config,
                           *tokenizer_, /*is_first_turn=*/true));
  ASSERT_EQ(templated_single.size(), 2);
  EXPECT_THAT(std::get<InputText>(templated_single[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_single[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("Hello World!"));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesWithTwoTextChunks) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  std::vector<InputData> two_chunks;
  two_chunks.emplace_back(InputText("First"));
  two_chunks.emplace_back(InputText("Second"));
  ASSERT_OK_AND_ASSIGN(
      auto templated_two,
      ApplyPromptTemplates(two_chunks, ContentType::kFirst, session_config,
                           *tokenizer_, /*is_first_turn=*/true));
  ASSERT_EQ(templated_two.size(), 3);
  EXPECT_THAT(std::get<InputText>(templated_two[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_two[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("<test>User\nFirst"));
  EXPECT_THAT(std::get<InputText>(templated_two[2]).GetRawTextString(),
              testing::status::IsOkAndHolds("Second"));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesDisabledWithTwoTextChunks) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");
  session_config.SetApplyPromptTemplateInSession(false);

  // Two text chunks. (First chunk: is_first=true, is_last=false;
  // Second chunk: is_first=false, is_last=true)
  std::vector<InputData> two_chunks;
  two_chunks.emplace_back(InputText("First"));
  two_chunks.emplace_back(InputText("Second"));
  ASSERT_OK_AND_ASSIGN(
      auto templated_two,
      ApplyPromptTemplates(two_chunks, ContentType::kNA, session_config,
                           *tokenizer_, /*is_first_turn=*/true));
  ASSERT_EQ(templated_two.size(), 3);
  EXPECT_THAT(std::get<InputText>(templated_two[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_two[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("First"));
  EXPECT_THAT(std::get<InputText>(templated_two[2]).GetRawTextString(),
              testing::status::IsOkAndHolds("Second"));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesWithThreeTextChunks) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  // Three text chunks. (Middle chunk: is_first=false, is_last=false)
  std::vector<InputData> three_chunks;
  three_chunks.emplace_back(InputText("First"));
  three_chunks.emplace_back(InputText("Middle"));
  three_chunks.emplace_back(InputText("Last"));
  ASSERT_OK_AND_ASSIGN(auto templated_three,
                       ApplyPromptTemplates(three_chunks, ContentType::kFirst,
                                            session_config, *tokenizer_,
                                            /*is_first_turn=*/true));
  ASSERT_EQ(templated_three.size(), 4);
  EXPECT_THAT(std::get<InputText>(templated_three[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_three[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("<test>User\nFirst"));
  EXPECT_THAT(std::get<InputText>(templated_three[2]).GetRawTextString(),
              testing::status::IsOkAndHolds("Middle"));
  EXPECT_THAT(std::get<InputText>(templated_three[3]).GetRawTextString(),
              testing::status::IsOkAndHolds("Last"));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesWithMixedChunksTextAndImage) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  // Mixed chunks - text and image. Non-text inputs are passed through.
  std::vector<InputData> mixed_chunks;
  mixed_chunks.emplace_back(InputText("Text1"));
  mixed_chunks.emplace_back(InputImage("123"));
  mixed_chunks.emplace_back(InputText("Text2"));
  ASSERT_OK_AND_ASSIGN(
      auto templated_mixed,
      ApplyPromptTemplates(mixed_chunks, ContentType::kFirst, session_config,
                           *tokenizer_, /*is_first_turn=*/true));
  ASSERT_EQ(templated_mixed.size(), 4);
  EXPECT_THAT(std::get<InputText>(templated_mixed[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_mixed[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("<test>User\nText1"));
  EXPECT_TRUE(std::holds_alternative<InputImage>(templated_mixed[2]));
  EXPECT_THAT(std::get<InputText>(templated_mixed[3]).GetRawTextString(),
              testing::status::IsOkAndHolds("Text2"));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesWithSubsequentTurn) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  std::vector<InputData> single_chunk_again;
  single_chunk_again.emplace_back(InputText("Another turn"));
  ASSERT_OK_AND_ASSIGN(
      auto templated_first_turn,
      ApplyPromptTemplates(single_chunk_again, ContentType::kFirst,
                           session_config, *tokenizer_,
                           /*is_first_turn=*/true));
  ASSERT_EQ(templated_first_turn.size(), 2);
  EXPECT_THAT(std::get<InputText>(templated_first_turn[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_first_turn[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("<test>User\nAnother turn"));

  ASSERT_OK_AND_ASSIGN(
      auto templated_again,
      ApplyPromptTemplates(single_chunk_again, ContentType::kFirst,
                           session_config, *tokenizer_,
                           /*is_first_turn=*/false));
  ASSERT_EQ(templated_again.size(), 1);
  EXPECT_THAT(std::get<InputText>(templated_again[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("<test>User\nAnother turn"));
}

TEST_F(SessionUtilsTest, ApplyPromptTemplatesWithSingleImageInput) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  // Single image input. Templates are applied to the first and
  // last chunks. In this case, the image input is both the first and last
  // chunks, and the text chunks (templates) will be added before and after
  // the image.
  std::vector<InputData> single_image;
  single_image.emplace_back(InputImage("456"));
  ASSERT_OK_AND_ASSIGN(
      auto templated_image,
      ApplyPromptTemplates(single_image, ContentType::kFirst, session_config,
                           *tokenizer_, /*is_first_turn=*/true));
  ASSERT_EQ(templated_image.size(), 3);
  EXPECT_THAT(std::get<InputText>(templated_image[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s>"));
  EXPECT_THAT(std::get<InputText>(templated_image[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("<test>User\n"));
  EXPECT_TRUE(std::holds_alternative<InputImage>(templated_image[2]));
}

TEST_F(SessionUtilsTest, PreprocessContents) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  std::vector<InputData> contents;
  contents.emplace_back(InputText("</s>Hello World!"));
  std::optional<BenchmarkInfo> benchmark_info;
  ASSERT_OK_AND_ASSIGN(auto preprocessed_contents,
                       PreprocessContents(contents, session_config, *tokenizer_,
                                          benchmark_info));
  ASSERT_EQ(preprocessed_contents.size(), 1);
  ASSERT_TRUE(std::holds_alternative<InputText>(preprocessed_contents[0]));
  const auto& text_data = std::get<InputText>(preprocessed_contents[0]);
  ASSERT_TRUE(text_data.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto text_tensor, text_data.GetPreprocessedTextTensor());
  ASSERT_NE(text_tensor, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto token_ids_span,
                              ReferTensorBufferAsSpan<int>(*text_tensor));
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(2, 90, 547, 58, 735, 210, 466, 2294));
}

TEST_F(SessionUtilsTest, PreprocessContentsMultimodal) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  std::vector<InputData> contents;
  contents.emplace_back(InputText("</s>Hello World!"));

  std::vector<float> dummy_image_data = {0.1f, 0.2f, 0.3f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto image_tensor,
      CopyToTensorBuffer<float>(dummy_image_data, {1, 1, 1, 3}));
  contents.emplace_back(InputImage(std::move(image_tensor)));
  contents.emplace_back(InputImageEnd());

  absl::flat_hash_map<std::string, litert::TensorBuffer> tensor_map;
  std::vector<float> map_data = {0.7f, 0.8f};
  LITERT_ASSERT_OK_AND_ASSIGN(auto map_tensor,
                              CopyToTensorBuffer<float>(map_data, {1, 2}));
  tensor_map["key1"] = std::move(map_tensor);
  contents.emplace_back(InputImage(std::move(tensor_map)));
  contents.emplace_back(InputImageEnd());

  std::vector<float> dummy_audio_data = {0.4f, 0.5f, 0.6f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_tensor,
      CopyToTensorBuffer<float>(dummy_audio_data, {1, 3, 1}));
  contents.emplace_back(InputAudio(std::move(audio_tensor)));
  contents.emplace_back(InputAudioEnd());

  std::optional<BenchmarkInfo> benchmark_info;
  ASSERT_OK_AND_ASSIGN(auto preprocessed_contents,
                       PreprocessContents(contents, session_config, *tokenizer_,
                                          benchmark_info));
  ASSERT_EQ(preprocessed_contents.size(), 7);
  ASSERT_TRUE(std::holds_alternative<InputText>(preprocessed_contents[0]));
  const auto& text_data = std::get<InputText>(preprocessed_contents[0]);
  ASSERT_TRUE(text_data.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto text_tensor, text_data.GetPreprocessedTextTensor());
  ASSERT_NE(text_tensor, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto token_ids_span,
                              ReferTensorBufferAsSpan<int>(*text_tensor));
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(2, 90, 547, 58, 735, 210, 466, 2294));

  ASSERT_TRUE(std::holds_alternative<InputImage>(preprocessed_contents[1]));
  const auto& image_data = std::get<InputImage>(preprocessed_contents[1]);
  ASSERT_TRUE(image_data.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto img_tensor_out,
                       image_data.GetPreprocessedImageTensor());
  LITERT_ASSERT_OK_AND_ASSIGN(auto img_span,
                              ReferTensorBufferAsSpan<float>(*img_tensor_out));
  EXPECT_THAT(std::vector<float>(img_span.begin(), img_span.end()),
              testing::ElementsAre(0.1f, 0.2f, 0.3f));

  ASSERT_TRUE(std::holds_alternative<InputImageEnd>(preprocessed_contents[2]));

  ASSERT_TRUE(std::holds_alternative<InputImage>(preprocessed_contents[3]));
  const auto& image_map_data = std::get<InputImage>(preprocessed_contents[3]);
  ASSERT_TRUE(image_map_data.IsTensorBufferMap());
  ASSERT_OK_AND_ASSIGN(auto img_map_out,
                       image_map_data.GetPreprocessedImageTensorMap());
  ASSERT_TRUE(img_map_out->contains("key1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto map_span, ReferTensorBufferAsSpan<float>(img_map_out->at("key1")));
  EXPECT_THAT(std::vector<float>(map_span.begin(), map_span.end()),
              testing::ElementsAre(0.7f, 0.8f));

  ASSERT_TRUE(std::holds_alternative<InputImageEnd>(preprocessed_contents[4]));

  ASSERT_TRUE(std::holds_alternative<InputAudio>(preprocessed_contents[5]));
  const auto& audio_data = std::get<InputAudio>(preprocessed_contents[5]);
  ASSERT_TRUE(audio_data.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto audio_tensor_out,
                       audio_data.GetPreprocessedAudioTensor());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_span, ReferTensorBufferAsSpan<float>(*audio_tensor_out));
  EXPECT_THAT(std::vector<float>(audio_span.begin(), audio_span.end()),
              testing::ElementsAre(0.4f, 0.5f, 0.6f));

  ASSERT_TRUE(std::holds_alternative<InputAudioEnd>(preprocessed_contents[6]));
}

TEST_F(SessionUtilsTest, PreprocessContentsWithEmptyInputText) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  std::vector<InputData> contents;
  contents.emplace_back(InputText(""));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_contents,
                       PreprocessContents(contents, session_config, *tokenizer_,
                                          /*benchmark_info=*/std::nullopt));
  EXPECT_TRUE(preprocessed_contents.empty());
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensWithText) {
  MockEngine mock_engine;
  EXPECT_CALL(mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // 1. Raw text string
  std::vector<InputData> contents;
  contents.emplace_back(InputText("Hello World!"));
  ASSERT_OK_AND_ASSIGN(int tokens,
                       CalculateInputDataTokens(contents, mock_engine));
  ASSERT_OK_AND_ASSIGN(auto expected_ids,
                       tokenizer_->TextToTokenIds("Hello World!"));
  EXPECT_EQ(tokens, expected_ids.size());

  // 2. Preprocessed text tensor
  std::vector<int> token_ids = {10, 20, 30, 40};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor,
                              CopyToTensorBuffer<int>(token_ids, {1, 4}));
  std::vector<InputData> tensor_contents;
  tensor_contents.emplace_back(InputText(std::move(tensor)));
  ASSERT_OK_AND_ASSIGN(int tensor_tokens,
                       CalculateInputDataTokens(tensor_contents, mock_engine));
  EXPECT_EQ(tensor_tokens, 4);
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensWithVisionNonViT) {
  MockEngine mock_engine;
  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 256;
  vision_props.patch_num_shrink_factor = std::nullopt;

  EXPECT_CALL(mock_engine, GetVisionExecutorProperties())
      .WillOnce(testing::Return(vision_props));

  std::vector<float> dummy_image = {0.1f, 0.2f};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor,
                              CopyToTensorBuffer<float>(dummy_image, {1, 2}));
  std::vector<InputData> contents;
  contents.emplace_back(InputImage(std::move(tensor)));
  contents.emplace_back(InputImageEnd());

  ASSERT_OK_AND_ASSIGN(int tokens,
                       CalculateInputDataTokens(contents, mock_engine));
  // 256 tokens per image + 1 for InputImageEnd = 257.
  EXPECT_EQ(tokens, 257);
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensWithVisionViT) {
  MockEngine mock_engine;
  VisionExecutorProperties vision_props;
  vision_props.num_tokens_per_image = 256;
  vision_props.patch_num_shrink_factor = 9;

  EXPECT_CALL(mock_engine, GetVisionExecutorProperties())
      .WillOnce(testing::Return(vision_props));

  // 2520 patches / shrink 9 = 280 tokens.
  std::vector<int> dummy_positions(2520 * 2, 0);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto pos_tensor, CopyToTensorBuffer<int>(dummy_positions, {1, 2520, 2}));
  absl::flat_hash_map<std::string, TensorBuffer> tensor_map;
  tensor_map["positions_xy"] = std::move(pos_tensor);

  std::vector<InputData> contents;
  contents.emplace_back(InputImage(std::move(tensor_map)));
  contents.emplace_back(InputImageEnd());

  ASSERT_OK_AND_ASSIGN(int tokens,
                       CalculateInputDataTokens(contents, mock_engine));
  // 280 tokens for image patches + 1 for InputImageEnd = 281.
  EXPECT_EQ(tokens, 281);
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensWithStaticAudio) {
  MockEngine mock_engine;
  AudioExecutorProperties audio_props;
  audio_props.is_streaming_model = false;
  audio_props.audio_shrink_factor = 16;

  EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
      .WillOnce(testing::Return(audio_props));

  // seq_len = 512, shrink = 16 => (512 + 16 - 1) / 16 = 32 tokens.
  std::vector<float> dummy_audio(512, 0.0f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 512, 1}));
  std::vector<InputData> contents;
  contents.emplace_back(InputAudio(std::move(audio_tensor)));
  contents.emplace_back(InputAudioEnd());

  ASSERT_OK_AND_ASSIGN(int tokens,
                       CalculateInputDataTokens(contents, mock_engine));
  // 32 audio tokens + 1 for InputAudioEnd = 33.
  EXPECT_EQ(tokens, 33);
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensWithStreamingAudioTinyGemma) {
  // tiny_gemma_audio_19m_40m_streaming_cpu: overlap=3, chunk=51, shrink=16.
  // stride = 51 - 3 = 48, chunk_output_tokens = 48 / 16 = 3.
  AudioExecutorProperties audio_props;
  audio_props.is_streaming_model = true;
  audio_props.streaming_chunk_size = 51;
  audio_props.streaming_chunk_overlap_size = 3;
  audio_props.audio_shrink_factor = 16;
  // Case 1: seq_len = 30 (< window_size = 51).
  // Without InputAudioEnd (flush=false): 0 full chunks => 0 tokens.
  // With InputAudioEnd (flush=true): ceil(30 / 16) = 2 audio tokens + 1 End
  // = 3.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(30, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 30, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 0);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 3);
  }
  // Case 2: seq_len = 51 (1 full chunk).
  // Without InputAudioEnd (flush=false): 1 * 3 = 3 tokens.
  // With InputAudioEnd (flush=true): 3 audio tokens + 1 End = 4 tokens.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(51, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 51, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 3);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 4);
  }
  // Case 3: seq_len = 99 (2 full chunks: 51 + 48).
  // Without InputAudioEnd (flush=false): 2 * 3 = 6 tokens.
  // With InputAudioEnd (flush=true): 6 audio tokens + 1 End = 7 tokens.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(99, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 99, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 6);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 7);
  }
  // Case 4: seq_len = 100 (2 full chunks + 1 trailing frame, last_chunk_len=4).
  // Without InputAudioEnd (flush=false): 2 full chunks => 2 * 3 = 6 tokens.
  // With InputAudioEnd (flush=true): 2 * 3 + ceil(4 / 16) = 7 audio tokens + 1
  // End = 8 tokens.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(100, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 100, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 6);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 8);
  }
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensWithStreamingAudioGemma4) {
  // Gemma4 streaming audio: overlap=3, chunk=51, shrink=4.
  // stride = 51 - 3 = 48, chunk_output_tokens = 48 / 4 = 12.
  AudioExecutorProperties audio_props;
  audio_props.is_streaming_model = true;
  audio_props.streaming_chunk_size = 51;
  audio_props.streaming_chunk_overlap_size = 3;
  audio_props.audio_shrink_factor = 4;
  // Case 1: seq_len = 30 (< window_size = 51).
  // Without InputAudioEnd (flush=false): 0 full chunks => 0 tokens.
  // With InputAudioEnd (flush=true): ceil(30 / 4) = 8 audio tokens + 1 End = 9.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(30, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 30, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 0);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 9);
  }
  // Case 2: seq_len = 51 (1 full chunk).
  // Without InputAudioEnd (flush=false): 1 * 12 = 12 tokens.
  // With InputAudioEnd (flush=true): 12 audio tokens + 1 End = 13 tokens.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(51, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 51, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 12);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 13);
  }
  // Case 3: seq_len = 99 (2 full chunks: 51 + 48).
  // Without InputAudioEnd (flush=false): 2 * 12 = 24 tokens.
  // With InputAudioEnd (flush=true): 24 audio tokens + 1 End = 25 tokens.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(99, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 99, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 24);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 25);
  }
  // Case 4: seq_len = 100 (2 full chunks + 1 trailing frame, last_chunk_len=4).
  // Without InputAudioEnd (flush=false): 2 full chunks => 2 * 12 = 24 tokens.
  // With InputAudioEnd (flush=true): 2 * 12 + ceil(4 / 4) = 25 audio tokens + 1
  // End = 26 tokens.
  {
    MockEngine mock_engine;
    EXPECT_CALL(mock_engine, GetAudioExecutorProperties())
        .WillRepeatedly(testing::Return(audio_props));
    std::vector<float> dummy_audio(100, 0.0f);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 100, 1}));
    std::vector<InputData> contents;
    contents.emplace_back(InputAudio(std::move(audio_tensor)));
    ASSERT_OK_AND_ASSIGN(int tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(tokens, 24);
    contents.emplace_back(InputAudioEnd());
    ASSERT_OK_AND_ASSIGN(int flushed_tokens,
                         CalculateInputDataTokens(contents, mock_engine));
    EXPECT_EQ(flushed_tokens, 26);
  }
}

TEST_F(SessionUtilsTest, CalculateInputDataTokensErrorHandling) {
  MockEngine mock_engine;
  // Text only should NEVER call GetVisionExecutorProperties or
  // GetAudioExecutorProperties.
  EXPECT_CALL(mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));
  EXPECT_CALL(mock_engine, GetVisionExecutorProperties()).Times(0);
  EXPECT_CALL(mock_engine, GetAudioExecutorProperties()).Times(0);

  std::vector<InputData> text_only;
  text_only.emplace_back(InputText("Hello"));
  EXPECT_OK(CalculateInputDataTokens(text_only, mock_engine));

  // Image when GetVisionExecutorProperties returns error:
  MockEngine error_engine;
  EXPECT_CALL(error_engine, GetVisionExecutorProperties())
      .WillOnce(testing::Return(absl::InternalError("Vision not supported")));
  std::vector<float> dummy_image = {0.1f};
  LITERT_ASSERT_OK_AND_ASSIGN(auto img_tensor,
                              CopyToTensorBuffer<float>(dummy_image, {1, 1}));
  std::vector<InputData> img_contents;
  img_contents.emplace_back(InputImage(std::move(img_tensor)));
  EXPECT_THAT(CalculateInputDataTokens(img_contents, error_engine),
              testing::status::StatusIs(absl::StatusCode::kInternal,
                                        "Vision not supported"));

  // Audio when GetAudioExecutorProperties returns error:
  EXPECT_CALL(error_engine, GetAudioExecutorProperties())
      .WillOnce(testing::Return(absl::InternalError("Audio not supported")));
  std::vector<float> dummy_audio = {0.1f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_tensor, CopyToTensorBuffer<float>(dummy_audio, {1, 1, 1}));
  std::vector<InputData> audio_contents;
  audio_contents.emplace_back(InputAudio(std::move(audio_tensor)));
  EXPECT_THAT(CalculateInputDataTokens(audio_contents, error_engine),
              testing::status::StatusIs(absl::StatusCode::kInternal,
                                        "Audio not supported"));
}

}  // namespace
}  // namespace litert::lm
