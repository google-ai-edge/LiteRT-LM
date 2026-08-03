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
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "support/tokenizer/sentencepiece_tokenizer.h"  // from @litert
#include "support/tokenizer/tokenizer.h"  // from @litert
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

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
    ASSIGN_OR_RETURN(auto tokenizer,
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
          ASSIGN_OR_RETURN(
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
      ASSIGN_OR_RETURN(auto text_ids, tokenizer_->TextToTokenIds(text));
      token_ids.insert(token_ids.end(), text_ids.begin(), text_ids.end());
    }
    return token_ids;
  }

  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override {
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

}  // namespace
}  // namespace litert::lm
