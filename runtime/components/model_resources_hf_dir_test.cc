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

#include "runtime/components/model_resources_hf_dir.h"

#include <filesystem>  // NOLINT: Required for test directory setup.
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::filesystem operations.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT
#ifdef ENABLE_HUGGINGFACE_TOKENIZER
#include "support/tokenizer/tokenizer.h"
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

std::string MakeTempDir(absl::string_view name) {
  const auto dir = std::filesystem::path(::testing::TempDir()) /
                   std::string("hf_dir_test_") / std::string(name);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  ABSL_CHECK(!ec) << "Cannot create " << dir.string();
  return dir.string();
}

void WriteFile(absl::string_view dir, absl::string_view name,
               absl::string_view contents) {
  const std::string path = absl::StrCat(dir, "/", name);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ABSL_CHECK(file.is_open()) << "Cannot write " << path;
  file << contents;
}

std::string WordLevelTokenizerJson(int vocab_size) {
  constexpr absl::string_view kSpecialTokens[] = {"<unk>", "<|im_start|>",
                                                  "<|im_end|>"};
  std::string added_tokens;
  std::string vocab;
  for (int id = 0; id < 3; ++id) {
    absl::StrAppend(
        &added_tokens, id == 0 ? "" : ",", R"({"id": )", id,
        R"(, "content": ")", kSpecialTokens[id],
        R"(", "single_word": false, "lstrip": false, )",
        R"("rstrip": false, "normalized": false, "special": true})");
    absl::StrAppend(&vocab, id == 0 ? "" : ",", "\"", kSpecialTokens[id],
                    "\": ", id);
  }
  for (int id = 3; id < vocab_size; ++id) {
    absl::StrAppend(&vocab, ",\"t", id, "\": ", id);
  }
  return absl::StrCat(
      R"({"version": "1.0", "truncation": null, "padding": null, )",
      R"("added_tokens": [)", added_tokens, R"(], "normalizer": null, )",
      R"("pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, )",
      R"("decoder": null, "model": {"type": "WordLevel", "vocab": {)", vocab,
      R"(}, "unk_token": "<unk>"}})");
}

std::string WriteModelDir(absl::string_view name,
                          absl::string_view config_fields = "") {
  const std::string dir = MakeTempDir(name);
  WriteFile(
      dir, "config.json",
      absl::StrCat(R"({"model_type": "custom_hf_model")",
                   config_fields.empty() ? "" : ", ", config_fields, "}"));
  return dir;
}

absl::StatusOr<std::unique_ptr<ModelResourcesHfDir>> CreateResources(
    absl::string_view model_dir) {
  ABSL_ASSIGN_OR_RETURN(ModelAssets model_assets,
                        ModelAssets::Create(model_dir));
  return ModelResourcesHfDir::Create(model_assets);
}

TEST(ModelResourcesHfDirTest, ReadsMetadataOfACheckpoint) {
  const std::string model_dir = WriteModelDir(
      "checkpoint",
      R"("text_config": {"eos_token_id": 5, "max_position_embeddings": 1024})");
  WriteFile(model_dir, "tokenizer_config.json", R"({
    "bos_token": {"content": "<s>", "special": true},
    "eos_token": "</s>",
    "chat_template": "{{ messages }}"
  })");
  WriteFile(model_dir, "generation_config.json", R"({"eos_token_id": [6, 5]})");

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> resources,
                       CreateResources(model_dir));
  EXPECT_EQ(resources->model_dir(), model_dir);
  EXPECT_EQ(resources->model_type(), "custom_hf_model");
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* metadata,
                       resources->GetLlmMetadata());
  EXPECT_EQ(metadata->start_token().token_str(), "<s>");
  ASSERT_EQ(metadata->stop_tokens_size(), 3);
  EXPECT_THAT(metadata->stop_tokens(0).token_ids().ids(), ElementsAre(6));
  EXPECT_THAT(metadata->stop_tokens(1).token_ids().ids(), ElementsAre(5));
  EXPECT_EQ(metadata->stop_tokens(2).token_str(), "</s>");
  EXPECT_EQ(metadata->max_num_tokens(), 1024);
  EXPECT_EQ(metadata->jinja_prompt_template(), "{{ messages }}");
  EXPECT_TRUE(metadata->llm_model_type().has_generic_model());

  const std::string no_bos_dir = WriteModelDir("no_bos");
  WriteFile(no_bos_dir, "tokenizer_config.json", R"({
    "bos_token": "<|endoftext|>",
    "add_bos_token": false
  })");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> no_bos_resources,
                       CreateResources(no_bos_dir));
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* no_bos_metadata,
                       no_bos_resources->GetLlmMetadata());
  EXPECT_FALSE(no_bos_metadata->has_start_token());
}

TEST(ModelResourcesHfDirTest, PrefersChatTemplateFileAndSupportsNamedDefault) {
  const std::string model_dir = WriteModelDir("chat_template_file");
  WriteFile(model_dir, "tokenizer_config.json",
            R"({"chat_template": "from tokenizer_config"})");
  WriteFile(model_dir, "chat_template.jinja", "from chat_template.jinja");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> resources,
                       CreateResources(model_dir));
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* metadata,
                       resources->GetLlmMetadata());
  EXPECT_EQ(metadata->jinja_prompt_template(), "from chat_template.jinja");

  const std::string named_dir = WriteModelDir("named_chat_templates");
  WriteFile(named_dir, "tokenizer_config.json", R"({"chat_template": [
    {"name": "tool_use", "template": "tools"},
    {"name": "default", "template": "chat"}
  ]})");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> named_resources,
                       CreateResources(named_dir));
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* named_metadata,
                       named_resources->GetLlmMetadata());
  EXPECT_EQ(named_metadata->jinja_prompt_template(), "chat");
}

TEST(ModelResourcesHfDirTest, ReadsNestedOrMissingModelTypeWithoutBackendDeps) {
  const std::string nested_dir = MakeTempDir("nested_model_type");
  WriteFile(nested_dir, "config.json",
            R"({"text_config": {"model_type": "qwen3_5_text"}, )"
            R"("eos_token_id": 2})");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> nested,
                       CreateResources(nested_dir));
  EXPECT_EQ(nested->model_type(), "qwen3_5_text");
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* metadata,
                       nested->GetLlmMetadata());
  EXPECT_FALSE(metadata->has_start_token());
  ASSERT_EQ(metadata->stop_tokens_size(), 1);
  EXPECT_THAT(metadata->stop_tokens(0).token_ids().ids(), ElementsAre(2));
  EXPECT_EQ(metadata->max_num_tokens(), 4096);
  EXPECT_FALSE(metadata->has_jinja_prompt_template());

  const std::string untyped_dir = MakeTempDir("untyped");
  WriteFile(untyped_dir, "config.json",
            R"({"vocab_size": 32, "max_position_embeddings": 131072})");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> untyped,
                       CreateResources(untyped_dir));
  EXPECT_EQ(untyped->model_type(), "");
  ASSERT_OK_AND_ASSIGN(const proto::LlmMetadata* untyped_metadata,
                       untyped->GetLlmMetadata());
  EXPECT_EQ(untyped_metadata->max_num_tokens(), 4096);
}

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
TEST(ModelResourcesHfDirTest, LoadsHuggingFaceTokenizer) {
  const std::string model_dir = WriteModelDir("tokenizer");
  WriteFile(model_dir, "tokenizer.json", WordLevelTokenizerJson(16));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModelResourcesHfDir> resources,
                       CreateResources(model_dir));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<litert::support::Tokenizer> tokenizer,
                       resources->GetTokenizer());
  EXPECT_THAT(tokenizer->TextToTokenIds("t3 t4 t15"),
              IsOkAndHolds(ElementsAre(3, 4, 15)));
  EXPECT_THAT(tokenizer->TokenToId("<|im_end|>"), IsOkAndHolds(2));
  EXPECT_EQ(tokenizer->GetVocabSize(), 16);
}
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

TEST(ModelResourcesHfDirTest, RejectsMissingConfigNonDirectoryOrMalformedJson) {
  const std::string empty_dir = MakeTempDir("no_config");
  EXPECT_THAT(CreateResources(empty_dir),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(CreateResources(absl::StrCat(empty_dir, "/does_not_exist")),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const std::string model_dir = WriteModelDir("not_a_dir");
  const std::string file_path = absl::StrCat(model_dir, "/config.json");
  EXPECT_THAT(CreateResources(file_path),
              StatusIs(absl::StatusCode::kInvalidArgument));

  const std::string malformed_dir = WriteModelDir("malformed");
  WriteFile(malformed_dir, "tokenizer_config.json", "{");
  EXPECT_THAT(CreateResources(malformed_dir),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
