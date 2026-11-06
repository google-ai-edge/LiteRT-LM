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
//
// C++ port of the `#[cfg(test)]` module of `src/processors/mod.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0, Copyright
// 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/processors/processors.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/processors/bert.h"
#include "support/tokenizer/tokenizers/processors/roberta.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace processors {
namespace {

TEST(ProcessorsTest, DeserializeBertRobertaCorrectly) {
  RobertaProcessing roberta;
  const std::string roberta_r =
      R"({"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true})";
  EXPECT_EQ(DumpJson(roberta.ToJson()), roberta_r);
  ASSERT_OK_AND_ASSIGN(Json roberta_json, ParseJson(roberta_r));
  ASSERT_OK_AND_ASSIGN(PostProcessorWrapper roberta_wrapper,
                       PostProcessorWrapper::FromJson(roberta_json));
  EXPECT_EQ(roberta_wrapper, PostProcessorWrapper(roberta));

  BertProcessing bert;
  const std::string bert_r =
      R"({"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]})";
  EXPECT_EQ(DumpJson(bert.ToJson()), bert_r);
  ASSERT_OK_AND_ASSIGN(Json bert_json, ParseJson(bert_r));
  ASSERT_OK_AND_ASSIGN(PostProcessorWrapper bert_wrapper,
                       PostProcessorWrapper::FromJson(bert_json));
  EXPECT_EQ(bert_wrapper, PostProcessorWrapper(bert));
}

TEST(ProcessorsTest, PostProcessorDeserializationNoType) {
  {
    const std::string json =
        R"({"add_prefix_space": true, "trim_offsets": false, "use_regex": false})";
    ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(json));
    absl::StatusOr<PostProcessorWrapper> reconstructed =
        PostProcessorWrapper::FromJson(parsed);
    ASSERT_FALSE(reconstructed.ok());
    EXPECT_EQ(
        reconstructed.status().message(),
        "data did not match any variant of untagged enum PostProcessorWrapper");
  }

  {
    const std::string json = R"({"sep":["[SEP]",102],"cls":["[CLS]",101]})";
    ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(json));
    ASSERT_OK_AND_ASSIGN(PostProcessorWrapper reconstructed,
                         PostProcessorWrapper::FromJson(parsed));
    EXPECT_TRUE(reconstructed.Is<BertProcessing>());
  }

  {
    const std::string json =
        R"({"sep":["</s>",2], "cls":["<s>",0], "trim_offsets":true, "add_prefix_space":true})";
    ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(json));
    ASSERT_OK_AND_ASSIGN(PostProcessorWrapper reconstructed,
                         PostProcessorWrapper::FromJson(parsed));
    EXPECT_TRUE(reconstructed.Is<RobertaProcessing>());
  }

  {
    const std::string json =
        R"({"type":"RobertaProcessing", "sep":["</s>",2] })";
    ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(json));
    absl::StatusOr<PostProcessorWrapper> reconstructed =
        PostProcessorWrapper::FromJson(parsed);
    ASSERT_FALSE(reconstructed.ok());
    EXPECT_EQ(
        reconstructed.status().message(),
        "data did not match any variant of untagged enum PostProcessorWrapper");
  }
}

}  // namespace
}  // namespace processors
}  // namespace tokenizers
