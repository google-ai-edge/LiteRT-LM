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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

TEST(ConversationLiteTest, PromptTemplateIsNoopInLiteTarget) {
  PromptTemplate tmpl("{{ message }}");
  PromptTemplateInput input;
  EXPECT_THAT(tmpl.Apply(input), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ConversationLiteTest, HuggingFaceTokenizerIsNoopInLiteTarget) {
  EXPECT_THAT(litert::support::HuggingFaceTokenizer::CreateFromJson("{}"),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ConversationLiteTest, ToolUseParserLiteReturnsErrorWithoutParsers) {
  EXPECT_THAT(
      ParseTextAndToolCalls("```tool_code\nfoo(x=1)\n```", "```tool_code\n",
                            "\n```", SyntaxType::kPython),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ConversationLiteTest, OnlyGemmaAndGenericProcessorsAreLinkedInLiteTarget) {
  EXPECT_OK(CreateModelDataProcessor(Gemma3DataProcessorConfig()));
  EXPECT_OK(CreateModelDataProcessor(Gemma4DataProcessorConfig()));
  EXPECT_THAT(CreateModelDataProcessor(Qwen3DataProcessorConfig()),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(CreateModelDataProcessor(FastVlmDataProcessorConfig()),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ConversationLiteTest,
     CreateDataProcessorConfigFromLlmModelTypeInLiteTarget) {
  proto::LlmModelType gemma3_type;
  gemma3_type.mutable_gemma3();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(gemma3_type));

  proto::LlmModelType gemma3n_type;
  gemma3n_type.mutable_gemma3n();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(gemma3n_type));

  proto::LlmModelType gemma4_type;
  gemma4_type.mutable_gemma4();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(gemma4_type));

  proto::LlmModelType generic_type;
  generic_type.mutable_generic_model();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(generic_type));

  proto::LlmModelType qwen3_type;
  qwen3_type.mutable_qwen3();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(qwen3_type),
              StatusIs(absl::StatusCode::kUnimplemented));

  proto::LlmModelType fastvlm_type;
  fastvlm_type.mutable_fast_vlm();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(fastvlm_type),
              StatusIs(absl::StatusCode::kUnimplemented));

  proto::LlmModelType lfm2_type;
  lfm2_type.mutable_lfm2();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(lfm2_type),
              StatusIs(absl::StatusCode::kUnimplemented));

  proto::LlmModelType function_gemma_type;
  function_gemma_type.mutable_function_gemma();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(function_gemma_type),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace litert::lm
