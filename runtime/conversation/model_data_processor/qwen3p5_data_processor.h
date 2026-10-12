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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN3P5_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN3P5_DATA_PROCESSOR_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/engine/io_types.h"

namespace re2 {
class RE2;
}  // namespace re2

namespace litert::lm {

// Qwen3p5DataProcessor is a ModelDataProcessor for Qwen3.5 models. When tools
// are set, it parses <tool_call><function=NAME><parameter=KEY>VALUE</parameter>
// blocks into tool calls, converting each value to its parameter's schema type.
class Qwen3p5DataProcessor
    : public TypeSafeModelDataProcessor<Qwen3p5DataProcessorConfig,
                                        Qwen3p5DataProcessorArguments> {
 public:
  static absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Create(
      Qwen3p5DataProcessorConfig config,
      std::optional<Preface> preface = std::nullopt);

  ~Qwen3p5DataProcessor() override;

  // Returns the start of tool call blocks.
  absl::string_view CodeFenceStart() const override;

  // Returns the end of tool call blocks.
  absl::string_view CodeFenceEnd() const override;

  // Returns the config of the model data processor.
  const Qwen3p5DataProcessorConfig& GetConfig() const override {
    return config_;
  }

 private:
  explicit Qwen3p5DataProcessor(Qwen3p5DataProcessorConfig config,
                                std::optional<Preface> preface,
                                std::unique_ptr<re2::RE2> fence_regex);

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const Qwen3p5DataProcessorArguments& args) const override;

  Qwen3p5DataProcessorConfig config_;
  std::optional<Preface> preface_;
  // Matches (text) <start> (block) <end>, built from the config in Create().
  std::unique_ptr<re2::RE2> fence_regex_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN3P5_DATA_PROCESSOR_H_
