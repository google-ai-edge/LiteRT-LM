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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_HF_DIR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_HF_DIR_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

// `ModelResources` implementation for unconverted Hugging Face checkpoint
// directories (`config.json` plus optional `*.safetensors`, `tokenizer.json`,
// `tokenizer_config.json`, `generation_config.json`, and
// `chat_template.jinja`). Single-file model formats (such as `.gguf` or `.sbs`)
// instead register via `ModelResourcesRegistry` by file extension and magic
// bytes.
//
// Only checkpoint metadata (including raw Jinja chat template text populated
// into `LlmMetadata::jinja_prompt_template` for `PromptTemplate` to render) and
// the tokenizer are loaded here; executor backends validate `model_type()` and
// load the model weights inside their `Create` factory methods.
class ModelResourcesHfDir : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<ModelResourcesHfDir>> Create(
      const ModelAssets& model_assets);

  ~ModelResourcesHfDir() override = default;

  using ModelResources::GetTokenizer;

  // Loads `tokenizer.json` from `model_dir()`.
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override;

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return &llm_metadata_;
  }

  absl::string_view model_dir() const { return model_dir_; }

  // Value of `model_type` in `config.json` (or `text_config.model_type`), or
  // empty if `config.json` does not declare one.
  absl::string_view model_type() const { return model_type_; }

 private:
  ModelResourcesHfDir(std::string model_dir, std::string model_type,
                      proto::LlmMetadata llm_metadata);

  std::string model_dir_;
  std::string model_type_;
  proto::LlmMetadata llm_metadata_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_HF_DIR_H_
