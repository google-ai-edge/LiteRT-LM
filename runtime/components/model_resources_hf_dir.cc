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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::lm {
namespace {

using ::nlohmann::json;

// Upper bound of the default context length (`LlmMetadata::max_num_tokens`):
// checkpoints advertise up to 262,144 positions, and executors may allocate
// decoder state (KV caches) for the configured context up front.
constexpr int kDefaultMaxNumTokens = 4096;

std::string JoinPath(absl::string_view dir, absl::string_view name) {
  if (!dir.empty() && dir.back() == '/') {
    return absl::StrCat(dir, name);
  }
  return absl::StrCat(dir, "/", name);
}

absl::StatusOr<std::string> ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError(absl::StrCat("Cannot open ", path));
  }
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

absl::StatusOr<json> ParseJsonObject(const std::string& path,
                                     const std::string& contents) {
  json parsed =
      json::parse(contents, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (!parsed.is_object()) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, " does not hold a JSON object"));
  }
  return parsed;
}

absl::StatusOr<json> ReadRequiredJsonObject(absl::string_view model_dir,
                                            absl::string_view name) {
  const std::string path = JoinPath(model_dir, name);
  ABSL_ASSIGN_OR_RETURN(const std::string contents, ReadFile(path));
  return ParseJsonObject(path, contents);
}

absl::StatusOr<json> ReadOptionalJsonObject(absl::string_view model_dir,
                                            absl::string_view name) {
  const std::string path = JoinPath(model_dir, name);
  absl::StatusOr<std::string> contents = ReadFile(path);
  if (absl::IsNotFound(contents.status())) {
    return json::object();
  }
  ABSL_RETURN_IF_ERROR(contents.status());
  return ParseJsonObject(path, *contents);
}

absl::StatusOr<std::string> OptionalStringField(const json& obj,
                                                const std::string& key,
                                                absl::string_view source) {
  const auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) {
    return "";
  }
  if (!it->is_string()) {
    return absl::InvalidArgumentError(
        absl::StrCat(source, " field '", key, "' must be a string"));
  }
  return it->get<std::string>();
}

// A token of `tokenizer_config.json`: a string, or an object with the string
// in `content`.
std::optional<std::string> TokenString(const json& config,
                                       const std::string& key) {
  const auto it = config.find(key);
  if (it == config.end()) {
    return std::nullopt;
  }
  const json* value = &*it;
  if (value->is_object()) {
    const auto content = value->find("content");
    if (content == value->end()) {
      return std::nullopt;
    }
    value = &*content;
  }
  if (!value->is_string() || value->get_ref<const std::string&>().empty()) {
    return std::nullopt;
  }
  return value->get<std::string>();
}

// Appends the end-of-sequence token ids of a Hugging Face config to `ids`:
// its `eos_token_id` holds one token id or a list of them.
void AppendEosTokenIds(const json& config, std::vector<int>& ids) {
  const auto it = config.find("eos_token_id");
  if (it == config.end()) {
    return;
  }
  const auto append = [&ids](const json& id) {
    if (!id.is_number_integer()) {
      return;
    }
    const int64_t value = id.get<int64_t>();
    if (value >= 0 && value <= std::numeric_limits<int>::max()) {
      ids.push_back(static_cast<int>(value));
    }
  };
  if (it->is_array()) {
    for (const json& id : *it) {
      append(id);
    }
  } else {
    append(*it);
  }
}

int PositiveInt(const json& config, const std::string& key) {
  const auto it = config.find(key);
  if (it == config.end() || !it->is_number_integer()) {
    return 0;
  }
  const int64_t value = it->get<int64_t>();
  return value > 0 ? static_cast<int>(std::min<int64_t>(
                         value, std::numeric_limits<int>::max()))
                   : 0;
}

// Extracts the raw Jinja chat template string from `chat_template.jinja` (as
// recent `transformers` versions save it) or from `chat_template` in
// `tokenizer_config.json` (either a string or named templates, using
// `"default"`). Stored into `LlmMetadata::jinja_prompt_template`, where
// LiteRT-LM's `PromptTemplate` compiles and renders it.
std::optional<std::string> ChatTemplate(absl::string_view model_dir,
                                        const json& tokenizer_config) {
  if (absl::StatusOr<std::string> jinja =
          ReadFile(JoinPath(model_dir, "chat_template.jinja"));
      jinja.ok()) {
    return *std::move(jinja);
  }
  const auto it = tokenizer_config.find("chat_template");
  if (it == tokenizer_config.end()) {
    return std::nullopt;
  }
  if (it->is_string()) {
    return it->get<std::string>();
  }
  if (it->is_array()) {
    for (const json& entry : *it) {
      if (!entry.is_object()) {
        continue;
      }
      const auto name = entry.find("name");
      const auto chat_template = entry.find("template");
      if (name != entry.end() && *name == "default" &&
          chat_template != entry.end() && chat_template->is_string()) {
        return chat_template->get<std::string>();
      }
    }
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ModelResourcesHfDir>>
ModelResourcesHfDir::Create(const ModelAssets& model_assets) {
  ABSL_ASSIGN_OR_RETURN(const absl::string_view path, model_assets.GetPath());
  if (path.empty()) {
    return absl::InvalidArgumentError(
        "ModelResourcesHfDir requires a non-empty model directory path.");
  }
  if (!model_assets.IsDirectory()) {
    return absl::InvalidArgumentError(
        absl::StrCat("ModelResourcesHfDir requires a directory path: ", path));
  }

  ABSL_ASSIGN_OR_RETURN(const json config,
                        ReadRequiredJsonObject(path, "config.json"));
  ABSL_ASSIGN_OR_RETURN(const json tokenizer_config,
                        ReadOptionalJsonObject(path, "tokenizer_config.json"));
  ABSL_ASSIGN_OR_RETURN(const json generation_config,
                        ReadOptionalJsonObject(path, "generation_config.json"));

  // Multimodal checkpoints keep the text decoder's settings in `text_config`.
  json text_config = json::object();
  if (const auto it = config.find("text_config");
      it != config.end() && it->is_object()) {
    text_config = *it;
  }

  ABSL_ASSIGN_OR_RETURN(
      std::string model_type,
      OptionalStringField(config, "model_type", "config.json"));
  if (model_type.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        model_type,
        OptionalStringField(text_config, "model_type",
                            "config.json.text_config"));
  }

  proto::LlmMetadata metadata;
  const auto add_bos_it = tokenizer_config.find("add_bos_token");
  const bool add_bos_token = add_bos_it == tokenizer_config.end() ||
                             !add_bos_it->is_boolean() ||
                             add_bos_it->get<bool>();
  if (add_bos_token) {
    if (std::optional<std::string> bos =
            TokenString(tokenizer_config, "bos_token")) {
      metadata.mutable_start_token()->set_token_str(*std::move(bos));
    }
  }

  std::vector<int> eos_ids;
  AppendEosTokenIds(generation_config, eos_ids);
  AppendEosTokenIds(config, eos_ids);
  AppendEosTokenIds(text_config, eos_ids);
  absl::flat_hash_set<int> seen_eos_ids;
  for (const int id : eos_ids) {
    if (seen_eos_ids.insert(id).second) {
      metadata.add_stop_tokens()->mutable_token_ids()->add_ids(id);
    }
  }
  if (std::optional<std::string> eos =
          TokenString(tokenizer_config, "eos_token")) {
    metadata.add_stop_tokens()->set_token_str(*std::move(eos));
  }

  const int max_positions =
      std::max(PositiveInt(config, "max_position_embeddings"),
               PositiveInt(text_config, "max_position_embeddings"));
  const int max_num_tokens =
      max_positions > 0 ? std::min(max_positions, kDefaultMaxNumTokens)
                        : kDefaultMaxNumTokens;
  metadata.set_max_num_tokens(max_num_tokens);

  if (std::optional<std::string> chat_template =
          ChatTemplate(path, tokenizer_config)) {
    metadata.set_jinja_prompt_template(*std::move(chat_template));
  }
  metadata.mutable_llm_model_type()->mutable_generic_model();

  return absl::WrapUnique(new ModelResourcesHfDir(
      std::string(path), std::move(model_type), std::move(metadata)));
}

ModelResourcesHfDir::ModelResourcesHfDir(std::string model_dir,
                                         std::string model_type,
                                         proto::LlmMetadata llm_metadata)
    : model_dir_(std::move(model_dir)),
      model_type_(std::move(model_type)),
      llm_metadata_(std::move(llm_metadata)) {}

absl::StatusOr<std::unique_ptr<Tokenizer>> ModelResourcesHfDir::GetTokenizer() {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                        litert::support::HuggingFaceTokenizer::CreateFromFile(
                            JoinPath(model_dir_, "tokenizer.json")));
  return tokenizer;
}

}  // namespace litert::lm
