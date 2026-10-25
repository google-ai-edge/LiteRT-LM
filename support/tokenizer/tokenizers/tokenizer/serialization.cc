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
// C++ port of `src/tokenizer/serialization.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/serialization.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/decoders.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"
#include "support/tokenizer/tokenizers/processors/processors.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/padding.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/truncation.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {

namespace {

// Deserializes an `Option<T>`: `null` (or a missing field, handled by the
// caller) is `None`.
template <typename T>
absl::StatusOr<std::optional<T>> OptionFromJson(const Json& json) {
  if (json.is_null()) return std::optional<T>(std::nullopt);
  ASSIGN_OR_RETURN(T value, T::FromJson(json));
  return std::optional<T>(std::move(value));
}

// Serializes an `Option<T>`: `None` is `null`.
template <typename T>
Json OptionToJson(const std::optional<T>& value) {
  if (!value.has_value()) return Json(nullptr);
  return value->ToJson();
}

}  // namespace

// `impl<M, N, PT, PP, D> Serialize for TokenizerImpl<M, N, PT, PP, D>`
Json Tokenizer::ToJson() const {
  // `serializer.serialize_struct("Tokenizer", 9)`
  Json tokenizer = Json::object();

  // Start by adding the current version
  tokenizer["version"] = std::string(kSerializationVersion);

  // Params
  tokenizer["truncation"] = OptionToJson(truncation_);
  tokenizer["padding"] = OptionToJson(padding_);

  // Added tokens
  tokenizer["added_tokens"] = added_vocabulary_.ToJson();

  // Then add our parts
  tokenizer["normalizer"] = OptionToJson(normalizer_);
  tokenizer["pre_tokenizer"] = OptionToJson(pre_tokenizer_);
  tokenizer["post_processor"] = OptionToJson(post_processor_);
  tokenizer["decoder"] = OptionToJson(decoder_);
  tokenizer["model"] = model_.ToJson();

  return tokenizer;
}

// `impl<'de, M, N, PT, PP, D> Deserialize<'de> for TokenizerImpl<M, N, PT, PP,
// D>` + `impl Visitor for TokenizerVisitor`
absl::StatusOr<Tokenizer> Tokenizer::FromJson(const Json& json) {
  // `deserializer.deserialize_struct("Tokenizer", &[...], TokenizerVisitor)`:
  // anything but a map is rejected by `expecting()` → "struct Tokenizer".
  if (!json.is_object()) {
    return InvalidTypeError(json, "struct Tokenizer");
  }

  // `fn visit_map<V>(self, mut map: V)`
  TokenizerBuilder builder;
  std::vector<AddedTokenWithId> tokens;
  for (const auto& [key, value] : json.items()) {
    if (key == "version") {
      // `let v: String = map.next_value()?;`
      if (!value.is_string()) return InvalidTypeError(value, "a string");
      const std::string v = value.get<std::string>();
      if (v != "1.0") {
        return absl::InvalidArgumentError(
            absl::StrCat("Unknown tokenizer version '", v, "'"));
      }
    } else if (key == "truncation") {
      ASSIGN_OR_RETURN(std::optional<TruncationParams> truncation,
                       OptionFromJson<TruncationParams>(value));
      builder.WithTruncation(std::move(truncation));
    } else if (key == "padding") {
      ASSIGN_OR_RETURN(std::optional<PaddingParams> padding,
                       OptionFromJson<PaddingParams>(value));
      builder.WithPadding(std::move(padding));
    } else if (key == "added_tokens") {
      // `tokens = map.next_value()?;` — `Vec<AddedTokenWithId>`
      if (!value.is_array()) return InvalidTypeError(value, "a sequence");
      tokens.clear();
      tokens.reserve(value.size());
      for (const Json& item : value) {
        ASSIGN_OR_RETURN(AddedTokenWithId token,
                         AddedTokenWithId::FromJson(item));
        tokens.push_back(std::move(token));
      }
    } else if (key == "normalizer") {
      ASSIGN_OR_RETURN(std::optional<normalizers::NormalizerWrapper> normalizer,
                       OptionFromJson<normalizers::NormalizerWrapper>(value));
      builder.WithNormalizer(std::move(normalizer));
    } else if (key == "pre_tokenizer") {
      ASSIGN_OR_RETURN(
          std::optional<pre_tokenizers::PreTokenizerWrapper> pre_tokenizer,
          OptionFromJson<pre_tokenizers::PreTokenizerWrapper>(value));
      builder.WithPreTokenizer(std::move(pre_tokenizer));
    } else if (key == "model") {
      ASSIGN_OR_RETURN(models::ModelWrapper model,
                       models::ModelWrapper::FromJson(value));
      builder.WithModel(std::move(model));
    } else if (key == "decoder") {
      ASSIGN_OR_RETURN(std::optional<decoders::DecoderWrapper> decoder,
                       OptionFromJson<decoders::DecoderWrapper>(value));
      builder.WithDecoder(std::move(decoder));
    } else if (key == "post_processor") {
      ASSIGN_OR_RETURN(
          std::optional<processors::PostProcessorWrapper> post_processor,
          OptionFromJson<processors::PostProcessorWrapper>(value));
      builder.WithPostProcessor(std::move(post_processor));
    } else {
      // PORT NOTE: the Rust visitor has `_ => {}` here but never consumes the
      // value, so serde_json fails on the next key with "expected `,` or `}`":
      // an unknown top-level field is a hard error in the original too. The
      // message is replaced by serde's regular unknown-field text.
      return UnknownFieldError(key, kTokenizerFields);
    }
  }
  ASSIGN_OR_RETURN(Tokenizer tokenizer, builder.Build());

  // Single-pass: warn on ID mismatches, then add all tokens.
  // `add_tokens` computes normalization internally for tokens with
  // `normalized = true`.
  for (const AddedTokenWithId& t : tokens) {
    std::optional<uint32_t> rid = tokenizer.TokenToId(t.token.content);
    if (rid.has_value() && *rid != t.id) {
      ABSL_LOG(WARNING) << "Warning: Token '" << t.token.content
                        << "' was expected to have ID '" << t.id
                        << "' but was given ID '" << *rid << "'";
    }
  }
  std::vector<AddedToken> added_tokens;
  added_tokens.reserve(tokens.size());
  for (AddedTokenWithId& t : tokens) {
    added_tokens.push_back(std::move(t.token));
  }
  RETURN_IF_ERROR(tokenizer.AddTokens(std::move(added_tokens)).status());

  return tokenizer;
}

}  // namespace tokenizers
