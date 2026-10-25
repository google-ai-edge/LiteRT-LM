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
// C++ port of `src/decoders/fuse.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/fuse.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {
namespace decoders {

absl::StatusOr<std::vector<std::string>> Fuse::DecodeChain(
    std::vector<std::string> tokens) const {
  std::string new_string = absl::StrJoin(tokens, "");
  return std::vector<std::string>{new_string};
}

absl::StatusOr<Fuse> Fuse::FromJson(const Json& json) {
  // `type_: MustBe!("Fuse")`: the field is mandatory and must hold exactly
  // this string (monostate error texts).
  RETURN_IF_ERROR(RequireObject(json, "struct Fuse"));
  auto it = json.find("type");
  if (it == json.end()) return MissingFieldError("type");
  if (!it->is_string()) return InvalidTypeError(*it, "string \"Fuse\"");
  if (it->get_ref<const std::string&>() != "Fuse") {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid value: string \"", it->get_ref<const std::string&>(),
        "\", expected string \"Fuse\""));
  }
  return Fuse();
}

Json Fuse::ToJson() const {
  Json json = Json::object();
  json["type"] = "Fuse";
  return json;
}

}  // namespace decoders
}  // namespace tokenizers
