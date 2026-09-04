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

#include "runtime/conversation/model_data_processor/data_utils.h"

#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/memory_mapped_file.h"

namespace litert::lm {

using ::nlohmann::ordered_json;

absl::StatusOr<std::unique_ptr<MemoryMappedFile>> LoadItemData(
    const ordered_json& item) {
  if (!item.contains("type")) {
    // If `item` doesn't contain a type, it won't be loaded.
    return nullptr;
  }
  if (item["type"] == "text") {
    return InMemoryFile::Create(item["text"]);
  } else if (item["type"] == "image" || item["type"] == "audio") {
    if (item.contains("path")) {
      return MemoryMappedFile::Create(item["path"].get<std::string>());
    }
    if (item.contains("blob")) {
      std::string blob_b64 = item["blob"].get<std::string>();
      std::string blob;
      if (!absl::Base64Unescape(blob_b64, &blob) &&
          !absl::WebSafeBase64Unescape(blob_b64, &blob)) {
        return absl::InvalidArgumentError("Failed to decode base64 blob.");
      }
      return InMemoryFile::Create(blob);
    }
    return absl::InvalidArgumentError(
        "Audio or image item must contain a path or blob.");
  } else if (item["type"] == "tool_response") {
    return nullptr;
  }
  return absl::UnimplementedError("Unsupported item type: " +
                                  item["type"].get<std::string>());
}

}  // namespace litert::lm
