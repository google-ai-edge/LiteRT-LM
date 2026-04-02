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

#include "runtime/conversation/channel_util.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/engine/io_types.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

namespace {
constexpr absl::string_view kChannelsKey = "channels";
}  // namespace

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ExtractChannelContent(const std::vector<Channel>& channels,
                      Responses& responses) {
  absl::flat_hash_map<std::string, std::string> extracted_fields;
  if (responses.GetTexts().empty()) {
    return extracted_fields;
  }

  if (responses.GetTexts().size() > 1) {
    return absl::InvalidArgumentError(
        "When extracting channel text, responses must not have more than one "
        "text element.");
  }

  if (!responses.GetTexts().empty()) {
    std::string content = responses.GetTexts()[0];
    for (const auto& channel : channels) {
      std::string escaped_start = RE2::QuoteMeta(channel.start);
      std::string escaped_end = RE2::QuoteMeta(channel.end);
      RE2 re("(?s)(.*?)" + escaped_start + "(.*?)(" + escaped_end + "|$)");

      std::string channel_content;
      std::string new_content;
      absl::string_view remaining_content(content);
      std::string text_before;
      std::string text_inside;
      std::string end_match;

      while (RE2::Consume(&remaining_content, re, &text_before, &text_inside,
                          &end_match)) {
        new_content += text_before;
        channel_content += text_inside;
      }
      new_content += std::string(remaining_content);

      if (!channel_content.empty()) {
        content = new_content;
        extracted_fields[channel.channel_name] += channel_content;
      }
    }
    responses.GetMutableTexts()[0] = content;
  }
  return extracted_fields;
}

void InsertChannelContentIntoMessage(
    const absl::flat_hash_map<std::string, std::string>& channel_content,
    Message& assistant_message) {
  for (const auto& [channel_name, value] : channel_content) {
    assistant_message[std::string(kChannelsKey)][channel_name] = value;
  }
}

}  // namespace litert::lm
