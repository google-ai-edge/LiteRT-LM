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

#include <cstddef>
#include <optional>
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
                      Responses& responses,
                      const std::optional<std::string>& open_channel_name) {
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

      bool matched = false;
      if (open_channel_name.has_value() &&
          *open_channel_name == channel.channel_name) {
        RE2 first_re("(?s)(.*?)(" + escaped_end + "|$)");
        if (RE2::Consume(&remaining_content, first_re, &text_inside,
                         &end_match)) {
          channel_content += text_inside;
          matched = true;
        }
      }

      while (RE2::Consume(&remaining_content, re, &text_before, &text_inside,
                          &end_match)) {
        new_content += text_before;
        channel_content += text_inside;
        matched = true;
      }
      new_content += std::string(remaining_content);

      if (matched) {
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
    assistant_message.channels[channel_name] = value;
  }
}

std::optional<std::string> GetOpenChannelName(
    absl::string_view text, const std::vector<Channel>& channels) {
  std::optional<std::string> open_channel_name;
  size_t max_start_pos = std::string::npos;

  for (const auto& channel : channels) {
    if (channel.start.empty()) {
      continue;
    }

    size_t last_start = text.rfind(channel.start);
    if (last_start == absl::string_view::npos) {
      continue;
    }

    size_t last_end =
        channel.end.empty() ? absl::string_view::npos : text.rfind(channel.end);

    bool is_open =
        (last_end == absl::string_view::npos) || (last_start > last_end);

    if (is_open) {
      if (max_start_pos == std::string::npos || last_start > max_start_pos) {
        max_start_pos = last_start;
        open_channel_name = channel.channel_name;
      }
    }
  }

  return open_channel_name;
}

}  // namespace litert::lm
