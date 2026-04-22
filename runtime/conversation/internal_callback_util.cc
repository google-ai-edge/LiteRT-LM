// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may
// may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/internal_callback_util.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/conversation/channel_util.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {
namespace {

// Returns the number of overlapping characters between the suffix of string
// `a` and the prefix of string `b`.
size_t SuffixPrefixOverlap(absl::string_view a, absl::string_view b) {
  if (a.empty() || b.empty()) {
    return 0;
  }

  size_t max_overlap = std::min(a.length(), b.length());

  for (size_t len = max_overlap; len > 0; --len) {
    if (a.substr(a.length() - len) == b.substr(0, len)) {
      return len;
    }
  }

  return 0;
};

// Sends standard text and tool calls to the user callback. The text and/or tool
// calls are first wrapped in a Message object via
// `model_data_processor.ToMessage` before being sent.
void SendMessage(
    absl::AnyInvocable<void(absl::StatusOr<Message>)>& user_callback,
    absl::string_view text, const ModelDataProcessor& model_data_processor,
    DataProcessorArguments processor_args) {
  if (text.empty()) {
    return;
  }
  auto message = model_data_processor.ToMessage(
      Responses(TaskState::kProcessing, {std::string(text)}), processor_args);
  if (!message.ok()) {
    user_callback(message.status());
    return;
  }
  user_callback(std::move(message.value()));
}

// Sends streamed text associated with a specific channel. It wraps the text in
// a Message under the provided `target_channel_name` with a role of "assistant"
// and bypasses `model_data_processor.ToMessage` formatting.
void SendMessageToChannel(
    absl::AnyInvocable<void(absl::StatusOr<Message>)>& user_callback,
    absl::string_view text, absl::string_view channel_name) {
  if (text.empty()) {
    return;
  }
  Message message;
  message["role"] = "assistant";
  message["channels"] = nlohmann::ordered_json::object();
  message["channels"][std::string(channel_name)] = std::string(text);
  user_callback(std::move(message));
}

// Sends remaining un-flushed text at the end of generation and then invokes
// `complete_message_callback` on the final message.
void SendCompleteMessage(
    absl::AnyInvocable<void(absl::StatusOr<Message>)>& user_callback,
    absl::string_view accumulated_response_text,
    const ModelDataProcessor& model_data_processor,
    DataProcessorArguments processor_args, int cursor,
    absl::AnyInvocable<void(Message)>& complete_message_callback,
    const std::string& active_channel_name,
    const std::vector<Channel>& channels,
    const std::optional<std::string>& initial_open_channel_name =
        std::nullopt) {
  // Send remaining un-flushed text at the end of generation.
  if (cursor < accumulated_response_text.size()) {
    if (!active_channel_name.empty()) {
      SendMessageToChannel(user_callback,
                           accumulated_response_text.substr(cursor),
                           active_channel_name);
    } else {
      SendMessage(user_callback, accumulated_response_text.substr(cursor),
                  model_data_processor, processor_args);
    }
  }

  // Wrap the accumulated response text in a `Responses` object.
  Responses responses(TaskState::kProcessing,
                      {std::string(accumulated_response_text)});

  // Exclude tool call channel (indicated by empty channel_name) from extraction
  // so it gets properly parsed by model_data_processor.ToMessage.
  std::vector<Channel> custom_channels;
  for (const auto& channel : channels) {
    if (!channel.channel_name.empty()) {
      custom_channels.push_back(channel);
    }
  }

  // Extract channel content from the responses. Modifies responses in place.
  auto extracted_channels = ExtractChannelContent(custom_channels, responses,
                                                  initial_open_channel_name);
  if (!extracted_channels.ok()) {
    user_callback(extracted_channels.status());
    return;
  }
  auto complete_message =
      model_data_processor.ToMessage(responses, processor_args);
  if (!complete_message.ok()) {
    user_callback(complete_message.status());
    return;
  }
  InsertChannelContentIntoMessage(*extracted_channels, *complete_message);
  if (complete_message_callback) {
    complete_message_callback(*complete_message);
  }
  user_callback(Message());
}

// Returns the complete list of channels the parser should search for, including
// any tool call code blocks (treated as a special channel with no target
// message field) and custom channels passed in by the user config.
std::vector<Channel> GetChannels(const ModelDataProcessor& model_data_processor,
                                 const std::vector<Channel>& custom_channels) {
  std::vector<Channel> channels;
  // Add the tool call channel if the code fence start is not empty.
  if (!model_data_processor.CodeFenceStart().empty()) {
    channels.push_back({"", std::string(model_data_processor.CodeFenceStart()),
                        std::string(model_data_processor.CodeFenceEnd())});
  }
  // Add the custom channels.
  for (const auto& channel : custom_channels) {
    if (!channel.start.empty()) {
      channels.push_back({channel.channel_name, channel.start, channel.end});
    }
  }
  return channels;
}

// Searches the provided text string starting at `cursor` for the earliest
// matching channel start delimiter out of the possible channels provided.
//
// Returns a pointer to the matching channel and mutates `best_start_pos` to
// store its index in the accumulated response text.
//
// Returns nullptr if no channel start delimiter is found.
const Channel* FindNextChannelStart(
    const std::vector<Channel>& possible_channels, absl::string_view text,
    size_t cursor, size_t& best_start_pos) {
  best_start_pos = std::string::npos;
  const Channel* best_match = nullptr;

  for (const auto& channel : possible_channels) {
    size_t start_pos = text.find(channel.start, cursor);
    if (start_pos != std::string::npos) {
      if (best_start_pos == std::string::npos || start_pos < best_start_pos) {
        best_start_pos = start_pos;
        best_match = &channel;
      }
    }
  }
  return best_match;
}

// Checks if the end of the un-parsed string might potentially be the first part
// of any channel start delimiter. Returns the maximum character substring
// overlap length between the end of the response string and the start of any
// active channels.
size_t FindMaxOverlap(const std::vector<Channel>& channels,
                      absl::string_view text) {
  size_t max_overlap = 0;
  for (const auto& channel : channels) {
    size_t overlap = SuffixPrefixOverlap(text, channel.start);
    if (overlap > max_overlap) {
      max_overlap = overlap;
    }
  }
  return max_overlap;
}

// Streams out channel tokens. Only streams text that it safely validates could
// not possibly be a partial overlap of the active channel end delimiter.
void StreamActiveChannel(
    absl::AnyInvocable<void(absl::StatusOr<Message>)>& user_callback,
    absl::string_view accumulated_response_text, size_t search_start,
    size_t& cursor, absl::string_view active_channel_end,
    const std::string& active_channel_name) {
  // Stream channel content except for potential partial matches of
  // the end delimiter.
  size_t overlap = SuffixPrefixOverlap(
      accumulated_response_text.substr(search_start), active_channel_end);
  size_t safe_end = accumulated_response_text.size() - overlap;
  if (safe_end > cursor) {
    SendMessageToChannel(
        user_callback,
        accumulated_response_text.substr(cursor, safe_end - cursor),
        active_channel_name);
    cursor = safe_end;
  }
}

}  // namespace

// Creates an internal callback that parses the model's raw text responses.
//
// This parser supports "channels" defined by start and end delimiters.
// 1. Tool Calls: These act as a special channel (defined by
// CodeFenceStart/End).
//    Because their content needs to be parsed as a whole object (e.g. JSON),
//    tool calls do not stream. The parser buffers the text and waits for the
//    end delimiter before using `model_data_processor.ToMessage` to emit them.
//    This is represented internally by an empty `active_message_field`.
//    `custom_channels`. If a custom channel specifies a
//    `channel_name`, its text is streamed out immediately as new JSON messages
//    with that specific field as it arrives instead of being buffered.
absl::AnyInvocable<void(absl::StatusOr<Responses>)> CreateInternalCallback(
    const ModelDataProcessor& model_data_processor,
    const DataProcessorArguments processor_args,
    const std::vector<Channel>& custom_channels,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    absl::AnyInvocable<void()> cancel_callback,
    absl::AnyInvocable<void(Message)> complete_message_callback,
    const std::optional<std::string>& open_channel_name) {
  auto channels = GetChannels(model_data_processor, custom_channels);

  bool initial_inside_channel = false;
  std::string initial_active_channel_end;
  std::string initial_active_channel_name;

  if (open_channel_name.has_value()) {
    auto open_channel_it = std::find_if(
        channels.begin(), channels.end(), [&](const auto& channel) {
          return channel.channel_name == *open_channel_name;
        });
    if (open_channel_it != channels.end()) {
      initial_inside_channel = true;
      initial_active_channel_end = open_channel_it->end;
      initial_active_channel_name = open_channel_it->channel_name;
    }
  }

  return [&model_data_processor, processor_args,
          user_callback = std::move(user_callback),
          cancel_callback = std::move(cancel_callback),
          complete_message_callback = std::move(complete_message_callback),
          accumulated_response_text = std::string(), cursor = size_t(0),
          channels = std::move(channels),
          inside_channel = initial_inside_channel,
          active_channel_end = std::move(initial_active_channel_end),
          active_channel_start_pos = size_t(0),
          active_channel_start_size = size_t(0),
          active_channel_name = std::move(initial_active_channel_name),
          open_channel_name](absl::StatusOr<Responses> responses) mutable {
    if (!responses.ok()) {
      // If the error is due to cancellation, then we should trigger the cancel
      // callback for removing the last message from the history.
      if (cancel_callback && absl::IsCancelled(responses.status())) {
        cancel_callback();
      }
      user_callback(responses.status());
      return;
    }

    // If there are no more new responses, it means the model has finished
    // generating content, trigger the complete message callback and return an
    // OK status to indicate the inference is done.
    if (responses->GetTaskState() == TaskState::kDone ||
        responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
      SendCompleteMessage(user_callback, accumulated_response_text,
                          model_data_processor, processor_args, cursor,
                          complete_message_callback,
                          inside_channel ? active_channel_name : "", channels,
                          open_channel_name);
      cursor = accumulated_response_text.size();
      return;
    }

    // Else, add the new response text to the accumulated text and process the
    // response text.(Which sends to the user callback accordingly.)
    if (responses->GetTaskState() == TaskState::kProcessing) {
      // If there are no new responses, it is just a state update and we can
      // return early.
      if (responses->GetTexts().empty()) {
        return;
      }

      // Append the new response text to the accumulated text.
      accumulated_response_text += responses->GetTexts()[0];

      // Loop through the accumulated response text and send to the user
      // callback accordingly.
      while (cursor < accumulated_response_text.size()) {
        if (!inside_channel) {
          size_t channel_start_pos;
          const Channel* next_channel = FindNextChannelStart(
              channels, accumulated_response_text, cursor, channel_start_pos);

          if (next_channel != nullptr) {
            // A channel start delimiter was found.
            // The text from the cursor up to the channel start is normal text
            // and can be sent to the user callback.
            SendMessage(user_callback,
                        absl::string_view(accumulated_response_text)
                            .substr(cursor, channel_start_pos - cursor),
                        model_data_processor, processor_args);

            // Move cursor up to channel start.
            cursor = channel_start_pos;
            inside_channel = true;
            active_channel_end = next_channel->end;
            active_channel_start_pos = channel_start_pos;
            active_channel_start_size = next_channel->start.size();
            active_channel_name = next_channel->channel_name;

            // For custom channels, move the cursor past the start delimiter so
            // that it is not included in the resulting streamed content.
            // For tool calls (empty message field), we leave the cursor alone
            // to buffer the complete block including delimiters.
            if (!active_channel_name.empty()) {
              cursor += active_channel_start_size;
            }
          } else {
            // A channel start delimiter was not found. We still need to check
            // if there's a partial match of any channel start at the very end
            // of the string.
            size_t max_overlap = FindMaxOverlap(
                channels,
                absl::string_view(accumulated_response_text).substr(cursor));

            if (max_overlap > 0) {
              // There's a partial match of a channel at the end of the
              // string.
              size_t possible_start_pos =
                  accumulated_response_text.size() - max_overlap;

              // Call the callback with text up to the potential start of the
              // channel.
              SendMessage(user_callback,
                          accumulated_response_text.substr(
                              cursor, possible_start_pos - cursor),
                          model_data_processor, processor_args);

              // Move cursor up to potential start of channel.
              cursor = possible_start_pos;

              // Break for the next token.
              break;
            } else {
              // Remaining string is text.
              SendMessage(user_callback,
                          accumulated_response_text.substr(cursor),
                          model_data_processor, processor_args);

              cursor = accumulated_response_text.size();
            }
          }
        }

        if (inside_channel) {
          // Look for channel end.
          size_t search_start =
              std::max(static_cast<size_t>(cursor),
                       active_channel_start_pos + active_channel_start_size);
          size_t end_pos =
              accumulated_response_text.find(active_channel_end, search_start);

          if (end_pos != std::string::npos) {
            // A channel end delimiter was found.
            if (!active_channel_name.empty()) {
              // Flush the active stream channel.
              SendMessageToChannel(user_callback,
                                   absl::string_view(accumulated_response_text)
                                       .substr(cursor, end_pos - cursor),
                                   active_channel_name);
            } else {
              // Treat as tool call: include everything up to and including the
              // end delimiter.
              SendMessage(
                  user_callback,
                  accumulated_response_text.substr(
                      cursor, end_pos + active_channel_end.size() - cursor),
                  model_data_processor, processor_args);
            }

            // Move cursor past the end of the channel block.
            cursor = end_pos + active_channel_end.size();
            inside_channel = false;
          } else {
            // We're inside a channel or tool call but the end has not been
            // found.
            if (!active_channel_name.empty()) {
              // If we're inside a channel, stream the text, but stop before
              // any potential partial match of the channel's end delimiter.
              StreamActiveChannel(user_callback, accumulated_response_text,
                                  search_start, cursor, active_channel_end,
                                  active_channel_name);
            }

            // Break for the next token.
            break;
          }
        }
      }
    }
  };
}

}  // namespace litert::lm
