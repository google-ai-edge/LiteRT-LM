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

#include "runtime/core/cached_session.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/tokenizer.h"
#include "runtime/core/prefix_cache.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

// A placeholder TaskController for synchronous or immediately-completed
// operations. This is returned when an operation is fully cached or completed
// instantly without launching any actual asynchronous background work,
// providing non-null, safe dummy implementations of control methods.
class NoOpTaskController : public SessionInterface::TaskController {
 public:
  NoOpTaskController() = default;
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return absl::OkStatus();
  }
  absl::Status Cancel() override { return absl::OkStatus(); }
};

// Helper to convert Expected to StatusOr
template <typename T>
absl::StatusOr<T> ToStatusOr(litert::Expected<T>&& expected) {
  if (!expected.HasValue()) {
    return litert::ErrorStatusBuilder(std::move(expected));
  }
  return std::move(expected.Value());
}

// Helper to hash raw memory.
std::string HashBytes(const void* ptr, size_t size) {
  size_t hash_val =
      absl::HashOf(std::string_view(static_cast<const char*>(ptr), size));
  return std::to_string(hash_val);
}

// Helper to hash a TensorBuffer.
absl::StatusOr<std::string> HashTensorBuffer(
    const litert::TensorBuffer& buffer) {
  ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ToStatusOr(litert::TensorBufferScopedLock::Create<const void>(
          buffer, litert::TensorBuffer::LockMode::kRead)));
  ASSIGN_OR_RETURN(size_t size, ToStatusOr(buffer.PackedSize()));
  return HashBytes(lock_and_addr.second, size);
}

}  // namespace

CachedSession::CachedSession(std::unique_ptr<SessionInterface> session,
                             Tokenizer* tokenizer,
                             const CachedSessionOptions& options)
    : session_(std::move(session)),
      tokenizer_(tokenizer),
      vision_properties_(options.vision_properties),
      audio_properties_(options.audio_properties),
      insert_bos_token_id_(options.insert_bos_token_id) {}

absl::StatusOr<MediaHash> CachedSession::GetImageHash(
    const InputImage& input_image) {
  std::string hash;

  // 1. Generate a unique cryptographic hash for the image content.
  if (input_image.IsTensorBuffer()) {
    // Case A: The image is a single preprocessed TensorBuffer.
    // Hash it directly.
    ASSIGN_OR_RETURN(const TensorBuffer* buffer,
                     input_image.GetPreprocessedImageTensor());
    ASSIGN_OR_RETURN(hash, HashTensorBuffer(*buffer));
  } else if (input_image.IsTensorBufferMap()) {
    // Case B: The image is a map of preprocessed TensorBuffers.
    // Sort the keys to guarantee a deterministic combination order.
    ASSIGN_OR_RETURN(const auto* map,
                     input_image.GetPreprocessedImageTensorMap());
    std::vector<std::string> keys;
    for (const auto& [k, _] : *map) {
      keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end());

    // Hash each TensorBuffer and combine their hashes.
    std::string combined_hash;
    for (const auto& k : keys) {
      ASSIGN_OR_RETURN(auto h, HashTensorBuffer(map->at(k)));
      combined_hash += h;
    }
    // The final identifier is the hash of the combined individual hashes.
    hash = HashBytes(combined_hash.data(), combined_hash.size());
  } else {
    // Case C: The image consists of raw bytes. Hash the raw bytes directly.
    ASSIGN_OR_RETURN(absl::string_view bytes, input_image.GetRawImageBytes());
    hash = HashBytes(bytes.data(), bytes.size());
  }

  // 2. Calculate the number of LLM tokens the image will consume.
  // Vision properties is required to determine the number of tokens per image.
  if (!vision_properties_.has_value()) {
    return absl::FailedPreconditionError(
        "Vision properties not set for image input.");
  }

  // Default to the statically configured token count.
  int token_length = vision_properties_->num_tokens_per_image;

  // If dynamic patching (e.g. ViT) is enabled and we have spatial metadata:
  if (vision_properties_->patch_num_shrink_factor.has_value() &&
      input_image.IsTensorBufferMap()) {
    ASSIGN_OR_RETURN(const auto* map,
                     input_image.GetPreprocessedImageTensorMap());
    // Look up spatial coordinates to calculate the number of active patches.
    if (map->contains("positions_xy")) {
      ASSIGN_OR_RETURN(auto type,
                       ToStatusOr(map->at("positions_xy").TensorType()));
      auto dims = type.Layout().Dimensions();
      if (dims.size() >= 2) {
        int num_patches_from_input = dims[1];
        int shrink = vision_properties_->patch_num_shrink_factor.value();
        // Calculate token length using ceiling division:
        // (num_patches / shrink_factor)
        token_length = (num_patches_from_input + shrink - 1) / shrink;
      }
    }
  }

  return MediaHash{hash, token_length};
}

absl::StatusOr<MediaHash> CachedSession::GetAudioHash(
    const InputAudio& input_audio) {
  if (!audio_properties_.has_value()) {
    return absl::FailedPreconditionError(
        "Audio properties not set for audio input.");
  }

  // Caching only supports preprocessed audio TensorBuffers because we must
  // know the sequence length to calculate the LLM token footprint.
  if (!input_audio.IsTensorBuffer()) {
    return absl::FailedPreconditionError(
        "Cannot determine audio token length for raw audio.");
  }

  // Generate a unique cryptographic hash for the preprocessed TensorBuffer.
  ASSIGN_OR_RETURN(const TensorBuffer* buffer,
                   input_audio.GetPreprocessedAudioTensor());
  ASSIGN_OR_RETURN(std::string hash, HashTensorBuffer(*buffer));

  // Extract the audio sequence length from the tensor's second-to-last
  // dimension (layout format dependent, e.g. [batch, seq_len, features]).
  ASSIGN_OR_RETURN(auto type, ToStatusOr(buffer->TensorType()));
  auto dims = type.Layout().Dimensions();
  int input_sequence_length = 0;
  if (dims.size() >= 2) {
    input_sequence_length = dims[dims.size() - 2];
  }

  if (input_sequence_length <= 0) {
    return absl::FailedPreconditionError(
        "Invalid or empty audio sequence length in TensorBuffer.");
  }

  int shrink = audio_properties_->audio_shrink_factor;
  int token_length = (input_sequence_length + shrink - 1) / shrink;

  return MediaHash{hash, token_length};
}

absl::StatusOr<ConversionResult> CachedSession::ConvertToCacheElements(
    const std::vector<InputData>& contents) {
  ConversionResult result;

  // Process each input element in the prompt sequentially.
  for (const auto& content : contents) {
    if (const auto* input_text = std::get_if<InputText>(&content)) {
      // The text is assumed to be a preprocessed TensorBuffer.
      if (!input_text->IsTensorBuffer()) {
        return absl::InternalError(
            "Expected preprocessed text input (TensorBuffer) in "
            "ConvertToCacheElements.");
      }
      ASSIGN_OR_RETURN(const litert::TensorBuffer* tensor,
                       input_text->GetPreprocessedTextTensor());
      ASSIGN_OR_RETURN(auto ids_vec,
                       Tokenizer::TensorBufferToTokenIds(*tensor));
      if (ids_vec.size() != 1) {
        return absl::InternalError(
            "Expected batch size 1 for input text tensor.");
      }
      result.content_sizes.push_back(ids_vec[0].size());
      // Append each token ID as a separate CacheElement.
      for (int id : ids_vec[0]) {
        result.elements.push_back(id);
      }
    } else if (const auto* input_image = std::get_if<InputImage>(&content)) {
      // Compute the unique hash and token footprint, and cache it.
      ASSIGN_OR_RETURN(auto media_hash, GetImageHash(*input_image));
      result.elements.push_back(media_hash);
      result.content_sizes.push_back(1);
    } else if (const auto* input_audio = std::get_if<InputAudio>(&content)) {
      // Compute the unique hash and token footprint, and cache it.
      ASSIGN_OR_RETURN(auto media_hash, GetAudioHash(*input_audio));
      result.elements.push_back(media_hash);
      result.content_sizes.push_back(1);
    } else if (std::holds_alternative<InputImageEnd>(content) ||
               std::holds_alternative<InputAudioEnd>(content)) {
      // Skip delimiters.
      result.content_sizes.push_back(0);
      continue;
    } else {
      return absl::InvalidArgumentError("Unsupported input type.");
    }
  }
  return result;
}

absl::StatusOr<std::vector<InputData>> CachedSession::SliceInputContents(
    const std::vector<InputData>& contents,
    const std::vector<int>& content_sizes,
    const std::vector<CacheElement>& cache_elements, int matched_elements) {
  if (content_sizes.size() != contents.size()) {
    return absl::InvalidArgumentError(
        "content_sizes size must match contents size.");
  }
  size_t expected_cache_size = 0;
  for (int size : content_sizes) {
    if (size < 0) {
      return absl::InvalidArgumentError(
          "content_sizes elements must be non-negative.");
    }
    expected_cache_size += size;
  }
  if (cache_elements.size() < expected_cache_size) {
    return absl::InvalidArgumentError(
        "cache_elements size is smaller than expected from content_sizes.");
  }

  // The slice of `contents` that isn't in the cache.
  std::vector<InputData> sliced_contents;

  // Index of current cache element as we iterate through `contents`.
  int cache_index = 0;

  for (size_t i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];

    if (i >= content_sizes.size()) {
      return absl::InternalError("Index out of bounds for content_sizes.");
    }
    // Number of cache elements corresponding to contents[i].
    int content_size = content_sizes[i];

    // The start and end indexes of the current InputData in the cache array.
    int content_start = cache_index;
    int content_end = cache_index + content_size;

    // Increment cache element index.
    cache_index += content_size;

    if (matched_elements >= content_end) {
      // Full match in cache. Skip it.
      continue;
    } else if (matched_elements > content_start) {
      // Partial match in cache. This can only happen with text input.
      if (!std::holds_alternative<InputText>(content)) {
        return absl::InternalError(
            "Partial cache match of non-text InputData.");
      }

      int matched_tokens_in_content = matched_elements - content_start;
      std::vector<int> remaining_tokens;
      for (int j = content_start + matched_tokens_in_content; j < content_end;
           ++j) {
        if (j >= cache_elements.size()) {
          return absl::InternalError("Index out of bounds for cache_elements.");
        }
        remaining_tokens.push_back(std::get<int>(cache_elements[j]));
      }

      // Convert only the unmatched suffix tokens to a TensorBuffer.
      ASSIGN_OR_RETURN(auto sliced_buffer,
                       tokenizer_->TokenIdsToTensorBuffer(remaining_tokens));
      sliced_contents.push_back(InputText(std::move(sliced_buffer)));
    } else {
      // No cache match. Use the original content.
      ASSIGN_OR_RETURN(auto copy, CreateInputDataCopy(content));
      sliced_contents.push_back(std::move(copy));
    }
  }

  return sliced_contents;
}

absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
CachedSession::RunPrefillAsync(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  // Preprocess the input contents, converting text inputs to TensorBuffers of
  // token IDs.
  ASSIGN_OR_RETURN(auto preprocessed_contents,
                   PreprocessContents(contents, GetSessionConfig(), *tokenizer_,
                                      /*benchmark_info=*/std::nullopt));

  if (insert_bos_token_id_) {
    int bos_token_id = session_->GetSessionConfig().GetStartTokenId();
    if (bos_token_id >= 0) {
      ASSIGN_OR_RETURN(auto bos_id_tensor,
                       tokenizer_->TokenIdsToTensorBuffer({bos_token_id}));
      preprocessed_contents.insert(preprocessed_contents.begin(),
                                   InputText(std::move(bos_id_tensor)));
    }
  }

  // Convert input elements (text, images, audio) to CacheElements.
  ASSIGN_OR_RETURN(auto incoming_elements,
                   ConvertToCacheElements(preprocessed_contents));

  // Search the local PrefixCache for the longest matched prefix.
  // Find the longest common prefix between the input and the cached elements.
  auto match_result =
      prefix_cache_.FindLongestCommonPrefix(incoming_elements.elements);

  // Complete cache miss.
  if (match_result.matched_elements == 0) {
    RETURN_IF_ERROR(session_->RewindToStep(0));
    prefix_cache_.Clear();

    // Wrap the callback to populate the cache with the full prompt on success.
    auto wrapped_callback = [this,
                             elements = std::move(incoming_elements.elements),
                             callback = std::move(callback)](
                                absl::StatusOr<Responses> responses) mutable {
      if (responses.ok() && responses->GetTaskState() == TaskState::kDone) {
        prefix_cache_.AppendElements(elements);
      }
      callback(std::move(responses));
    };
    return session_->PrefillPreprocessedContents(
        std::move(preprocessed_contents), std::move(wrapped_callback));
  }

  // If no new tokens need to be prefilled, complete immediately.
  if (incoming_elements.elements.size() == match_result.matched_elements) {
    callback(Responses(TaskState::kDone));
    return std::make_unique<NoOpTaskController>();
  }

  // Cache hit. Rewind the session to the matched token offset.
  RETURN_IF_ERROR(session_->RewindToStep(match_result.matched_tokens));

  // Truncate local cache to discard mismatched trailing entries.
  prefix_cache_.Truncate(match_result.matched_elements);

  // Get the cache elements corresponding to the part of the input we need to
  // prefill.
  std::vector<CacheElement> remaining_elements;
  for (size_t i = match_result.matched_elements;
       i < incoming_elements.elements.size(); ++i) {
    remaining_elements.push_back(incoming_elements.elements[i]);
  }

  // Slice the input contents for the remaining prefill.
  ASSIGN_OR_RETURN(
      auto sliced_contents,
      SliceInputContents(preprocessed_contents, incoming_elements.content_sizes,
                         incoming_elements.elements,
                         match_result.matched_elements));

  // Wrap the callback to append the newly prefilled elements to the cache.
  auto wrapped_callback = [this,
                           remaining_elements = std::move(remaining_elements),
                           callback = std::move(callback)](
                              absl::StatusOr<Responses> responses) mutable {
    if (responses.ok() && responses->GetTaskState() == TaskState::kDone) {
      prefix_cache_.AppendElements(remaining_elements);
    }
    callback(std::move(responses));
  };

  return session_->PrefillPreprocessedContents(std::move(sliced_contents),
                                               std::move(wrapped_callback));
}

absl::Status CachedSession::RunPrefill(const std::vector<InputData>& contents) {
  absl::Status status = absl::OkStatus();
  ASSIGN_OR_RETURN(
      auto task_controller,
      RunPrefillAsync(contents, [&status](absl::StatusOr<Responses> responses) {
        status = responses.status();
      }));
  RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return status;
}

absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
CachedSession::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return RunDecodeAsync(std::move(callback), DecodeConfig::CreateDefault());
}

absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
CachedSession::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  auto generated_tokens = std::make_shared<std::vector<int>>();

  auto wrapped_callback = [this, generated_tokens,
                           callback = std::move(callback)](
                              absl::StatusOr<Responses> responses) mutable {
    if (responses.ok()) {
      const auto& token_ids = responses->GetTokenIds();
      if (!token_ids.empty() && !token_ids[0].empty()) {
        generated_tokens->insert(generated_tokens->end(), token_ids[0].begin(),
                                 token_ids[0].end());
      }

      if (responses->GetTaskState() == TaskState::kDone ||
          responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
        if (!generated_tokens->empty()) {
          prefix_cache_.AppendTokens(*generated_tokens);
        }
      }
    }
    callback(std::move(responses));
  };

  return session_->RunDecodeAsync(std::move(wrapped_callback), decode_config);
}

absl::StatusOr<Responses> CachedSession::RunDecode(
    const DecodeConfig& decode_config) {
  ASSIGN_OR_RETURN(Responses responses, session_->RunDecode(decode_config));
  const auto& token_ids = responses.GetTokenIds();
  if (!token_ids.empty() && !token_ids[0].empty()) {
    prefix_cache_.AppendTokens(token_ids[0]);
  }
  return responses;
}

absl::StatusOr<Responses> CachedSession::RunDecode() {
  return RunDecode(DecodeConfig::CreateDefault());
}

}  // namespace litert::lm
