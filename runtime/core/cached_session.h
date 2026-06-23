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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_CACHED_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_CACHED_SESSION_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/core/prefix_cache.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

struct ConversionResult {
  std::vector<CacheElement> elements;
  std::vector<int> content_sizes;
};

struct CachedSessionOptions {
  std::optional<VisionExecutorProperties> vision_properties = std::nullopt;
  std::optional<AudioExecutorProperties> audio_properties = std::nullopt;
  bool insert_bos_token_id = false;
};

class CachedSession {
 public:
  CachedSession(std::unique_ptr<SessionInterface> session, Tokenizer* tokenizer,
                const CachedSessionOptions& options = {});

  ~CachedSession() = default;

  CachedSession(const CachedSession&) = delete;
  CachedSession& operator=(const CachedSession&) = delete;
  CachedSession(CachedSession&&) = default;
  CachedSession& operator=(CachedSession&&) = default;

  // Expects the *full* prompt. Matches input tokens and images/audio against
  // the cache, rewinds the inner session, and prefills the difference.
  // Input text can be raw or preprocessed; preprocessing is handled internally.
  absl::Status RunPrefill(const std::vector<InputData>& contents);

  // Async version of RunPrefill.
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunPrefillAsync(const std::vector<InputData>& contents,
                  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback);

  // Generates tokens and appends them to the PrefixCache.
  absl::StatusOr<Responses> RunDecode(const DecodeConfig& decode_config);

  // RunDecode with default decode_config.
  absl::StatusOr<Responses> RunDecode();

  // Async version of RunDecode.
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunDecodeAsync(absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
                 const DecodeConfig& decode_config);

  // Async version of RunDecode with default decode_config.
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunDecodeAsync(absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback);

  // Cancels the current process in the Session.
  void CancelProcess() { session_->CancelProcess(); }

  // Waits until the Session is done.
  absl::Status WaitUntilDone() { return session_->WaitUntilDone(); }

  // Returns the config of the contained Session.
  const SessionConfig& GetSessionConfig() const {
    return session_->GetSessionConfig();
  }

  // Sets whether to insert a BOS token ID at the beginning of the prefill
  // contents.
  void SetInsertBosTokenId(bool insert_bos_token_id) {
    insert_bos_token_id_ = insert_bos_token_id;
  }

  // Returns the contained Session.
  //
  // *Warning*: Operations performed on the Session pointer returned by this
  // function will *not* update the prefix cache.
  SessionInterface* GetSession() { return session_.get(); }

  // Returns the PrefixCache.
  const PrefixCache& GetPrefixCache() const { return prefix_cache_; }

 private:
  // Helper to convert InputData to CacheElements (tokens and media).
  // Assumes `contents` are already preprocessed (i.e., text inputs are
  // TensorBuffers of token IDs).
  absl::StatusOr<ConversionResult> ConvertToCacheElements(
      const std::vector<InputData>& contents);

  // Reconstructs the subset of prompt inputs that need to be prefilled,
  // starting from the first mismatched element.
  // Assumes `contents` are already preprocessed.
  //
  // - contents: List of InputData. Must already be preprocessed.
  // - content_sizes: A parallel array where content_sizes[i] represents
  //   the number of cache elements produced by contents[i].
  // - cache_elements: The vector of CacheElements corresponding to `contents`.
  // - matched_elements: The index of the first mismatched element in the
  //   elements array.
  absl::StatusOr<std::vector<InputData>> SliceInputContents(
      const std::vector<InputData>& contents,
      const std::vector<int>& content_sizes,
      const std::vector<CacheElement>& cache_elements, int matched_elements);

  absl::StatusOr<MediaHash> GetImageHash(const InputImage& input_image);
  absl::StatusOr<MediaHash> GetAudioHash(const InputAudio& input_audio);

  std::unique_ptr<SessionInterface> session_;
  Tokenizer* tokenizer_;  // Not owned.
  std::optional<VisionExecutorProperties> vision_properties_;
  std::optional<AudioExecutorProperties> audio_properties_;
  PrefixCache prefix_cache_;
  bool insert_bos_token_id_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_CACHED_SESSION_H_
