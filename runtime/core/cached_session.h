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
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/core/prefix_cache.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Configuration options for initializing a CachedSession.
struct CachedSessionOptions {
  // Executor properties for vision processing (e.g., tokens per image, patch
  // shrink factor). Required when processing image inputs to compute cache
  // token footprint.
  std::optional<VisionExecutorProperties> vision_properties = std::nullopt;

  // Executor properties for audio processing (e.g., audio shrink factor).
  // Required when processing audio inputs to compute cache token footprint.
  std::optional<AudioExecutorProperties> audio_properties = std::nullopt;

  // Whether to automatically prepend the beginning-of-sequence (BOS) token ID
  // to the prompt inputs during prefill.
  bool insert_bos_token_id = false;
};

// A wrapper around Session that provides automatic prefix caching across
// multiple execution requests.
//
// Usage:
// Unlike a standard Session where callers must manually track incremental
// prompt suffixes and manage session rewind steps, CachedSession expects the
// *full* prompt sequence on every prefill request.
//
// 1. Initialization: Construct CachedSession from an underlying Session, a
//    Tokenizer, and options.
// 2. Prefill: Call RunPrefill or RunPrefillAsync with the complete prompt.
//    CachedSession matches multimodal inputs (text, image, audio) against its
//    internal cache, automatically rewinds the underlying session if needed,
//    and only executes prefill on the unmatched suffix.
// 3. Decode: Call RunDecode or RunDecodeAsync. Newly generated tokens are
//    automatically recorded in the prefix cache.
class CachedSession {
 public:
  CachedSession(std::unique_ptr<SessionInterface> session,
                litert::support::Tokenizer* tokenizer,
                const CachedSessionOptions& options = {});

  ~CachedSession() = default;

  CachedSession(const CachedSession&) = delete;
  CachedSession& operator=(const CachedSession&) = delete;
  CachedSession(CachedSession&&) = default;
  CachedSession& operator=(CachedSession&&) = default;

  // Expects the *full* prompt. Matches input tokens and images/audio against
  // the cache, rewinds the inner session, and prefills the difference.
  // Input text can be raw or preprocessed; preprocessing is handled internally.
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunPrefillAsync(const std::vector<InputData>& contents,
                  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback);

  // Synchronous version of RunPrefillAsync.
  absl::Status RunPrefill(const std::vector<InputData>& contents);

  // Asynchronously generates tokens and appends them to the PrefixCache.
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunDecodeAsync(absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
                 const DecodeConfig& decode_config);

  // Same as RunDecodeAsync but with a default decode_config.
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunDecodeAsync(absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback);

  // Generates tokens and appends them to the PrefixCache.
  absl::StatusOr<Responses> RunDecode(const DecodeConfig& decode_config);

  // Same as RunDecode but with a default decode_config.
  absl::StatusOr<Responses> RunDecode();

  // Cancels the current process in the Session.
  void CancelProcess() { session_->CancelProcess(); }

  // Waits until the Session is done.
  absl::Status WaitUntilDone() { return session_->WaitUntilDone(); }

  // Resets the PrefixCache and rewinds the underlying Session to step 0.
  absl::Status Reset();

  // Returns the config of the contained Session.
  const SessionConfig& GetSessionConfig() const {
    return session_->GetSessionConfig();
  }

  // Returns the current step of the contained Session.
  absl::StatusOr<int> GetCurrentStep() const {
    return session_->GetCurrentStep();
  }

  // Returns the benchmark info of the contained Session.
  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() {
    return session_->GetBenchmarkInfo();
  }

  // Returns the mutable benchmark info of the contained Session.
  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() {
    return session_->GetMutableBenchmarkInfo();
  }

  // Clones the CachedSession along with its underlying Session and prefix
  // cache.
  absl::StatusOr<std::unique_ptr<CachedSession>> Clone() const;

  // Clones the CachedSession asynchronously along with its underlying Session
  // and prefix cache.
  absl::StatusOr<std::unique_ptr<CachedSession>> CloneAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) const;

  // Runs text scoring on the target texts. This function is meant to be used
  // for evaluation after the session has already been prefilled with context.
  // Note that `target_text` is only the candidate text to be evaluated against
  // the prefilled context; it should not include the whole prefill (unlike what
  // `RunPrefill` expects).
  absl::StatusOr<Responses> RunTextScoring(
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths);

  // Asynchronously runs text scoring on the target texts. This function is
  // meant to be used for evaluation after the session has already been
  // prefilled with context. Note that `target_text` is only the candidate text
  // to be evaluated against the prefilled context; it should not include the
  // whole prefill (unlike what `RunPrefillAsync` expects).
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
  RunTextScoringAsync(
      const std::vector<absl::string_view>& target_text,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      bool store_token_lengths);

  // Sets whether to insert a BOS token ID at the beginning of the prefill
  // contents.
  void SetInsertBosTokenId(bool insert_bos_token_id) {
    insert_bos_token_id_ = insert_bos_token_id;
  }

  // Returns the PrefixCache.
  const PrefixCache& GetPrefixCache() const { return prefix_cache_; }

 private:
  std::unique_ptr<SessionInterface> session_;
  litert::support::Tokenizer* tokenizer_;  // Not owned.
  std::optional<VisionExecutorProperties> vision_properties_;
  std::optional<AudioExecutorProperties> audio_properties_;
  PrefixCache prefix_cache_;
  bool insert_bos_token_id_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_CACHED_SESSION_H_
