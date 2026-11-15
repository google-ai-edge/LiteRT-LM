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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_WHISPER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_WHISPER_H_

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/omni_engine.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::asr {

// Resolves `Speech2TextMetadata` for Whisper ASR models, applying Whisper
// defaults and resolving `session_config`'s optional `language` (via
// `tokenizer` when available, with a fallback to the 99 shared Whisper
// language codes) into `decode_start_token_ids`
// (`[<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]`).
//
// Args:
//   metadata: Optional input `Speech2TextMetadata` proto to merge from.
//   session_config: Optional `Speech2TextSessionConfig` proto carrying an
//     optional BCP-47 `language` hint.
//   tokenizer: Optional Whisper tokenizer used to look up
//     `<|startoftranscript|>`, `<|lang|>`, `<|transcribe|>`, and
//     `<|notimestamps|>` token IDs.
//
// Returns:
//   Resolved `Speech2TextMetadata`, or `absl::InvalidArgumentError` if
//   `session_config` specifies an unrecognized language.
absl::StatusOr<lm::proto::Speech2TextMetadata> ResolveWhisperMetadata(
    const MessageLite* absl_nullable metadata,
    const MessageLite* absl_nullable session_config = nullptr,
    ::litert::support::Tokenizer* absl_nullable tokenizer = nullptr);

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_WHISPER_H_
