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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CJK_TEST_BLOBS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CJK_TEST_BLOBS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace litert::omni::tts::kokoro {

// Writers for the packed structures the CJK front end reads, so tests can
// state a dictionary in a few lines instead of shipping a fixture.
//
// The real blobs are produced by unidic-lite and jieba, and they are far too
// large to check in. These build the same layouts at a size a test can reason
// about.

// Appends a little-endian 32-bit unsigned integer to a binary byte buffer.
//
// args
// - value: 32-bit unsigned integer to serialize in little-endian byte order.
// - out: Output byte string to append the 4 serialized bytes to.
void AppendUint32(uint32_t value, std::string& out);

// Appends a little-endian 16-bit unsigned integer to a binary byte buffer.
//
// args
// - value: 16-bit unsigned integer to serialize in little-endian byte order.
// - out: Output byte string to append the 2 serialized bytes to.
void AppendUint16(uint16_t value, std::string& out);

// Builds a packed binary blob in the `LCJK` container layout emitted by
// `blob_container.py`: an 8-byte magic header (`LCJK\0\0\0\0`), format version
// (1), section count, and a directory of `(name, offset, size)` entries over an
// 8-byte-aligned body.
//
// Usage:
// ```cpp
// std::string raw_blob = BuildCjkBlob({
//     {"zh.jieba.keys", keys_bytes},
//     {"zh.jieba.meta", meta_bytes},
// });
// LITERT_ASSIGN_OR_RETURN(CjkBlob blob, CjkBlob::Create(raw_blob));
// ```
//
// args
// - sections: Ordered list of `(section_name, payload_bytes)` pairs to pack.
//   Each `section_name` must be at most 23 bytes long.
//
// returns
// - Serialized `LCJK` container bytes suitable for `CjkBlob::Create`.
std::string BuildCjkBlob(
    const std::vector<std::pair<std::string, std::string>>& sections);

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CJK_TEST_BLOBS_H_
