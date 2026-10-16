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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_TEST_BLOBS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_TEST_BLOBS_H_

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

// Builds a Darts 0.32 double array over `entries`, which map a key to the
// value MeCab packs as `(token index << 8) | token count`. Keys must be
// distinct.
//
// MeCab's own arrays are compacted, with unrelated nodes sharing slots. This
// one gives every trie node its own stride, which wastes most of the array but
// makes the layout something a reader can check by hand.
std::string BuildMecabDoubleArray(
    const std::vector<std::pair<std::string, int>>& entries);

// Builds one 16-byte MeCab token row.
std::string BuildMecabToken(int left_id, int right_id, int pos_id,
                            int word_cost, uint32_t feature_offset);

// Concatenates `records` into the NUL-separated feature blob token rows point
// into, and appends each record's offset to `offsets`.
std::string BuildMecabFeatures(const std::vector<std::string>& records,
                               std::vector<uint32_t>& offsets);

// Builds a `matrix.bin` whose body is `costs` indexed the way
// `Connector::cost` indexes it, `costs[right_id + right_context_count *
// left_id]`. The arguments are in MeCab's header order, which writes the
// right-context count first under the name `lsize`.
std::string BuildMecabConnectionMatrix(int right_context_count,
                                       int left_context_count,
                                       const std::vector<int>& costs);

// Packs one `char.bin` record. The argument order here is the order MeCab
// declares the bitfield in, so a caller cannot reproduce the swap of `group`
// and `invoke` that the names invite.
uint32_t PackMecabCharInfo(uint32_t types, int default_type, int length,
                           bool group, bool invoke);

// Builds a `char.bin` over `names`, taking `records` as the first code points
// and leaving every later one in category zero.
std::string BuildMecabCharProperty(const std::vector<std::string>& names,
                                   const std::vector<uint32_t>& records,
                                   int code_points);

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_TEST_BLOBS_H_
