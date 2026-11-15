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

#include "sync/sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm::sync {
namespace {

// FIPS 180-4 / NIST CAVP SHA-256 test vectors.
constexpr absl::string_view kEmptyDigest =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
constexpr absl::string_view kAbcDigest =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
constexpr absl::string_view k448BitMessage =
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
constexpr absl::string_view k448BitDigest =
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
constexpr absl::string_view k896BitMessage =
    "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
    "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
constexpr absl::string_view k896BitDigest =
    "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1";
constexpr absl::string_view kMillionADigest =
    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";

TEST(ComputeSha256HexTest, MatchesNistVectors) {
  EXPECT_EQ(ComputeSha256Hex(""), kEmptyDigest);
  EXPECT_EQ(ComputeSha256Hex("abc"), kAbcDigest);
  EXPECT_EQ(ComputeSha256Hex(k448BitMessage), k448BitDigest);
  EXPECT_EQ(ComputeSha256Hex(k896BitMessage), k896BitDigest);
}

TEST(ComputeSha256HexTest, HandlesPaddingBoundaries) {
  // Lengths around the 56-byte padding cutoff and the 64-byte block size
  // exercise both the one-block and two-block padding paths.
  struct TestCase {
    size_t length;
    absl::string_view digest;
  };
  static constexpr TestCase kCases[] = {
      {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
      {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
      {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
      {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
      {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
  };
  for (const TestCase& test_case : kCases) {
    EXPECT_EQ(ComputeSha256Hex(std::string(test_case.length, 'a')),
              test_case.digest)
        << "length=" << test_case.length;
  }
}

TEST(ComputeSha256HexTest, HashesEmbeddedNulBytes) {
  // The input is hashed by length, not as a NUL-terminated C string.
  EXPECT_EQ(ComputeSha256Hex(std::string(3, '\0')),
            "709e80c88487a2411e1ee4dfb9f22a861492d20c4765150c0c794abd70f8147c");
}

TEST(Sha256HasherTest, DefaultConstructedHasherHashesEmptyMessage) {
  Sha256Hasher hasher;
  EXPECT_EQ(hasher.FinalHex(), kEmptyDigest);
}

TEST(Sha256HasherTest, FinalReturnsRawDigestBytes) {
  static constexpr std::array<uint8_t, 32> kExpected = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
      0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
      0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  Sha256Hasher hasher;
  hasher.Update("abc");
  EXPECT_EQ(hasher.Final(), kExpected);
}

TEST(Sha256HasherTest, IncrementalUpdatesMatchOneShot) {
  for (size_t chunk_size = 1; chunk_size <= k896BitMessage.size();
       ++chunk_size) {
    Sha256Hasher hasher;
    for (size_t pos = 0; pos < k896BitMessage.size(); pos += chunk_size) {
      hasher.Update(k896BitMessage.substr(pos, chunk_size));
    }
    EXPECT_EQ(hasher.FinalHex(), k896BitDigest) << "chunk_size=" << chunk_size;
  }
}

TEST(Sha256HasherTest, MillionCharacterMessage) {
  // FIPS 180-4 long message: one million repetitions of 'a'.
  const std::string chunk(1000, 'a');
  Sha256Hasher hasher;
  for (int i = 0; i < 1000; ++i) {
    hasher.Update(chunk);
  }
  EXPECT_EQ(hasher.FinalHex(), kMillionADigest);
}

TEST(Sha256HasherTest, RawPointerUpdateMatchesStringUpdate) {
  constexpr absl::string_view kMessage = "abc";
  Sha256Hasher hasher;
  hasher.Update(kMessage.data(), kMessage.size());
  EXPECT_EQ(hasher.FinalHex(), kAbcDigest);
}

TEST(Sha256HasherTest, ZeroLengthUpdateIsNoOp) {
  Sha256Hasher hasher;
  hasher.Update(nullptr, 0);
  hasher.Update("abc");
  hasher.Update("");
  EXPECT_EQ(hasher.FinalHex(), kAbcDigest);
}

TEST(Sha256HasherTest, ResetDiscardsPendingInput) {
  Sha256Hasher hasher;
  hasher.Update("input that should be discarded");
  hasher.Reset();
  hasher.Update("abc");
  EXPECT_EQ(hasher.FinalHex(), kAbcDigest);
}

TEST(Sha256HasherTest, FinalResetsHasherForReuse) {
  Sha256Hasher hasher;
  hasher.Update("abc");
  EXPECT_EQ(hasher.FinalHex(), kAbcDigest);
  // After finalization, the hasher starts over from an empty message.
  EXPECT_EQ(hasher.FinalHex(), kEmptyDigest);
  hasher.Update("abc");
  EXPECT_EQ(hasher.FinalHex(), kAbcDigest);
}

}  // namespace
}  // namespace litert::lm::sync
