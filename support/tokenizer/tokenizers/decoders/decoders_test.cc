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
//
// C++ port of `src/decoders/mod.rs` (tests) from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/decoders.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/bpe.h"
#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"
#include "support/tokenizer/tokenizers/decoders/ctc.h"
#include "support/tokenizer/tokenizers/decoders/fuse.h"
#include "support/tokenizer/tokenizers/decoders/sequence.h"
#include "support/tokenizer/tokenizers/decoders/strip.h"
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace decoders {
namespace {

using ::testing::ElementsAre;
using ::testing::status::StatusIs;

Json J(absl::string_view s) {
  return Json::parse(s, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

TEST(DecoderWrapperTest, DecoderSerialization) {
  const std::string oldjson =
      R"({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]})";
  ASSERT_OK_AND_ASSIGN(DecoderWrapper olddecoder,
                       DecoderWrapper::FromJson(J(oldjson)));
  const std::string oldserialized = olddecoder.ToJson().dump();
  const std::string json =
      R"({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]})";
  EXPECT_EQ(oldserialized, json);

  ASSERT_OK_AND_ASSIGN(DecoderWrapper decoder,
                       DecoderWrapper::FromJson(J(json)));
  const std::string serialized = decoder.ToJson().dump();
  EXPECT_EQ(serialized, json);
}

TEST(DecoderWrapperTest, DecoderSerializationOtherNoArg) {
  const std::string json =
      R"({"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]})";
  ASSERT_OK_AND_ASSIGN(DecoderWrapper decoder,
                       DecoderWrapper::FromJson(J(json)));
  const std::string serialized = decoder.ToJson().dump();
  EXPECT_EQ(serialized, json);
}

TEST(DecoderWrapperTest, DecoderSerializationNoDecode) {
  std::string json =
      R"({"type":"Sequence","decoders":[{},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"}]})";
  EXPECT_THAT(
      DecoderWrapper::FromJson(J(json)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "data did not match any variant of untagged enum DecoderUntagged"));

  json = R"({"replacement":"▁","prepend_scheme":"always"})";
  EXPECT_THAT(
      DecoderWrapper::FromJson(J(json)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "data did not match any variant of untagged enum DecoderUntagged"));

  json = R"({"type":"Sequence","prepend_scheme":"always"})";
  EXPECT_THAT(
      DecoderWrapper::FromJson(J(json)),
      StatusIs(absl::StatusCode::kInvalidArgument, "missing field `decoders`"));
}

// Additional (not in the Rust file): every variant round-trips through the
// wrapper, the legacy untagged path and `DecodeChain` dispatch.
TEST(DecoderWrapperTest, AllVariantsRoundTrip) {
  const std::vector<std::string> jsons = {
      R"({"type":"BPEDecoder","suffix":"</w>"})",
      R"({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true})",
      R"({"type":"WordPiece","prefix":"##","cleanup":true})",
      R"({"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true})",
      R"({"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":true})",
      R"({"type":"Sequence","decoders":[]})",
      R"({"type":"Replace","pattern":{"String":"▁"},"content":" "})",
      R"({"type":"Fuse"})",
      R"({"type":"Strip","content":"H","start":1,"stop":0})",
      R"({"type":"ByteFallback"})",
  };
  for (const std::string& json : jsons) {
    ASSERT_OK_AND_ASSIGN(DecoderWrapper decoder,
                         DecoderWrapper::FromJson(J(json)));
    EXPECT_EQ(decoder.ToJson().dump(), json);
  }
  // Variant order of `std::variant` is the Rust declaration order.
  ASSERT_OK_AND_ASSIGN(DecoderWrapper decoder,
                       DecoderWrapper::FromJson(J(jsons[6])));
  EXPECT_TRUE(decoder.Is<Replace>());
  EXPECT_EQ(decoder.variant().index(), 6);
}

TEST(DecoderWrapperTest, LegacyUntagged) {
  // Plain `#[serde(tag = "type")]` structs do not validate their tag, so the
  // untagged fallback accepts them without one, in declaration order.
  ASSERT_OK_AND_ASSIGN(DecoderWrapper bpe,
                       DecoderWrapper::FromJson(J(R"({"suffix":"</w>"})")));
  EXPECT_TRUE(bpe.Is<BPEDecoder>());
  ASSERT_OK_AND_ASSIGN(
      DecoderWrapper wordpiece,
      DecoderWrapper::FromJson(J(R"({"prefix":"##","cleanup":true})")));
  EXPECT_TRUE(wordpiece.Is<WordPiece>());
  ASSERT_OK_AND_ASSIGN(
      DecoderWrapper ctc,
      DecoderWrapper::FromJson(J(
          R"({"pad_token":"<pad>","word_delimiter_token":"|","cleanup":true})")));
  EXPECT_TRUE(ctc.Is<CTC>());
  ASSERT_OK_AND_ASSIGN(
      DecoderWrapper strip,
      DecoderWrapper::FromJson(J(R"({"content":"H","start":1,"stop":0})")));
  EXPECT_TRUE(strip.Is<Strip>());
  // An unknown `type` also takes the untagged path.
  ASSERT_OK_AND_ASSIGN(
      DecoderWrapper unknown,
      DecoderWrapper::FromJson(J(R"({"type":"Foo","suffix":"</w>"})")));
  EXPECT_TRUE(unknown.Is<BPEDecoder>());
  // Types whose tag is validated are not accepted without it.
  EXPECT_THAT(
      DecoderWrapper::FromJson(J(R"({"decoders":[]})")),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "data did not match any variant of untagged enum DecoderUntagged"));
  EXPECT_THAT(
      DecoderWrapper::FromJson(J(R"("Fuse")")),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "data did not match any variant of untagged enum DecoderUntagged"));
}

TEST(DecoderWrapperTest, DecodeChainDispatch) {
  ASSERT_OK_AND_ASSIGN(
      DecoderWrapper decoder,
      DecoderWrapper::FromJson(J(
          R"({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]})")));
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       decoder.DecodeChain({"<0x61>", "b", "<0x63>"}));
  EXPECT_THAT(res, ElementsAre("abc"));
  ASSERT_OK_AND_ASSIGN(std::string decoded, decoder.Decode({"a", "b"}));
  EXPECT_EQ(decoded, "ab");
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
