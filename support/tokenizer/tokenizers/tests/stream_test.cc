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
// C++ port of `tests/stream.rs` and the `DecodeStream` doc-tests in
// `src/tokenizer/mod.rs` (lines 970-1060) from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/normalizers/byte_level.h"
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/normalizers/unicode.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/split.h"
#include "support/tokenizer/tokenizers/tests/common.h"
#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::decoders::ByteFallback;
using ::tokenizers::models::bpe::BPE;
using ::tokenizers::normalizers::NFC;
using ::tokenizers::normalizers::NormalizerWrapper;
using ::tokenizers::pre_tokenizers::Metaspace;
using ::tokenizers::pre_tokenizers::PreTokenizerWrapper;
using ::tokenizers::pre_tokenizers::Split;
using ::tokenizers::pre_tokenizers::SplitPattern;
using ::tokenizers::test_common::TestdataPath;

TEST(StreamTest, DecodingWithAddedBpe) {
  std::string llama3_path = TestdataPath("llama-3-tokenizer.json");
  if (!std::filesystem::exists(llama3_path)) {
    GTEST_SKIP() << "llama-3-tokenizer.json is excluded from the OSS mirror.";
  }
  ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer, Tokenizer::FromFile(llama3_path));
  ASSERT_OK(
      tokenizer.WithNormalizer(NormalizerWrapper(normalizers::ByteLevel())));
  ASSERT_OK_AND_ASSIGN(
      Split split,
      Split::New(
          SplitPattern::Regex(
              R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)"),
          SplitDelimiterBehavior::kIsolated, false));
  tokenizer.WithPreTokenizer(PreTokenizerWrapper(std::move(split)));
  ASSERT_OK(
      tokenizer.AddTokens({AddedToken::From("嗎", false).Normalized(false)})
          .status());
  ASSERT_OK_AND_ASSIGN(Encoding encoded,
                       tokenizer.Encode("Hey! how is this token: 嗎", false));
  EXPECT_EQ(encoded.GetIds(),
            (std::vector<uint32_t>{19182, 0, 1268, 602, 82, 62428, 82, 4037, 25,
                                   220, 128256}));
  EXPECT_EQ(encoded.GetTokens(),
            (std::vector<std::string>{"Hey", "!", "Ġhow", "Ġi", "s", "Ġthi",
                                      "s", "Ġtoken", ":", "Ġ", "嗎"}));

  ASSERT_OK_AND_ASSIGN(std::string decoded,
                       tokenizer.Decode(encoded.GetIds(), false));
  EXPECT_EQ(decoded, "Hey! how is this token: 嗎");

  ASSERT_OK(tokenizer.AddTokens({AddedToken::From("д", false).Normalized(true)})
                .status());
  ASSERT_OK_AND_ASSIGN(Encoding encoded2,
                       tokenizer.Encode("Hey! how is this token: д", false));
  EXPECT_EQ(encoded2.GetIds(),
            (std::vector<uint32_t>{19182, 0, 1268, 602, 82, 62428, 82, 4037, 25,
                                   220, 128257}));
  EXPECT_EQ(encoded2.GetTokens(),
            (std::vector<std::string>{"Hey", "!", "Ġhow", "Ġi", "s", "Ġthi",
                                      "s", "Ġtoken", ":", "Ġ", "Ð´"}));
  ASSERT_OK_AND_ASSIGN(std::string decoded2,
                       tokenizer.Decode(encoded2.GetIds(), false));
  EXPECT_EQ(decoded2, "Hey! how is this token: д");
}

TEST(StreamTest, DecodeStreamStepNoPanic) {
  std::string llama3_path = TestdataPath("llama-3-tokenizer.json");
  if (!std::filesystem::exists(llama3_path)) {
    GTEST_SKIP() << "llama-3-tokenizer.json is excluded from the OSS mirror.";
  }
  ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer, Tokenizer::FromFile(llama3_path));

  // "A B C D E F G H I J"
  DecodeStream decode_stream = tokenizer.DecodeStream(false);
  EXPECT_EQ(decode_stream.Step(32).value(), std::optional<std::string>("A"));
  EXPECT_EQ(decode_stream.Step(426).value(), std::optional<std::string>(" B"));
  EXPECT_EQ(decode_stream.Step(356).value(), std::optional<std::string>(" C"));
  EXPECT_EQ(decode_stream.Step(423).value(), std::optional<std::string>(" D"));
  EXPECT_EQ(decode_stream.Step(469).value(), std::optional<std::string>(" E"));
  EXPECT_EQ(decode_stream.Step(435).value(), std::optional<std::string>(" F"));
  EXPECT_EQ(decode_stream.Step(480).value(), std::optional<std::string>(" G"));
  EXPECT_EQ(decode_stream.Step(473).value(), std::optional<std::string>(" H"));
  EXPECT_EQ(decode_stream.Step(358).value(), std::optional<std::string>(" I"));
  EXPECT_EQ(decode_stream.Step(622).value(), std::optional<std::string>(" J"));

  // "삥뽕빵" (Korean words composed of 2-3 tokens: [80690, 98], [167, 121,
  // 243], and [102457, 113])
  DecodeStream decode_stream2 = tokenizer.DecodeStream(false);
  EXPECT_EQ(decode_stream2.Step(80690).value(), std::nullopt);
  EXPECT_EQ(decode_stream2.Step(98).value(), std::optional<std::string>("삥"));
  EXPECT_EQ(decode_stream2.Step(167).value(), std::nullopt);
  EXPECT_EQ(decode_stream2.Step(121).value(), std::nullopt);
  EXPECT_EQ(decode_stream2.Step(243).value(), std::optional<std::string>("뽕"));
  EXPECT_EQ(decode_stream2.Step(102457).value(), std::nullopt);
  EXPECT_EQ(decode_stream2.Step(113).value(), std::optional<std::string>("빵"));
}

// Doc-tests on `DecodeStream` in `src/tokenizer/mod.rs` (lines 970-1060).
TEST(StreamTest, DecodeStreamDocTests) {
  // Doc-test 1: roberta.json
  {
    ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer,
                         Tokenizer::FromFile(TestdataPath("roberta.json")));
    DecodeStream decode_stream = tokenizer.DecodeStream(false);
    EXPECT_EQ(decode_stream.Step(713).value(),
              std::optional<std::string>("This"));
    EXPECT_EQ(decode_stream.Step(16).value(),
              std::optional<std::string>(" is"));
    EXPECT_EQ(decode_stream.Step(41).value(),
              std::optional<std::string>(" an"));
    EXPECT_EQ(decode_stream.Step(1246).value(),
              std::optional<std::string>(" example"));
  }

  // Doc-test 2: byte_fallback returning None on partial UTF-8
  {
    absl::flat_hash_map<std::string, uint32_t> vocab = {
        {"<0x20>", 0},
        {"<0xC3>", 1},
        {"<0xA9>", 2},
        {" This", 3},
    };
    ASSERT_OK_AND_ASSIGN(BPE bpe, BPE::Builder()
                                      .VocabAndMerges(std::move(vocab), {})
                                      .ByteFallback(true)
                                      .Build());
    ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer,
                         TokenizerBuilder()
                             .WithModel(std::move(bpe))
                             .WithDecoder(ByteFallback())
                             .WithNormalizer(NFC())
                             .WithPreTokenizer(pre_tokenizers::ByteLevel())
                             .WithPostProcessor(pre_tokenizers::ByteLevel())
                             .Build());

    DecodeStream decode_stream = tokenizer.DecodeStream(false);
    // Single byte_fallback is valid utf-8
    EXPECT_EQ(decode_stream.Step(0).value(), std::optional<std::string>(" "));
    // Invalid utf-8
    EXPECT_EQ(decode_stream.Step(1).value(), std::nullopt);
    // Valid utf-8 again, this corresponds to both tokens: [1, 2]
    EXPECT_EQ(decode_stream.Step(2).value(), std::optional<std::string>("é"));
  }

  // Doc-test 3: Metaspace initial space handling
  {
    absl::flat_hash_map<std::string, uint32_t> vocab = {
        {"▁This", 0},
    };
    ASSERT_OK_AND_ASSIGN(BPE bpe, BPE::Builder()
                                      .VocabAndMerges(std::move(vocab), {})
                                      .ByteFallback(true)
                                      .Build());
    ASSERT_OK_AND_ASSIGN(Tokenizer tokenizer,
                         TokenizerBuilder()
                             .WithModel(std::move(bpe))
                             .WithDecoder(Metaspace())
                             .WithNormalizer(NFC())
                             .WithPreTokenizer(pre_tokenizers::ByteLevel())
                             .WithPostProcessor(pre_tokenizers::ByteLevel())
                             .Build());

    // Strip decoder removes the extra initial space
    EXPECT_EQ(tokenizer.Decode({0, 0}, false).value(), "This This");
    // Decoding one token at a time would produce "ThisThis"
    EXPECT_EQ(tokenizer.Decode({0}, false).value(), "This");

    // Using a stream fixes it by keeping the necessary state.
    DecodeStream decode_stream = tokenizer.DecodeStream(false);
    EXPECT_EQ(decode_stream.Step(0).value(),
              std::optional<std::string>("This"));
    EXPECT_EQ(decode_stream.Step(0).value(),
              std::optional<std::string>(" This"));
  }
}

}  // namespace
}  // namespace tokenizers
