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
// C++ port of `src/models/bpe/model.rs` (`#[cfg(test)] mod tests`) from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/bpe/model.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::bpe {
namespace {

std::string WriteTempFile(absl::string_view filename,
                          absl::string_view contents) {
  std::string path = absl::StrCat(testing::TempDir(), "/", filename);
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  out.write(contents.data(), contents.size());
  out.close();
  return path;
}

TEST(BpeModelTest, TestCacheIsPerBpeInstance) {
  // Two BPE instances with different merges must tokenize the same
  // input differently even when they share a thread, i.e. the BPE
  // thread-local cache must not leak entries across instances.
  const Vocab vocab_a = {
      {"h", 0u}, {"e", 1},   {"l", 2},    {"o", 3},
      {"he", 4}, {"hel", 5}, {"hell", 6}, {"hello", 7},
  };
  const Merges merges_a = {
      {"h", "e"},
      {"he", "l"},
      {"hel", "l"},
      {"hell", "o"},
  };
  ASSERT_OK_AND_ASSIGN(BPE bpe_a,
                       BpeBuilder().VocabAndMerges(vocab_a, merges_a).Build());

  const Vocab vocab_b = {{"h", 0u}, {"e", 1}, {"l", 2}, {"o", 3}};
  ASSERT_OK_AND_ASSIGN(BPE bpe_b,
                       BpeBuilder().VocabAndMerges(vocab_b, {}).Build());

  const auto to_ids = [](const std::vector<Token>& tokens) {
    std::vector<uint32_t> ids;
    ids.reserve(tokens.size());
    for (const Token& t : tokens) {
      ids.push_back(t.id);
    }
    return ids;
  };

  // Interleave the two models so any cross-instance cache pollution
  // is visible on the second lookup.
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tok_a, bpe_a.Tokenize("hello"));
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tok_b, bpe_b.Tokenize("hello"));
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tok_a2, bpe_a.Tokenize("hello"));
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tok_b2, bpe_b.Tokenize("hello"));

  const std::vector<uint32_t> ids_a = to_ids(tok_a);
  const std::vector<uint32_t> ids_b = to_ids(tok_b);
  const std::vector<uint32_t> ids_a2 = to_ids(tok_a2);
  const std::vector<uint32_t> ids_b2 = to_ids(tok_b2);

  EXPECT_EQ(ids_a, (std::vector<uint32_t>{7u}))
      << "bpe_a must merge to [hello]";
  EXPECT_EQ(ids_b, (std::vector<uint32_t>{0u, 1, 2, 2, 3}))
      << "bpe_b has no merges";
  EXPECT_EQ(ids_a2, ids_a) << "bpe_a second call must match first";
  EXPECT_EQ(ids_b2, ids_b) << "bpe_b second call must match first";
}

TEST(BpeModelTest, TestOrderedVocabIter) {
  const VocabR vocab_r = {
      {0, "a"},
      {1, "b"},
      {2, "c"},
      {3, "ab"},
  };
  const OrderedVocabIter order_vocab_iter(vocab_r);
  const std::string serialized = DumpJson(order_vocab_iter.Serialize());
  EXPECT_EQ(serialized, R"({"a":0,"b":1,"c":2,"ab":3})");
}

TEST(BpeModelTest, TestUnkNotFused) {
  const Vocab vocab = {{"<unk>", 0}, {"a", 1}, {"b", 2}};
  ASSERT_OK_AND_ASSIGN(
      BPE bpe,
      BpeBuilder().VocabAndMerges(vocab, {}).UnkToken("<unk>").Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, bpe.Tokenize("c"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, "<unk>", {0, 1}}}));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("cc"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{0u, "<unk>", {0, 1}},
                        Token{0u, "<unk>", {1, 2}},
                    }));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("accb"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{1u, "a", {0, 1}},
                        Token{0u, "<unk>", {1, 2}},
                        Token{0u, "<unk>", {2, 3}},
                        Token{2u, "b", {3, 4}},
                    }));
}

TEST(BpeModelTest, TestUnkGetFused) {
  const Vocab vocab = {{"<unk>", 0}, {"a", 1}, {"b", 2}};
  ASSERT_OK_AND_ASSIGN(BPE bpe, BpeBuilder()
                                    .VocabAndMerges(vocab, {})
                                    .UnkToken("<unk>")
                                    .FuseUnk(true)
                                    .Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, bpe.Tokenize("c"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, "<unk>", {0, 1}}}));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("cc"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, "<unk>", {0, 2}}}));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("accb"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{1u, "a", {0, 1}},
                        Token{0u, "<unk>", {1, 3}},
                        Token{2u, "b", {3, 4}},
                    }));
}

// Test tokenization. With dropout set to 0 tokenization is deterministic,
// so we know exactly what the result should be.
//
// To test this, we'll build a simple model to tokenize the word 'unrelated'.
TEST(BpeModelTest, TestTokenizeWithAndWithoutDropout) {
  const Vocab vocab = {
      {"u", 0},     {"n", 1},    {"r", 2},        {"e", 3},
      {"l", 4},     {"a", 5},    {"t", 6},        {"d", 7},
      {"re", 8},    {"at", 9},   {"ed", 10},      {"un", 11},
      {"ated", 12}, {"rel", 13}, {"related", 14}, {"unrelated", 15},
  };
  const Merges merges = {
      {"r", "e"},   {"a", "t"},  {"e", "d"},      {"u", "n"},
      {"at", "ed"}, {"re", "l"}, {"rel", "ated"}, {"un", "related"},
  };
  ASSERT_OK_AND_ASSIGN(BPE bpe, BPE::New(vocab, merges));

  // With no dropout:
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, bpe.Tokenize("unrelated"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{15u, "unrelated", {0, 9}}}));

  // With dropout = 0.0 (equivalent to dropout == none)
  bpe.set_dropout(0.0f);
  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("unrelated"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{15u, "unrelated", {0, 9}}}));

  // Now set dropout to 1.0. Result should be no merges performed.
  bpe.set_dropout(1.0f);
  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("unrelated"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{0u, "u", {0, 1}},
                        Token{1u, "n", {1, 2}},
                        Token{2u, "r", {2, 3}},
                        Token{3u, "e", {3, 4}},
                        Token{4u, "l", {4, 5}},
                        Token{5u, "a", {5, 6}},
                        Token{6u, "t", {6, 7}},
                        Token{3u, "e", {7, 8}},
                        Token{7u, "d", {8, 9}},
                    }));

  // Now try with dropout between 0 and 1.
  bpe.set_dropout(0.5f);
  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("unrelated"));
  EXPECT_TRUE(!tokens.empty() && tokens.size() <= 9);
}

// Ensure `BPE::from_file` works as expected.
TEST(BpeModelTest, TestBpeFromFile) {
  const std::string vocab_path = WriteTempFile(
      "test_bpe_from_file_vocab.json", R"({"a": 0, "b": 1, "c": 2, "ab": 3})");
  const std::string merges_path =
      WriteTempFile("test_bpe_from_file_merges.txt", "#version: 0.2\na b");

  BpeBuilder builder = BPE::FromFile(vocab_path, merges_path);
  ASSERT_OK_AND_ASSIGN(BPE bpe, builder.Build());

  // Check merges.
  auto merge_it = bpe.merges().find(Pair(0, 1));
  ASSERT_NE(merge_it, bpe.merges().end());
  EXPECT_EQ(merge_it->second, (std::pair<uint32_t, uint32_t>{0u, 3u}));

  // Check vocab.
  EXPECT_EQ(bpe.vocab().at("a"), 0u);
  EXPECT_EQ(bpe.vocab().at("b"), 1u);
  EXPECT_EQ(bpe.vocab().at("c"), 2u);
  EXPECT_EQ(bpe.vocab().at("ab"), 3u);
}

// Ensure BPEBuilder with dropout = 0.0 doesn't error
TEST(BpeModelTest, TestBpeWithDropout0) {
  ASSERT_OK_AND_ASSIGN(BPE bpe, BPE::Builder().Dropout(0.0f).Build());
  EXPECT_EQ(bpe.dropout(), std::optional<float>(0.0f));
}

TEST(BpeModelTest, TestBpeWithContinuingSubwordPrefix) {
  const Vocab vocab = {
      {"a", 0}, {"##b", 1}, {"##c", 2}, {"ab", 3}, {"abc", 4},
  };
  const Merges merges = {
      {"a", "##b"},
      {"ab", "##c"},
  };

  ASSERT_OK_AND_ASSIGN(BPE bpe, BPE::Builder()
                                    .VocabAndMerges(vocab, merges)
                                    .UnkToken("[UNK]")
                                    .ContinuingSubwordPrefix("##")
                                    .Build());

  ASSERT_OK_AND_ASSIGN(std::vector<Token> res, bpe.Tokenize("ab"));
  EXPECT_EQ(res, (std::vector<Token>{Token{3, "ab", {0, 2}}}));

  ASSERT_OK_AND_ASSIGN(res, bpe.Tokenize("abc"));
  EXPECT_EQ(res, (std::vector<Token>{Token{4, "abc", {0, 3}}}));
}

// Ensure `MergeTokenOutOfVocabulary` error is returned when it should be.
TEST(BpeModelTest, TestBpeFromFileMergeTokenOov) {
  const std::string vocab_path = WriteTempFile(
      "test_bpe_oov_vocab.json", R"({"a": 0, "b": 1, "c": 2, "ab": 3})");
  const std::string merges_path =
      WriteTempFile("test_bpe_oov_merges.txt", "#version: 0.2\na b\na d");

  const absl::StatusOr<BPE> res =
      BPE::FromFile(vocab_path, merges_path).Build();
  ASSERT_FALSE(res.ok());
  EXPECT_EQ(res.status(), Error::MergeTokenOutOfVocabulary("d"));
}

// Ensure `BadMerges` error is returned when there is an invalid line in the
// merges.txt file.
TEST(BpeModelTest, TestBpeFromFileBadMerges) {
  const std::string vocab_path = WriteTempFile(
      "test_bpe_bad_vocab.json", R"({"a": 0, "b": 1, "c": 2, "ab": 3})");
  const std::string merges_path =
      WriteTempFile("test_bpe_bad_merges.txt", "#version: 0.2\na b\nc");

  const absl::StatusOr<BPE> res =
      BPE::FromFile(vocab_path, merges_path).Build();
  ASSERT_FALSE(res.ok());
  EXPECT_EQ(res.status(), Error::BadMerges(2));
}

TEST(BpeModelTest, TestBpeByteFallback) {
  // 0x61 == 'a' in bytes
  const Vocab vocab = {{"<unk>", 0}, {"<0x61>", 1}};
  ASSERT_OK_AND_ASSIGN(BPE bpe, BpeBuilder()
                                    .VocabAndMerges(vocab, {})
                                    .UnkToken("<unk>")
                                    .ByteFallback(true)
                                    .Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, bpe.Tokenize("c"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, "<unk>", {0, 1}}}));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("a"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{1u, "<0x61>", {0, 1}}}));
}

TEST(BpeModelTest, TestBpeByteFallbackNewline) {
  // 0x0A == '\n' in bytes
  const Vocab vocab = {{"<unk>", 0}, {"<0x0A>", 1}};
  ASSERT_OK_AND_ASSIGN(BPE bpe, BpeBuilder()
                                    .VocabAndMerges(vocab, {})
                                    .UnkToken("<unk>")
                                    .ByteFallback(true)
                                    .Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, bpe.Tokenize("\n"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{1u, "<0x0A>", {0, 1}}}));
}

TEST(BpeModelTest, TestIgnoreMerges) {
  const Vocab vocab = {
      {".:.:", 0},       {"Ġbelirtilen", 1}, {".", 2},   {":", 3},
      {"bel", 4},        {"irtilen", 5},     {"Ġ", 6},   {".:", 7},
      {"belirtilen", 8}, {".:.", 9},         {"be", 10}, {"l", 11},
      {"ir", 12},        {"ti", 13},         {"en", 14}, {"irtil", 15},
      {"irti", 16},      {"i", 17},          {"r", 18},  {"t", 19},
      {"b", 20},         {"e", 21},          {"n", 22},
  };
  ASSERT_OK_AND_ASSIGN(BPE bpe, BpeBuilder()
                                    .VocabAndMerges(vocab,
                                                    {
                                                        {".", ":"},
                                                        {"b", "e"},
                                                        {"be", "l"},
                                                        {"i", "r"},
                                                        {"t", "i"},
                                                        {"ir", "ti"},
                                                        {"e", "n"},
                                                        {"irti", "l"},
                                                    })
                                    .IgnoreMerges(true)
                                    .Build());
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, bpe.Tokenize(".:.:"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{0u, ".:.:", {0, 4}}}));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("Ġbelirtilen"));
  EXPECT_EQ(tokens, (std::vector<Token>{Token{1u, "Ġbelirtilen", {0, 12}}}));

  bpe.set_ignore_merges(false);

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize(".:.:"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{7u, ".:", {0, 2}},
                        Token{7u, ".:", {2, 4}},
                    }));

  ASSERT_OK_AND_ASSIGN(tokens, bpe.Tokenize("Ġbelirtilen"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{6, "Ġ", {0, 2}},
                        Token{4, "bel", {2, 5}},
                        Token{15, "irtil", {5, 10}},
                        Token{14, "en", {10, 12}},
                    }));
}

}  // namespace
}  // namespace tokenizers::models::bpe
