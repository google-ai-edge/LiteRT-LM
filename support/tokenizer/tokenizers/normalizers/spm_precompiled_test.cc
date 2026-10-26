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
// C++ port of `src/tests.rs` from the `spm_precompiled` crate v0.1.4
// (https://github.com/huggingface/spm_precompiled, Apache License 2.0).
//
// The crate's tests embed the `nmt_nfkc` precompiled charsmap of
// SentencePiece as a 237539-byte literal (`fn nmt_nfkc() -> Vec<u8>`) and
// deserialize the same bytes from `test.json`. The identical charsmap is the
// `Precompiled` normalizer of testdata/albert-base-v1-tokenizer.json, which is
// what these tests load instead.

#include "support/tokenizer/tokenizers/normalizers/spm_precompiled.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/normalizers.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::normalizers {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;

constexpr absl::string_view kAlbertTokenizerPath =
    "support/tokenizer/tokenizers/testdata/"
    "albert-base-v1-tokenizer.json";

// The JSON object `{"type":"Precompiled","precompiled_charsmap":"..."}` of
// the albert tokenizer (= the crate's `test.json`).
const Json& PrecompiledJson() {
  static const Json* const kJson = [] {
    std::ifstream input(
        absl::StrCat(::testing::SrcDir(), "/", kAlbertTokenizerPath),
        std::ios::binary);
    std::string contents((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    absl::StatusOr<Json> json = ParseJson(contents);
    CHECK_OK(json.status()) << kAlbertTokenizerPath;
    const Json* precompiled = nullptr;
    for (const Json& normalizer : (*json)["normalizer"]["normalizers"]) {
      if (normalizer["type"] == "Precompiled") precompiled = &normalizer;
    }
    CHECK(precompiled != nullptr)
        << "No Precompiled normalizer in " << kAlbertTokenizerPath;
    return new Json(*precompiled);
  }();
  return *kJson;
}

// `fn nmt_nfkc() -> Vec<u8>`
std::string NmtNfkc() {
  std::string bytes;
  CHECK(absl::Base64Unescape(
      PrecompiledJson()["precompiled_charsmap"].get<std::string>(), &bytes));
  CHECK_EQ(bytes.size(), 237539);
  return bytes;
}

TEST(SpmPrecompiledTest, TestLoadPrecompiledMap) {
  ASSERT_OK_AND_ASSIGN(Precompiled precompiled, Precompiled::From(NmtNfkc()));
  std::vector<int64_t> results =
      precompiled.trie().CommonPrefixSearch("\ufb01");
  EXPECT_THAT(results, ElementsAre(2130));
  // Check the null termination
  EXPECT_EQ(precompiled.normalized().substr(2130, 3),
            absl::string_view("fi\0", 3));

  results = precompiled.trie().CommonPrefixSearch(" ");
  EXPECT_THAT(results, IsEmpty());

  results = precompiled.trie().CommonPrefixSearch("𝔾");
  EXPECT_THAT(results, ElementsAre(1786));
  EXPECT_EQ(precompiled.normalized().substr(1786, 2),
            absl::string_view("G\0", 2));

  EXPECT_THAT(precompiled.Transform("𝔾"), Optional(absl::string_view("G")));
  EXPECT_THAT(precompiled.Transform("𝕠"), Optional(absl::string_view("o")));
  EXPECT_THAT(precompiled.Transform("\u200d"),
              Optional(absl::string_view(" ")));
}

TEST(SpmPrecompiledTest, TestPrecompiledFailureMode) {
  ASSERT_OK_AND_ASSIGN(Precompiled precompiled, Precompiled::From(NmtNfkc()));
  const std::string original = "เขาไม่ได้พูดสักคำ";
  const std::string normalized = "เขาไม\u0e48ได\u0e49พ\u0e39ดส\u0e31กค\u0e4dา";
  EXPECT_EQ(precompiled.NormalizeString(original), normalized);
}

TEST(SpmPrecompiledTest, TestPrecompiledHindi) {
  ASSERT_OK_AND_ASSIGN(Precompiled precompiled, Precompiled::From(NmtNfkc()));
  const std::string original = "ड़ी दुख";
  const std::string normalized = "ड\u093cी द\u0941ख";
  EXPECT_EQ(precompiled.NormalizeString(original), normalized);
}

TEST(SpmPrecompiledTest, TestPrecompiledMultiCharReplaceBug) {
  ASSERT_OK_AND_ASSIGN(Precompiled precompiled, Precompiled::From(NmtNfkc()));
  // آپ
  const std::string original_bytes = {'\xd8', '\xa7', '\xd9', '\x93'};
  std::vector<int64_t> results =
      precompiled.trie().CommonPrefixSearch(original_bytes);
  EXPECT_THAT(results, ElementsAre(4050));
  const std::string& original = original_bytes;
  // This grapheme is actually 2 chars.
  const std::string normalized = "آ";

  EXPECT_EQ(precompiled.normalized().substr(4050, 3),
            absl::string_view("آ\0", 3));
  EXPECT_EQ(precompiled.NormalizeString(original), normalized);
}

TEST(SpmPrecompiledTest, TestSerialization) {
  ASSERT_OK_AND_ASSIGN(Precompiled precompiled, Precompiled::From(NmtNfkc()));

  const std::string string = precompiled.ToJson().dump();
  ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(string));
  ASSERT_OK_AND_ASSIGN(Precompiled reconstructed,
                       Precompiled::FromJson(parsed));

  EXPECT_EQ(reconstructed, precompiled);

  // The Rust test compares against the 316 KB literal
  // `{"type":"Precompiled","precompiled_charsmap":"ALQCAACEAAAAAACAAQAAgM..."}`,
  // i.e. the base64 of `nmt_nfkc()`, which is the string in the test data.
  EXPECT_EQ(string,
            absl::StrCat(R"({"type":"Precompiled","precompiled_charsmap":")",
                         PrecompiledJson()["precompiled_charsmap"]
                             .get_ref<const std::string&>(),
                         R"("})"));

  // `let string = std::fs::read_to_string("test.json")`;
  // `let _reconstructed2: Precompiled = serde_json::from_str(&string)`.
  ASSERT_OK_AND_ASSIGN(Precompiled reconstructed2,
                       Precompiled::FromJson(PrecompiledJson()));
  EXPECT_EQ(reconstructed2, precompiled);
  // ... and through the `NormalizerWrapper` tagged path.
  ASSERT_OK_AND_ASSIGN(NormalizerWrapper wrapper,
                       NormalizerWrapper::FromJson(PrecompiledJson()));
  ASSERT_TRUE(wrapper.Is<Precompiled>());
  EXPECT_EQ(wrapper.Get<Precompiled>(), precompiled);
}

}  // namespace
}  // namespace tokenizers::normalizers
