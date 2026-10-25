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
// C++ port of the tests of `src/normalizers/strip.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/normalizers/strip.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/unicode.h"
#include "support/tokenizer/tokenizers/normalizers/utils.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/unicode/normalization.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::normalizers {
namespace {

using ::testing::ElementsAreArray;

// `s.nfkd().map(|(c, _)| c).collect::<String>()`
std::string Nfkd(absl::string_view s) {
  std::string result;
  for (const auto& [c, change] : unicode::Nfkd(s)) {
    AppendUtf8(c, &result);
  }
  return result;
}

TEST(StripTest, TestStripAccents) {
  // Unicode combining char
  std::string original = Nfkd("Me llamó");
  std::string normalized = "Me llamo";
  ASSERT_NE(original, normalized);
  NormalizedString n = NormalizedString::From(original);
  ASSERT_OK(StripAccents().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);

  // Ignores regular ascii
  original = "Me llamo";
  normalized = "Me llamo";
  ASSERT_EQ(original, normalized);
  n = NormalizedString::From(original);
  ASSERT_OK(StripAccents().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);

  // Does not change chinese
  original = Nfkd("这很简单");
  normalized = "这很简单";
  ASSERT_EQ(original, normalized);
  n = NormalizedString::From(original);
  ASSERT_OK(StripAccents().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
}

TEST(StripTest, TestVietnameseBug) {
  std::string original = "ậ…";
  std::string normalized = "a...";
  ASSERT_NE(original, normalized);
  NormalizedString n = NormalizedString::From(original);
  ASSERT_OK(NFKD().Normalize(n));
  ASSERT_OK(StripAccents().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
  ASSERT_OK(Lowercase().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);

  original =
      "Cụ thể, bạn sẽ tham gia một nhóm các giám đốc điều hành tổ chức, các "
      "nhà lãnh đạo doanh nghiệp, các học giả, chuyên gia phát triển và tình "
      "nguyện viên riêng biệt trong lĩnh vực phi lợi nhuận…";
  normalized =
      "cu the, ban se tham gia mot nhom cac giam đoc đieu hanh to chuc, cac "
      "nha lanh đao doanh nghiep, cac hoc gia, chuyen gia phat trien va tinh "
      "nguyen vien rieng biet trong linh vuc phi loi nhuan...";
  n = NormalizedString::From(original);
  ASSERT_OK(NFKD().Normalize(n));
  ASSERT_OK(StripAccents().Normalize(n));
  ASSERT_OK(Lowercase().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
}

TEST(StripTest, TestThaiBug) {
  const std::string original = "ำน\u0e49ำ3ลำ";
  const std::string normalized = "านา3ลา";
  ASSERT_NE(original, normalized);
  NormalizedString n = NormalizedString::From(original);
  ASSERT_OK(NFKD().Normalize(n));
  ASSERT_OK(StripAccents().Normalize(n));
  ASSERT_OK(Lowercase().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
}

TEST(StripTest, TestStripAccentsMultiple) {
  const std::string original = "e\u0304\u0304\u0304o";
  const std::string normalized = "eo";
  ASSERT_NE(original, normalized);
  NormalizedString n = NormalizedString::From(original);
  ASSERT_OK(StripAccents().Normalize(n));
  EXPECT_EQ(n.Get(), normalized);
  EXPECT_EQ(n,
            NormalizedString::New(original, normalized, {{0, 1}, {7, 8}}, 0));
  EXPECT_THAT(
      n.AlignmentsOriginal(),
      ElementsAreArray(std::vector<std::pair<size_t, size_t>>{
          {0, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 2}}));
}

}  // namespace
}  // namespace tokenizers::normalizers
