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
// C++ port of `src/decoders/byte_fallback.rs` (tests) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/byte_fallback.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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

TEST(ByteFallbackTest, Decode) {
  ByteFallback decoder;
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       decoder.DecodeChain({"Hey", "friend!"}));
  EXPECT_THAT(res, ElementsAre("Hey", "friend!"));

  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({"<0x61>"}));
  EXPECT_THAT(res, ElementsAre("a"));

  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({"<0xE5>"}));
  EXPECT_THAT(res, ElementsAre("\xEF\xBF\xBD"));

  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({"<0xE5>", "<0x8f>"}));
  EXPECT_THAT(res, ElementsAre("\xEF\xBF\xBD", "\xEF\xBF\xBD"));

  // 叫
  ASSERT_OK_AND_ASSIGN(res,
                       decoder.DecodeChain({"<0xE5>", "<0x8f>", "<0xab>"}));
  EXPECT_THAT(res, ElementsAre("叫"));

  ASSERT_OK_AND_ASSIGN(
      res, decoder.DecodeChain({"<0xE5>", "<0x8f>", "<0xab>", "a"}));
  EXPECT_THAT(res, ElementsAre("叫", "a"));

  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({"<0xE5>", "<0x8f>", "a"}));
  EXPECT_THAT(res, ElementsAre("\xEF\xBF\xBD", "\xEF\xBF\xBD", "a"));
}

// Additional (not in the Rust file): serde round trip of the `MustBe!` tag.
TEST(ByteFallbackTest, Serde) {
  ByteFallback decoder;
  EXPECT_EQ(decoder.ToJson().dump(), R"({"type":"ByteFallback"})");
  EXPECT_OK(ByteFallback::FromJson(J(R"({"type":"ByteFallback"})")));
  EXPECT_THAT(ByteFallback::FromJson(J(R"({"type":"Fuse"})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "invalid value: string \"Fuse\", expected string "
                       "\"ByteFallback\""));
  EXPECT_THAT(
      ByteFallback::FromJson(J(R"({})")),
      StatusIs(absl::StatusCode::kInvalidArgument, "missing field `type`"));
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
