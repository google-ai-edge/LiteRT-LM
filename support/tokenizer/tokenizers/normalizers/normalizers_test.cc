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
// C++ port of the tests of `src/normalizers/mod.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/normalizers/normalizers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/normalizers/prepend.h"
#include "support/tokenizer/tokenizers/normalizers/strip.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::normalizers {
namespace {

using ::testing::status::StatusIs;

// `serde_json::from_str::<NormalizerWrapper>(json)`
absl::StatusOr<NormalizerWrapper> FromStr(absl::string_view json) {
  ASSIGN_OR_RETURN(Json value, ParseJson(json));
  return NormalizerWrapper::FromJson(value);
}

TEST(NormalizerWrapperTest, PostProcessorDeserializationNoType) {
  absl::string_view json = R"({"strip_left":false, "strip_right":true})";
  ASSERT_OK_AND_ASSIGN(NormalizerWrapper reconstructed, FromStr(json));
  EXPECT_TRUE(reconstructed.Is<Strip>());

  json = R"({"trim_offsets":true, "add_prefix_space":true})";
  EXPECT_THAT(FromStr(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "data did not match any variant of untagged enum "
                       "NormalizerUntagged"));

  json = R"({"prepend":"a"})";
  ASSERT_OK_AND_ASSIGN(reconstructed, FromStr(json));
  EXPECT_TRUE(reconstructed.Is<Prepend>());
}

TEST(NormalizerWrapperTest, NormalizerSerialization) {
  absl::string_view json = R"({"type":"Sequence","normalizers":[]})";
  EXPECT_OK(FromStr(json));
  json = R"({"type":"Sequence","normalizers":[{}]})";
  EXPECT_THAT(FromStr(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "data did not match any variant of untagged enum "
                       "NormalizerUntagged"));

  json = R"({"replacement":"▁","prepend_scheme":"always"})";
  EXPECT_THAT(FromStr(json),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "data did not match any variant of untagged enum "
                       "NormalizerUntagged"));

  json = R"({"type":"Sequence","prepend_scheme":"always"})";
  EXPECT_THAT(FromStr(json), StatusIs(absl::StatusCode::kInvalidArgument,
                                      "missing field `normalizers`"));
}

}  // namespace
}  // namespace tokenizers::normalizers
