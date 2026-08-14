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

#include "support/tokenizer/buffered_streaming_detokenizer.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // NOLINT

namespace litert::support {
namespace {

using ::testing::status::IsOkAndHolds;

class MockTokenizer : public Tokenizer {
 public:
  TokenizerType GetTokenizerType() const override {
    return TokenizerType::kUnspecified;
  }

  absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) override {
    return absl::UnimplementedError("");
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    return absl::UnimplementedError("");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) override {
    if (token_ids.empty()) {
      return "";
    }

    std::string result;
    for (size_t i = 0; i < token_ids.size(); ++i) {
      int id = token_ids[i];
      if (id == 1) {
        result += "Hello";
      } else if (id == 2) {
        result += " World";
      } else if (id == 3) {
        result += '!';
      } else if (id == 4) {
        size_t run_len = 0;
        while (i + run_len < token_ids.size() && token_ids[i + run_len] == 4) {
          run_len++;
        }
        if (i + run_len < token_ids.size() && token_ids[i + run_len] == 5) {
          result += "🌟";
          i += run_len;  // skip all 4s and the 5
        } else {
          result += "\xef\xbf\xbd";
        }
      } else {
        result += std::to_string(id);
      }
    }
    return result;
  }

  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 0; }
};

TEST(BufferedStreamingDetokenizerTest, BasicStreaming) {
  MockTokenizer tokenizer;
  BufferedStreamingDetokenizer detokenizer(&tokenizer, /*output_heads=*/1);

  // Step 1: Input [1]
  // Accumulated: [1] (Hello)
  // Prev: empty -> release nothing
  EXPECT_THAT(detokenizer.ProcessStep({{1}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 2: Input [2]
  // Accumulated: [1, 2] (Hello World)
  // Prev: [1] -> decode [1] -> release "Hello"
  EXPECT_THAT(detokenizer.ProcessStep({{2}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"Hello", {1}}}));

  // Step 3: Input [3]
  // Accumulated: [1, 2, 3] (Hello World!)
  // Prev: [1, 2] -> decode [1, 2] -> release " World"
  EXPECT_THAT(detokenizer.ProcessStep({{3}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{" World", {2}}}));

  // Step 4: Flush
  // Accumulated: [1, 2, 3] -> decode [1, 2, 3] -> release "!"
  EXPECT_THAT(detokenizer.Flush(),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"!", {3}}}));
}

TEST(BufferedStreamingDetokenizerTest, IncompleteBpe) {
  MockTokenizer tokenizer;
  BufferedStreamingDetokenizer detokenizer(&tokenizer, /*output_heads=*/1);

  // Step 1: Input [1]
  // Accumulated: [1] (Hello)
  // Prev: empty -> release nothing
  EXPECT_THAT(detokenizer.ProcessStep({{1}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 2: Input [4] (incomplete BPE)
  // Accumulated: [1, 4] (incomplete but decodes to "Hello\xef\xbf\xbd")
  // We compare with last valid "Hello". LCP is "Hello". Release "Hello".
  EXPECT_THAT(detokenizer.ProcessStep({{4}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"Hello", {1}}}));

  // Step 3: Input [5] (completes 4 to "🌟")
  // Accumulated: [1, 4, 5] (valid "Hello🌟")
  // We compare with last valid "Hello". LCP is "Hello". Release nothing.
  EXPECT_THAT(detokenizer.ProcessStep({{5}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 4: Input [3]
  // Accumulated: [1, 4, 5, 3] (valid "Hello🌟!")
  // Prev: [1, 4, 5] -> decode [1, 4, 5] -> "Hello🌟" -> release "🌟"
  EXPECT_THAT(detokenizer.ProcessStep({{3}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"🌟", {4, 5}}}));

  // Step 5: Flush
  // Accumulated: [1, 4, 5, 3] -> decode [1, 4, 5, 3] -> "Hello🌟!" ->
  // release "!"
  EXPECT_THAT(detokenizer.Flush(),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"!", {3}}}));
}

TEST(BufferedStreamingDetokenizerTest, EmojiConsolidation) {
  MockTokenizer tokenizer;
  BufferedStreamingDetokenizer detokenizer(&tokenizer, /*output_heads=*/1);

  // Step 1: Input [1] -> "Hello"
  // Released: "" (lag)
  EXPECT_THAT(detokenizer.ProcessStep({{1}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 2: Input [4] -> "Hello\xef\xbf\xbd"
  // Decoded: "Hello\xef\xbf\xbd", trimmed: "Hello"
  // LCP(Hello, Hello) = 5
  // Released: "Hello"
  EXPECT_THAT(detokenizer.ProcessStep({{4}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"Hello", {1}}}));

  // Step 3: Input [4] -> "Hello\xef\xbf\xbd\xef\xbf\xbd"
  // Decoded: "Hello\xef\xbf\xbd\xef\xbf\xbd", trimmed: "Hello"
  // LCP(Hello\xef\xbf\xbd, Hello) = 5.
  // Released: "" (since released_len is 5)
  EXPECT_THAT(detokenizer.ProcessStep({{4}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 4: Input [5] -> "Hello🌟"
  // Decoded: "Hello🌟", trimmed: "Hello🌟"
  // LCP(Hello\xef\xbf\xbd\xef\xbf\xbd, Hello🌟) = 5
  // Released: "" (released_len: 5)
  EXPECT_THAT(detokenizer.ProcessStep({{5}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 5: Input [3] -> "Hello🌟!"
  // Decoded: "Hello🌟!", trimmed: "Hello🌟!"
  // LCP(Hello🌟, Hello🌟!) = 9 (Hello🌟)
  // Released: "🌟" (released_len becomes 9)
  EXPECT_THAT(detokenizer.ProcessStep({{3}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"🌟", {4, 4, 5}}}));

  // Step 6: Flush
  // Released: "!"
  EXPECT_THAT(detokenizer.Flush(),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"!", {3}}}));
}

TEST(BufferedStreamingDetokenizerTest, BatchStreaming) {
  MockTokenizer tokenizer;
  BufferedStreamingDetokenizer detokenizer(&tokenizer, /*output_heads=*/2);

  // Step 1: Input H0: [1], H1: [2]
  EXPECT_THAT(detokenizer.ProcessStep({{1}, {2}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}, {"", {}}}));

  // Step 2: Input H0: [2], H1: [3]
  // H0 Prev: [1] -> "Hello"
  // H1 Prev: [2] -> " World"
  EXPECT_THAT(detokenizer.ProcessStep({{2}, {3}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"Hello", {1}},
                                                        {" World", {2}}}));

  // Step 3: Flush
  // H0 Accumulated: [1, 2] -> "Hello World" -> release " World"
  // H1 Accumulated: [2, 3] -> " World!" -> release "!"
  EXPECT_THAT(detokenizer.Flush(), IsOkAndHolds(std::vector<DetokenizedStep>{
                                       {" World", {2}}, {"!", {3}}}));
}

TEST(BufferedStreamingDetokenizerTest, LongStreamingWithPruning) {
  MockTokenizer tokenizer;
  // Use lookback_tokens = 2 to trigger pruning early.
  BufferedStreamingDetokenizer detokenizer(&tokenizer, /*output_heads=*/1,
                                           /*lookback_tokens=*/2);

  // Step 1: Input [1] -> "Hello" (lag by 1 step)
  EXPECT_THAT(detokenizer.ProcessStep({{1}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"", {}}}));

  // Step 2: Input [2] -> " World"
  // Released: "Hello", token [1]
  EXPECT_THAT(detokenizer.ProcessStep({{2}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"Hello", {1}}}));

  // Step 3: Input [3] -> "!"
  // Released: " World", token [2]
  EXPECT_THAT(detokenizer.ProcessStep({{3}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{" World", {2}}}));

  // Step 4: Input [2] -> " World"
  // Released: "!", token [3] (pruning triggered as released_token_indices > 2)
  EXPECT_THAT(detokenizer.ProcessStep({{2}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"!", {3}}}));

  // Step 5: Input [3] -> "!"
  // Released: " World", token [2]
  EXPECT_THAT(detokenizer.ProcessStep({{3}}),
              IsOkAndHolds(std::vector<DetokenizedStep>{{" World", {2}}}));

  // Step 6: Flush -> release "!" with token [3]
  EXPECT_THAT(detokenizer.Flush(),
              IsOkAndHolds(std::vector<DetokenizedStep>{{"!", {3}}}));
}

}  // namespace
}  // namespace litert::support
