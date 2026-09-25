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

#include "omni/tts/kokoro/chinese_g2p.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/tts/kokoro/cjk_blob.h"
#include "omni/tts/kokoro/cjk_test_blobs.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts::kokoro {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

void AppendDouble(double value, std::string& out) {
  char buf[sizeof(double)];
  std::memcpy(buf, &value, sizeof(double));
  out.append(buf, sizeof(double));
}

void PackStringMap(std::vector<std::pair<std::string, std::string>> items,
                   std::string& k_idx, std::string& k_blob, std::string& v_idx,
                   std::string& v_blob) {
  std::sort(items.begin(), items.end());
  for (const auto& [k, v] : items) {
    AppendUint32(static_cast<uint32_t>(k_blob.size()), k_idx);
    k_blob.append(k);
    k_blob.push_back('\0');
    AppendUint32(static_cast<uint32_t>(v_blob.size()), v_idx);
    v_blob.append(v);
    v_blob.push_back('\0');
  }
  AppendUint32(static_cast<uint32_t>(k_blob.size()), k_idx);
  AppendUint32(static_cast<uint32_t>(v_blob.size()), v_idx);
}

std::string BuildSyntheticChineseG2pBlob() {
  std::vector<std::pair<std::string, uint32_t>> words = {
      {"一点", 800},
      {"一万多一点", 500},
      {"今天", 2000},
      {"天气", 1500},
  };
  std::sort(words.begin(), words.end());
  std::string w_idx;
  std::string w_key;
  std::string w_frq;
  for (const auto& [w, f] : words) {
    AppendUint32(static_cast<uint32_t>(w_key.size()), w_idx);
    w_key.append(w);
    w_key.push_back('\0');
    AppendUint32(f, w_frq);
  }
  AppendUint32(static_cast<uint32_t>(w_key.size()), w_idx);
  std::string w_tot;
  AppendDouble(std::log(5000.0), w_tot);

  std::string hmm_st;
  for (double p : {-0.26, -3.14e100, -3.14e100, -1.46}) {
    AppendDouble(p, hmm_st);
  }
  std::string hmm_tr(16 * sizeof(double), '\0');
  std::string hmm_en(4 * sizeof(uint32_t), '\0');

  std::string p_idx;
  std::string p_key;
  std::string p_vidx;
  std::string p_val;
  // "一万多一点" is a jieba word, NOT a pypinyin phrase; "一点" IS a pypinyin
  // phrase where "一" undergoes tone sandhi (`i↘tiɛn↓`).
  PackStringMap(
      {
          {"一点", "i↘tiɛn↓"},
          {"今天", "tɕin→tʰjɛn→"},
      },
      p_idx, p_key, p_vidx, p_val);

  std::string c_idx;
  std::string c_key;
  std::string c_vidx;
  std::string c_val;
  PackStringMap(
      {
          {"一", "i→"},
          {"万", "wan↘"},
          {"多", "twɔ→"},
          {"天", "tʰjɛn→"},
          {"气", "tɕʰi↘"},
          {"点", "tiɛn↓"},
      },
      c_idx, c_key, c_vidx, c_val);

  return BuildCjkBlob({
      {"w-idx", w_idx},
      {"w-key", w_key},
      {"w-frq", w_frq},
      {"w-tot", w_tot},
      {"hmm-st", hmm_st},
      {"hmm-tr", hmm_tr},
      {"hmm-en", hmm_en},
      {"hmm-ec", ""},
      {"hmm-ep", ""},
      {"p-idx", p_idx},
      {"p-key", p_key},
      {"p-vidx", p_vidx},
      {"p-val", p_val},
      {"c-idx", c_idx},
      {"c-key", c_key},
      {"c-vidx", c_vidx},
      {"c-val", c_val},
  });
}

TEST(ChineseG2pTest, UsesMmsegInsideJiebaWordForPhraseSandhi) {
  const std::string data = BuildSyntheticChineseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const ChineseG2p g2p, ChineseG2p::Create(blob));

  // "一万多一点" is a single jieba word, but pypinyin's mmseg cuts it into
  // "一", "万", "多", "一点", picking up the phrase entry for "一点"
  // (`i↘tiɛn↓`) while leaving the leading "一" in first tone (`i→`).
  EXPECT_THAT(g2p.TextToIpa("一万多一点"), IsOkAndHolds("i→wan↘twɔ→i↘tiɛn↓"));
}

TEST(ChineseG2pTest, SegmentsAndTranscribesMultiWordSentences) {
  const std::string data = BuildSyntheticChineseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const ChineseG2p g2p, ChineseG2p::Create(blob));

  EXPECT_THAT(g2p.TextToIpa("今天天气"),
              IsOkAndHolds("tɕin→tʰjɛn→ tʰjɛn→tɕʰi↘"));
}

TEST(ChineseG2pTest, ReportsFalseWhenCharacterIsMissingFromLexicon) {
  const std::string data = BuildSyntheticChineseG2pBlob();
  ASSERT_OK_AND_ASSIGN(const CjkBlob blob, CjkBlob::Create(data));
  ASSERT_OK_AND_ASSIGN(const ChineseG2p g2p, ChineseG2p::Create(blob));

  std::string out;
  EXPECT_FALSE(g2p.WordToIpa("瓧", out));
  EXPECT_THAT(g2p.TextToIpa("瓧"), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
