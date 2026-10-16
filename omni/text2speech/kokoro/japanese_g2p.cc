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

#include "omni/text2speech/kokoro/japanese_g2p.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/cjk_lattice.h"
#include "omni/text2speech/kokoro/mecab_lattice.h"
#include "omni/text2speech/kokoro/unicode_utils.h"

namespace litert::omni::tts::kokoro {
namespace {

// UniDic character categories (`char.def`) that `misaki.ja` distinguishes.
constexpr int kSymbolCharType = 3;
constexpr int kHiraganaCharType = 6;
constexpr int kKatakanaCharType = 7;

constexpr uint32_t kSokuon = 0x3063;       // っ
constexpr uint32_t kMoraicNasal = 0x3093;  // ん
constexpr uint32_t kChoonpu = 0x30FC;      // ー
constexpr uint32_t kGlottalStop = 0x0294;  // ʔ

// Drops the last UTF-8 character of `text`, which must be valid UTF-8.
absl::string_view DropLastUtf8Char(absl::string_view text) {
  if (text.empty()) return text;
  size_t end = text.size() - 1;
  while (end > 0 && (static_cast<uint8_t>(text[end]) & 0xC0) == 0x80) --end;
  return text.substr(0, end);
}

bool IsAllAscii(absl::string_view text) {
  for (char c : text) {
    if (static_cast<uint8_t>(c) >= 0x80) return false;
  }
  return true;
}

bool IsWhitespaceOnly(absl::string_view text) {
  for (uint32_t cp : Utf8ToCodePoints(text)) {
    if (cp != ' ' && cp != '\t' && cp != '\n' && cp != '\r' && cp != 0x3000) {
      return false;
    }
  }
  return true;
}

// Converts katakana (U+30A1..U+30F6) in `text` to hiragana (U+3041..U+3096),
// matching `jaconv.kata2hira`.
std::string KataToHira(absl::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (uint32_t cp : Utf8ToCodePoints(text)) {
    AppendUtf8((cp >= 0x30A1 && cp <= 0x30F6) ? cp - 0x60 : cp, out);
  }
  return out;
}

// Parses one CSV line from a MeCab token feature record.
std::vector<std::string> ParseCsvRecord(absl::string_view record) {
  std::vector<std::string> fields;
  std::string current;
  bool in_quotes = false;
  for (size_t i = 0; i < record.size(); ++i) {
    const char c = record[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < record.size() && record[i + 1] == '"') {
          current.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        current.push_back(c);
      }
    } else if (c == '"') {
      in_quotes = true;
    } else if (c == ',') {
      fields.push_back(std::move(current));
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  fields.push_back(std::move(current));
  return fields;
}

// Port of `misaki/num2kana.py::Convert(digits, 'hiragana')`.
absl::string_view DigitHira(char d) {
  constexpr absl::string_view kDigits[] = {"ゼロ", "いち",  "に",   "さん",
                                           "よん", "ご",    "ろく", "なな",
                                           "はち", "きゅう"};
  return (d >= '0' && d <= '9') ? kDigits[d - '0'] : "";
}

std::string NumLenTwo(absl::string_view n) {
  if (n[0] == '0') return std::string(DigitHira(n[1]));
  if (n == "10") return "じゅう";
  if (n[0] == '1') return absl::StrCat("じゅう", DigitHira(n[1]));
  if (n[1] == '0') return absl::StrCat(DigitHira(n[0]), "じゅう");
  return absl::StrCat(DigitHira(n[0]), "じゅう", DigitHira(n[1]));
}

std::string NumLenThree(absl::string_view n) {
  std::string out;
  if (n[0] == '1') {
    out = "ひゃく";
  } else if (n[0] == '3') {
    out = "さんびゃく";
  } else if (n[0] == '6') {
    out = "ろっぴゃく";
  } else if (n[0] == '8') {
    out = "はっぴゃく";
  } else {
    out = absl::StrCat(DigitHira(n[0]), "ひゃく");
  }
  if (n.substr(1) == "00") return out;
  if (n[1] == '0') {
    absl::StrAppend(&out, DigitHira(n[2]));
  } else {
    absl::StrAppend(&out, NumLenTwo(n.substr(1)));
  }
  return out;
}

std::string NumLenFour(absl::string_view n, bool stand_alone) {
  if (n == "0000") return "";
  while (!n.empty() && n[0] == '0') {
    n.remove_prefix(1);
  }
  if (n.size() == 1) return std::string(DigitHira(n[0]));
  if (n.size() == 2) return NumLenTwo(n);
  if (n.size() == 3) return NumLenThree(n);

  std::string out;
  if (n[0] == '1' && stand_alone) {
    out = "せん";
  } else if (n[0] == '1') {
    out = "いっせん";
  } else if (n[0] == '3') {
    out = "さんぜん";
  } else if (n[0] == '8') {
    out = "はっせん";
  } else {
    out = absl::StrCat(DigitHira(n[0]), "せん");
  }
  if (n.substr(1) == "000") return out;
  if (n[1] == '0') {
    absl::StrAppend(&out, NumLenTwo(n.substr(2)));
  } else {
    absl::StrAppend(&out, NumLenThree(n.substr(1)));
  }
  return out;
}

std::string NumToHiragana(absl::string_view num) {
  while (num.size() > 1 && num[0] == '0') {
    num.remove_prefix(1);
  }
  if (num.empty()) return "";
  if (num.size() > 9) {
    return "Number length too long, choose less than 10 digits";
  }
  if (num.size() == 1) return std::string(DigitHira(num[0]));
  if (num.size() == 2) return NumLenTwo(num);
  if (num.size() == 3) return NumLenThree(num);
  if (num.size() == 4) return NumLenFour(num, true);

  const absl::string_view head = num.substr(0, num.size() - 4);
  const absl::string_view tail = num.substr(num.size() - 4);
  std::string out;
  if (head.size() == 1) {
    out = absl::StrCat(DigitHira(head[0]), "まん");
  } else if (head.size() == 2) {
    out = absl::StrCat(NumLenTwo(head), "まん");
  } else if (head.size() == 3) {
    out = absl::StrCat(NumLenThree(head), "まん");
  } else if (head.size() == 4) {
    out = absl::StrCat(NumLenFour(head, false), "まん");
  } else if (head.size() == 5) {
    out = absl::StrCat(DigitHira(head[0]), "おく",
                       NumLenFour(head.substr(1, 4), false));
    if (head.substr(1, 4) != "0000") {
      absl::StrAppend(&out, "まん");
    }
  }
  absl::StrAppend(&out, NumLenFour(tail, false));
  return out;
}

// A hiragana and its voiced (dakuten) form.
struct DakutenPair {
  char32_t plain;
  char32_t voiced;
};

// Port of `misaki.ja.add_dakuten`. Listed explicitly because the code points
// do not follow one parity rule: small `っ` (U+3063) sits between `ち` and
// `つ`, so `つ`, `て`, `と` are even while `か`..`ち` are odd.
constexpr DakutenPair kDakutenPairs[] = {
    {U'か', U'が'}, {U'き', U'ぎ'}, {U'く', U'ぐ'}, {U'け', U'げ'},
    {U'こ', U'ご'}, {U'さ', U'ざ'}, {U'し', U'じ'}, {U'す', U'ず'},
    {U'せ', U'ぜ'}, {U'そ', U'ぞ'}, {U'た', U'だ'}, {U'ち', U'ぢ'},
    {U'つ', U'づ'}, {U'て', U'で'}, {U'と', U'ど'}, {U'は', U'ば'},
    {U'ひ', U'び'}, {U'ふ', U'ぶ'}, {U'へ', U'べ'}, {U'ほ', U'ぼ'},
};

// Returns the voiced form of hiragana `cp`, or 0 if it has none.
uint32_t AddDakuten(uint32_t cp) {
  for (const DakutenPair& pair : kDakutenPairs) {
    if (pair.plain == cp) return pair.voiced;
  }
  return 0;
}

// A full-width katakana code point and whether it absorbed the following
// half-width (han)dakuten mark.
struct FullwidthKatakana {
  uint32_t code_point = 0;
  bool consumed_mark = false;
};

// Maps half-width katakana (U+FF61..U+FF9F) to full-width katakana, composing
// a following dakuten (U+FF9E) or handakuten (U+FF9F) the way NFKC does in
// `misaki.ja.JAG2P._normalize_text`. A mark that does not compose (e.g. after
// small `ｯ`) is left as its combining form.
FullwidthKatakana HalfwidthKatakanaToFull(uint32_t cp, uint32_t next_cp) {
  static constexpr uint16_t kBase[64] = {
      0xFF60, 0x3002, 0x300C, 0x300D, 0x3001, 0x30FB, 0x30F2, 0x30A1,
      0x30A3, 0x30A5, 0x30A7, 0x30A9, 0x30E3, 0x30E5, 0x30E7, 0x30C3,
      0x30FC, 0x30A2, 0x30A4, 0x30A6, 0x30A8, 0x30AA, 0x30AB, 0x30AD,
      0x30AF, 0x30B1, 0x30B3, 0x30B5, 0x30B7, 0x30B9, 0x30BB, 0x30BD,
      0x30BF, 0x30C1, 0x30C4, 0x30C6, 0x30C8, 0x30CA, 0x30CB, 0x30CC,
      0x30CD, 0x30CE, 0x30CF, 0x30D2, 0x30D5, 0x30D8, 0x30DB, 0x30DE,
      0x30DF, 0x30E0, 0x30E1, 0x30E2, 0x30E4, 0x30E6, 0x30E8, 0x30E9,
      0x30EA, 0x30EB, 0x30EC, 0x30ED, 0x30EF, 0x30F3, 0x3099, 0x309A,
  };
  if (cp < 0xFF61 || cp > 0xFF9F) return {cp, false};
  const uint32_t mapped = kBase[cp - 0xFF60];
  if (next_cp == 0xFF9E) {
    if (mapped == 0x30A6) return {0x30F4, true};  // ウ -> ヴ
    if (mapped == 0x30EF) return {0x30F7, true};  // ワ -> ヷ
    if (mapped == 0x30F2) return {0x30FA, true};  // ヲ -> ヺ
    // Katakana sits exactly 0x60 above its hiragana.
    if (mapped >= 0x30A1 && mapped <= 0x30F6) {
      const uint32_t voiced = AddDakuten(mapped - 0x60);
      if (voiced != 0) return {voiced + 0x60, true};
    }
  } else if (next_cp == 0xFF9F && mapped >= 0x30CF && mapped <= 0x30DB &&
             (mapped - 0x30CF) % 3 == 0) {
    return {mapped + 2, true};  // ハヒフヘホ -> パピプペポ
  }
  return {mapped, false};
}

bool IsDigitOrFullwidthDigit(uint32_t cp) {
  return (cp >= '0' && cp <= '9') || (cp >= 0xFF10 && cp <= 0xFF19);
}

// Port of `JAG2P._normalize_text`.
std::string NormalizeJapaneseText(absl::string_view text) {
  const std::vector<char32_t> cps = Utf8ToCodePoints(text);
  std::vector<uint32_t> stage1;
  stage1.reserve(cps.size());

  static constexpr uint16_t kKatakanaExt[16] = {
      0x30AF, 0x30B7, 0x30B9, 0x30C8, 0x30CC, 0x30CF, 0x30D2, 0x30D5,
      0x30D8, 0x30DB, 0x30E0, 0x30E9, 0x30EA, 0x30EB, 0x30EC, 0x30ED,
  };

  for (size_t i = 0; i < cps.size(); ++i) {
    const uint32_t cp = cps[i];
    const uint32_t next_cp = (i + 1 < cps.size()) ? cps[i + 1] : 0;
    if ((cp == 0x301C || cp == 0xFF5E) && IsDigitOrFullwidthDigit(next_cp)) {
      stage1.push_back(0x304B);  // か
      stage1.push_back(0x3089);  // ら
    } else if (cp >= 0x31F0 && cp <= 0x31FF) {
      stage1.push_back(kKatakanaExt[cp - 0x31F0]);
    } else if (cp >= 0xFF01 && cp <= 0xFF5E) {
      stage1.push_back(cp - 0xFEE0);
    } else if (cp == 0x3000) {
      stage1.push_back(' ');
    } else if (cp >= 0xFF61 && cp <= 0xFF9F) {
      const FullwidthKatakana full = HalfwidthKatakanaToFull(cp, next_cp);
      stage1.push_back(full.code_point);
      if (full.consumed_mark) ++i;
    } else {
      stage1.push_back(cp);
    }
  }

  std::string out;
  size_t i = 0;
  while (i < stage1.size()) {
    if (stage1[i] >= '0' && stage1[i] <= '9') {
      std::string digits;
      while (i < stage1.size() && stage1[i] >= '0' && stage1[i] <= '9') {
        digits.push_back(static_cast<char>(stage1[i]));
        ++i;
      }
      out.push_back(' ');
      out.append(NumToHiragana(digits));
    } else {
      AppendUtf8(stage1[i], out);
      ++i;
    }
  }
  return out;
}

// One entry of `misaki.ja`'s `HEPBURN` table: a kana (or kana digraph when
// `second` is non-zero) or punctuation mark and its IPA.
struct HepburnEntry {
  char32_t first;
  char32_t second;
  absl::string_view ipa;
};

constexpr HepburnEntry kHepburn[] = {
    // Digraphs.
    {U'い', U'ぇ', "je"},
    {U'う', U'ぃ', "βi"},
    {U'う', U'ぇ', "βe"},
    {U'う', U'ぉ', "βo"},
    {U'き', U'ぇ', "kʲe"},
    {U'き', U'ゃ', "kʲa"},
    {U'き', U'ゅ', "kʲɨ"},
    {U'き', U'ょ', "kʲo"},
    {U'ぎ', U'ゃ', "ɡʲa"},
    {U'ぎ', U'ゅ', "ɡʲɨ"},
    {U'ぎ', U'ょ', "ɡʲo"},
    {U'く', U'ぁ', "kᵝa"},
    {U'く', U'ぃ', "kᵝi"},
    {U'く', U'ぇ', "kᵝe"},
    {U'く', U'ぉ', "kᵝo"},
    {U'ぐ', U'ぁ', "ɡᵝa"},
    {U'ぐ', U'ぃ', "ɡᵝi"},
    {U'ぐ', U'ぇ', "ɡᵝe"},
    {U'ぐ', U'ぉ', "ɡᵝo"},
    {U'し', U'ぇ', "ɕe"},
    {U'し', U'ゃ', "ɕa"},
    {U'し', U'ゅ', "ɕɨ"},
    {U'し', U'ょ', "ɕo"},
    {U'じ', U'ぇ', "ʥe"},
    {U'じ', U'ゃ', "ʥa"},
    {U'じ', U'ゅ', "ʥɨ"},
    {U'じ', U'ょ', "ʥo"},
    {U'ち', U'ぇ', "ʨe"},
    {U'ち', U'ゃ', "ʨa"},
    {U'ち', U'ゅ', "ʨɨ"},
    {U'ち', U'ょ', "ʨo"},
    {U'ぢ', U'ゃ', "ʥa"},
    {U'ぢ', U'ゅ', "ʥɨ"},
    {U'ぢ', U'ょ', "ʥo"},
    {U'つ', U'ぁ', "ʦa"},
    {U'つ', U'ぃ', "ʦʲi"},
    {U'つ', U'ぇ', "ʦe"},
    {U'つ', U'ぉ', "ʦo"},
    {U'て', U'ぃ', "tʲi"},
    {U'て', U'ゅ', "tʲɨ"},
    {U'で', U'ぃ', "dʲi"},
    {U'で', U'ゅ', "dʲɨ"},
    {U'と', U'ぅ', "tɯ"},
    {U'ど', U'ぅ', "dɯ"},
    {U'に', U'ぇ', "ɲe"},
    {U'に', U'ゃ', "ɲa"},
    {U'に', U'ゅ', "ɲɨ"},
    {U'に', U'ょ', "ɲo"},
    {U'ひ', U'ぇ', "çe"},
    {U'ひ', U'ゃ', "ça"},
    {U'ひ', U'ゅ', "çɨ"},
    {U'ひ', U'ょ', "ço"},
    {U'び', U'ゃ', "bʲa"},
    {U'び', U'ゅ', "bʲɨ"},
    {U'び', U'ょ', "bʲo"},
    {U'ぴ', U'ゃ', "pʲa"},
    {U'ぴ', U'ゅ', "pʲɨ"},
    {U'ぴ', U'ょ', "pʲo"},
    {U'ふ', U'ぁ', "ɸa"},
    {U'ふ', U'ぃ', "ɸʲi"},
    {U'ふ', U'ぇ', "ɸe"},
    {U'ふ', U'ぉ', "ɸo"},
    {U'ふ', U'ゅ', "ɸʲɨ"},
    {U'ふ', U'ょ', "ɸʲo"},
    {U'み', U'ゃ', "mʲa"},
    {U'み', U'ゅ', "mʲɨ"},
    {U'み', U'ょ', "mʲo"},
    {U'り', U'ゃ', "ɾʲa"},
    {U'り', U'ゅ', "ɾʲɨ"},
    {U'り', U'ょ', "ɾʲo"},
    {U'ゔ', U'ぁ', "va"},
    {U'ゔ', U'ぃ', "vʲi"},
    {U'ゔ', U'ぇ', "ve"},
    {U'ゔ', U'ぉ', "vo"},
    {U'ゔ', U'ゅ', "bʲɨ"},
    {U'ゔ', U'ょ', "bʲo"},
    // Single kana.
    {U'ぁ', 0, "a"},
    {U'あ', 0, "a"},
    {U'ぃ', 0, "i"},
    {U'い', 0, "i"},
    {U'ぅ', 0, "ɯ"},
    {U'う', 0, "ɯ"},
    {U'ぇ', 0, "e"},
    {U'え', 0, "e"},
    {U'ぉ', 0, "o"},
    {U'お', 0, "o"},
    {U'か', 0, "ka"},
    {U'が', 0, "ɡa"},
    {U'き', 0, "kʲi"},
    {U'ぎ', 0, "ɡʲi"},
    {U'く', 0, "kɯ"},
    {U'ぐ', 0, "ɡɯ"},
    {U'け', 0, "ke"},
    {U'げ', 0, "ɡe"},
    {U'こ', 0, "ko"},
    {U'ご', 0, "ɡo"},
    {U'さ', 0, "sa"},
    {U'ざ', 0, "ʣa"},
    {U'し', 0, "ɕi"},
    {U'じ', 0, "ʥi"},
    {U'す', 0, "sɨ"},
    {U'ず', 0, "zɨ"},
    {U'せ', 0, "se"},
    {U'ぜ', 0, "ʣe"},
    {U'そ', 0, "so"},
    {U'ぞ', 0, "ʣo"},
    {U'た', 0, "ta"},
    {U'だ', 0, "da"},
    {U'ち', 0, "ʨi"},
    {U'ぢ', 0, "ʥi"},
    {U'つ', 0, "ʦɨ"},
    {U'づ', 0, "zɨ"},
    {U'て', 0, "te"},
    {U'で', 0, "de"},
    {U'と', 0, "to"},
    {U'ど', 0, "do"},
    {U'な', 0, "na"},
    {U'に', 0, "ɲi"},
    {U'ぬ', 0, "nɯ"},
    {U'ね', 0, "ne"},
    {U'の', 0, "no"},
    {U'は', 0, "ha"},
    {U'ば', 0, "ba"},
    {U'ぱ', 0, "pa"},
    {U'ひ', 0, "çi"},
    {U'び', 0, "bʲi"},
    {U'ぴ', 0, "pʲi"},
    {U'ふ', 0, "ɸɯ"},
    {U'ぶ', 0, "bɯ"},
    {U'ぷ', 0, "pɯ"},
    {U'へ', 0, "he"},
    {U'べ', 0, "be"},
    {U'ぺ', 0, "pe"},
    {U'ほ', 0, "ho"},
    {U'ぼ', 0, "bo"},
    {U'ぽ', 0, "po"},
    {U'ま', 0, "ma"},
    {U'み', 0, "mʲi"},
    {U'む', 0, "mɯ"},
    {U'め', 0, "me"},
    {U'も', 0, "mo"},
    {U'ゃ', 0, "ja"},
    {U'や', 0, "ja"},
    {U'ゅ', 0, "jɯ"},
    {U'ゆ', 0, "jɯ"},
    {U'ょ', 0, "jo"},
    {U'よ', 0, "jo"},
    {U'ら', 0, "ɾa"},
    {U'り', 0, "ɾʲi"},
    {U'る', 0, "ɾɯ"},
    {U'れ', 0, "ɾe"},
    {U'ろ', 0, "ɾo"},
    {U'ゎ', 0, "βa"},
    {U'わ', 0, "βa"},
    {U'ゐ', 0, "i"},
    {U'ゑ', 0, "e"},
    {U'を', 0, "o"},
    {U'ゔ', 0, "vɯ"},
    {U'ゕ', 0, "ka"},
    {U'ゖ', 0, "ke"},
    {U'ヷ', 0, "va"},
    {U'ヸ', 0, "vʲi"},
    {U'ヹ', 0, "ve"},
    {U'ヺ', 0, "vo"},
    // Punctuation.
    {U'。', 0, "."},
    {U'、', 0, ","},
    {U'？', 0, "?"},
    {U'！', 0, "!"},
    {U'「', 0, "“"},
    {U'『', 0, "“"},
    {U'«', 0, "“"},
    {U'」', 0, "”"},
    {U'』', 0, "”"},
    {U'»', 0, "”"},
    {U'：', 0, ":"},
    {U'；', 0, ";"},
    {U'（', 0, "("},
    {U'《', 0, "("},
    {U'）', 0, ")"},
    {U'》', 0, ")"},
    {U'【', 0, "["},
    {U'】', 0, "]"},
    {U'・', 0, " "},
    {U'，', 0, ","},
    {U'～', 0, "—"},
    {U'〜', 0, "—"},
    {U'—', 0, "—"},
};

// Returns the IPA for kana `first` (followed by `second`, for a digraph), or
// an empty string if `kHepburn` has no entry for it.
absl::string_view LookupHepburn(uint32_t first, uint32_t second = 0) {
  for (const HepburnEntry& entry : kHepburn) {
    if (entry.first == first && entry.second == second) return entry.ipa;
  }
  return "";
}

bool IsSutegana(uint32_t cp) {
  return cp == 0x3083 || cp == 0x3085 || cp == 0x3087 || cp == 0x3041 ||
         cp == 0x3043 || cp == 0x3045 || cp == 0x3047 || cp == 0x3049;
}

bool IsOdori(uint32_t cp) {
  return cp == 0x3003 || cp == 0x3005 || cp == 0x309D || cp == 0x309E ||
         cp == 0x30FD || cp == 0x30FE;
}

// Maps the moraic nasal `ん` to the nasal assimilated to the place of
// articulation of `next_ipa`, the IPA of the kana that follows it.
absl::string_view AssimilateMoraicNasal(absl::string_view next_ipa) {
  if (next_ipa.empty()) return "ɴ";
  const uint32_t first_cp = DecodeUtf8CodePoint(next_ipa, 0).code_point;
  // Bilabial: m, p, b.
  if (first_cp == 'm' || first_cp == 'p' || first_cp == 'b') return "m";
  // Velar: k, ɡ (U+0261).
  if (first_cp == 'k' || first_cp == 0x0261) return "ŋ";
  // Palatal: ɲ (U+0272), ʨ (U+02A8), ʥ (U+02A5).
  if (first_cp == 0x0272 || first_cp == 0x02A8 || first_cp == 0x02A5) {
    return "ɲ";
  }
  // Alveolar: n, t, d, ɾ (U+027E), z.
  if (first_cp == 'n' || first_cp == 't' || first_cp == 'd' ||
      first_cp == 0x027E || first_cp == 'z') {
    return "n";
  }
  return "ɴ";
}

// Maps hiragana `kk` to IPA given its neighbours `pk` and `nk` (0 at the word
// edges), following `misaki.ja`'s Hepburn rules.
std::string MapSingleKana(uint32_t pk, uint32_t kk, uint32_t nk) {
  if (IsOdori(kk)) {
    if (pk == 0) return "";
    // `ゝ`/`ヽ` repeat the previous kana and `ゞ`/`ヾ` repeat it voiced. misaki
    // returns the raw previous kana for `ゝ`, which leaks kana into the IPA;
    // this emits its IPA instead.
    if (kk == 0x309D || kk == 0x30FD) return std::string(LookupHepburn(pk));
    if (kk == 0x309E || kk == 0x30FE) {
      const uint32_t voiced = AddDakuten(pk);
      return voiced == 0 ? "" : std::string(LookupHepburn(voiced));
    }
    // `々` and `〃` repeat a span that cannot be inferred reliably.
    return "";
  }
  if (pk != 0) {
    const absl::string_view digraph = LookupHepburn(pk, kk);
    if (!digraph.empty()) return std::string(digraph);
  }
  if (nk != 0 && !LookupHepburn(kk, nk).empty()) return "";
  if (nk != 0 && IsSutegana(nk)) {
    if (kk == kSokuon) return "";
    return absl::StrCat(DropLastUtf8Char(LookupHepburn(kk)), LookupHepburn(nk));
  }
  if (IsSutegana(kk)) return "";
  if (kk == kChoonpu) return "ː";
  if (kk == kSokuon) return "ʔ";
  if (kk == kMoraicNasal) {
    return std::string(AssimilateMoraicNasal(nk != 0 ? LookupHepburn(nk) : ""));
  }
  return std::string(LookupHepburn(kk));
}

struct WordItem {
  std::string surface;
  std::string hira;
  int char_type = 0;
};

// A romanized word and whether a space follows it in the joined output.
struct TokenOut {
  std::string surface;
  bool space = false;
};

std::string RomajiWord(const WordItem& word) {
  if (IsAllAscii(word.surface)) return word.surface;
  std::string out;
  if (word.char_type == kSymbolCharType) {
    for (uint32_t cp : Utf8ToCodePoints(word.surface)) {
      if (cp == 0x3099 || cp == 0x309A) continue;
      const absl::string_view mapped = LookupHepburn(cp);
      if (mapped.empty()) {
        AppendUtf8(cp, out);
      } else {
        out.append(mapped);
      }
    }
    return out;
  }
  if (word.char_type != kHiraganaCharType) return "";
  const std::vector<char32_t> hira = Utf8ToCodePoints(word.hira);
  for (size_t i = 0; i < hira.size(); ++i) {
    const uint32_t pk = (i > 0) ? hira[i - 1] : 0;
    const uint32_t nk = (i + 1 < hira.size()) ? hira[i + 1] : 0;
    out.append(MapSingleKana(pk, hira[i], nk));
  }
  return out;
}

bool IsOpeningBracketSurface(absl::string_view surface) {
  return surface == "「" || surface == "『" || surface == "«";
}

bool IsClosingPunctSurface(absl::string_view surface) {
  return surface == "」" || surface == "』" || surface == "»";
}

bool IsClosingPunctRoma(absl::string_view roma) {
  return roma == "]" || roma == ")" || roma == "." || roma == "," ||
         roma == "?" || roma == "!" || roma == ":";
}

std::string StripSokuon(absl::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (uint32_t cp : Utf8ToCodePoints(text)) {
    if (cp != kSokuon) AppendUtf8(cp, out);
  }
  return out;
}

bool IsKeptBeforeGlottal(uint32_t cp) {
  return cp == '!' || cp == '"' || cp == ',' || cp == '.' || cp == ':' ||
         cp == ';' || cp == '?' || cp == 0x00BB || cp == 0x2014 ||
         cp == 0x2026 || cp == 0x201D;
}

bool IsKeptAfterGlottal(uint32_t cp) {
  return cp == '"' || cp == 0x00AB || cp == 0x201C;
}

// Collapses whitespace, converts '(' -> '«' and ')' -> '»', and deletes spaces
// adjacent to 'ʔ' matching `JAG2P.__call__`.
std::string PostProcessIpa(absl::string_view raw) {
  std::vector<uint32_t> collapsed;
  collapsed.reserve(raw.size());
  bool in_space = true;
  for (uint32_t cp : Utf8ToCodePoints(raw)) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') {
      if (!in_space) collapsed.push_back(' ');
      in_space = true;
      continue;
    }
    if (cp == '(') {
      cp = 0x00AB;
    } else if (cp == ')') {
      cp = 0x00BB;
    }
    collapsed.push_back(cp);
    in_space = false;
  }
  if (!collapsed.empty() && collapsed.back() == ' ') collapsed.pop_back();

  std::string out;
  for (size_t i = 0; i < collapsed.size(); ++i) {
    if (collapsed[i] == ' ') {
      const bool before_glottal =
          i + 1 < collapsed.size() && collapsed[i + 1] == kGlottalStop;
      const bool after_glottal = i > 0 && collapsed[i - 1] == kGlottalStop;
      if (before_glottal &&
          (i == 0 || !IsKeptBeforeGlottal(collapsed[i - 1]))) {
        continue;
      }
      if (after_glottal && (i + 1 >= collapsed.size() ||
                            !IsKeptAfterGlottal(collapsed[i + 1]))) {
        continue;
      }
    }
    AppendUtf8(collapsed[i], out);
  }
  return out;
}

// Returns whether column `idx` of a feature record holds a value.
bool HasField(const std::vector<std::string>& cols, size_t idx) {
  return idx < cols.size() && !cols[idx].empty() && cols[idx] != "*";
}

// Returns the kana reading of a MeCab feature record: `pron`, else `kana`,
// else `surface`. Pruned 8-column records keep them at columns 5 and 6; full
// 26-column unidic-lite records at 9 and 17.
std::string ReadingOf(const std::vector<std::string>& cols,
                      absl::string_view surface) {
  const bool pruned = cols.size() <= 10;
  const size_t pron_idx = pruned ? 5 : 9;
  const size_t kana_idx = pruned ? 6 : 17;
  if (HasField(cols, pron_idx)) return cols[pron_idx];
  if (HasField(cols, kana_idx)) return cols[kana_idx];
  return std::string(surface);
}

}  // namespace

JapaneseG2p::JapaneseG2p(MecabCostModel model,
                         std::optional<SortedStringIndex> words_index)
    : model_(std::move(model)), words_index_(words_index) {}

absl::StatusOr<JapaneseG2p> JapaneseG2p::Create(const CjkBlob& blob) {
  ABSL_ASSIGN_OR_RETURN(MecabCostModel model, MecabCostModel::Create(blob));
  std::optional<SortedStringIndex> words_index;
  if (blob.Contains("w-idx") && blob.Contains("w-key")) {
    ABSL_ASSIGN_OR_RETURN(const absl::string_view idx, blob.Section("w-idx"));
    ABSL_ASSIGN_OR_RETURN(const absl::string_view key, blob.Section("w-key"));
    ABSL_ASSIGN_OR_RETURN(words_index, SortedStringIndex::Create(idx, key));
  }
  return JapaneseG2p(std::move(model), words_index);
}

absl::StatusOr<std::string> JapaneseG2p::TextToIpa(
    absl::string_view text) const {
  if (text.empty()) return "";
  const std::string normalized = NormalizeJapaneseText(text);
  if (normalized.empty() || IsWhitespaceOnly(normalized)) return "";

  ABSL_ASSIGN_OR_RETURN(const std::vector<LatticeNode> path,
                        ViterbiSegment(normalized, model_));

  std::vector<WordItem> words;
  words.reserve(path.size());
  for (const LatticeNode& node : path) {
    if (node.surface.empty() || IsWhitespaceOnly(node.surface)) continue;

    const std::vector<std::string> cols =
        ParseCsvRecord(model_.Token(node).feature);
    if (!cols.empty() && cols[0] == "空白") continue;

    // `misaki.ja` treats every dictionary word and every unknown katakana run
    // as kana to be read; only unknown non-katakana keeps its own category.
    const int raw_char_type =
        model_.Characters().InfoAt(node.surface, 0).info.default_type;
    const int char_type =
        (raw_char_type == kKatakanaCharType || !node.is_unknown)
            ? kHiraganaCharType
            : raw_char_type;

    words.push_back({std::string(node.surface),
                     KataToHira(ReadingOf(cols, node.surface)), char_type});
  }

  // Group words of the same `char_type` when their concatenated surface appears
  // in `ja_words.txt` (`words_index_`).
  std::vector<WordItem> grouped;
  size_t i = 0;
  while (i < words.size()) {
    size_t best_j = 0;
    if (words_index_.has_value() && words_index_->Size() > 0) {
      std::string concat;
      for (size_t k = i;
           k < words.size() && words[k].char_type == words[i].char_type; ++k) {
        concat.append(words[k].surface);
        if (!words_index_->HasPrefix(concat)) break;
        if (words_index_->Find(concat) >= 0) best_j = k + 1;
      }
    }
    if (best_j == 0) {
      grouped.push_back(std::move(words[i]));
      ++i;
      continue;
    }
    WordItem merged;
    merged.char_type = words[i].char_type;
    for (size_t k = i; k < best_j; ++k) {
      merged.surface.append(words[k].surface);
      merged.hira.append(words[k].hira);
    }
    grouped.push_back(std::move(merged));
    i = best_j;
  }

  // Join tokens, attaching closing punctuation to the word before it and
  // opening brackets to the word after.
  std::vector<TokenOut> out;
  out.reserve(grouped.size());
  for (const WordItem& word : grouped) {
    TokenOut tok{StripSokuon(RomajiWord(word)), true};
    if (IsOpeningBracketSurface(word.surface) || tok.surface == "(" ||
        tok.surface == "[") {
      if (!out.empty()) out.back().space = true;
      tok.space = false;
    } else if (IsClosingPunctSurface(word.surface) ||
               IsClosingPunctRoma(tok.surface)) {
      if (!out.empty()) out.back().space = false;
    } else if (tok.surface == " ") {
      tok.space = false;
    }
    out.push_back(std::move(tok));
  }

  std::string joined;
  for (const TokenOut& tok : out) {
    joined.append(tok.surface);
    if (tok.space) joined.push_back(' ');
  }
  return PostProcessIpa(joined);
}

}  // namespace litert::omni::tts::kokoro
