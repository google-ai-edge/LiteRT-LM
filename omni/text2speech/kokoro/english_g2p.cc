// Copyright 2026 Google LLC.
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

#include "omni/text2speech/kokoro/english_g2p.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/unicode_utils.h"

namespace litert::omni::tts::kokoro {
namespace {

constexpr char32_t kPrimaryStress = U'ˈ';
constexpr char32_t kSecondaryStress = U'ˌ';
constexpr absl::string_view kPrimaryStressUtf8 = "ˈ";
constexpr absl::string_view kSecondaryStressUtf8 = "ˌ";
constexpr absl::string_view kUnkUtf8 = "❓";

bool IsDiphthong(char32_t cp) {
  switch (cp) {
    case U'A':
    case U'I':
    case U'O':
    case U'Q':
    case U'W':
    case U'Y':
    case U'ʤ':
    case U'ʧ':
      return true;
    default:
      return false;
  }
}

bool IsVowelPhoneme(char32_t cp) {
  switch (cp) {
    case U'A':
    case U'I':
    case U'O':
    case U'Q':
    case U'W':
    case U'Y':
    case U'a':
    case U'i':
    case U'u':
    case U'æ':
    case U'ɑ':
    case U'ɒ':
    case U'ɔ':
    case U'ə':
    case U'ɛ':
    case U'ɜ':
    case U'ɪ':
    case U'ʊ':
    case U'ʌ':
    case U'ᵻ':
      return true;
    default:
      return false;
  }
}

bool IsConsonantPhoneme(char32_t cp) {
  switch (cp) {
    case U'b':
    case U'd':
    case U'f':
    case U'h':
    case U'j':
    case U'k':
    case U'l':
    case U'm':
    case U'n':
    case U'p':
    case U's':
    case U't':
    case U'v':
    case U'w':
    case U'z':
    case U'ð':
    case U'ŋ':
    case U'ɡ':
    case U'ɹ':
    case U'ɾ':
    case U'ʃ':
    case U'ʒ':
    case U'ʤ':
    case U'ʧ':
    case U'θ':
      return true;
    default:
      return false;
  }
}

bool IsUsTau(char32_t cp) {
  switch (cp) {
    case U'A':
    case U'I':
    case U'O':
    case U'W':
    case U'Y':
    case U'i':
    case U'u':
    case U'æ':
    case U'ɑ':
    case U'ə':
    case U'ɛ':
    case U'ɪ':
    case U'ɹ':
    case U'ʊ':
    case U'ʌ':
      return true;
    default:
      return false;
  }
}

bool IsPunctPhoneme(char32_t cp) {
  switch (cp) {
    case U';':
    case U':':
    case U',':
    case U'.':
    case U'!':
    case U'?':
    case U'—':
    case U'…':
    case U'"':
    case U'“':
    case U'”':
      return true;
    default:
      return false;
  }
}

bool IsNonQuotePunctPhoneme(char32_t cp) {
  return IsPunctPhoneme(cp) && cp != U'"' && cp != U'“' && cp != U'”';
}

bool IsSubtokenJunk(char32_t cp) {
  switch (cp) {
    case U'\'':
    case U',':
    case U'-':
    case U'.':
    case U'_':
    case U'‘':
    case U'’':
    case U'/':
      return true;
    default:
      return false;
  }
}

bool IsLexiconOrd(char32_t cp) {
  return cp == U'\'' || cp == U'-' || (cp >= U'A' && cp <= U'Z') ||
         (cp >= U'a' && cp <= U'z');
}

bool AllLexiconOrds(absl::string_view word) {
  for (size_t i = 0; i < word.size();) {
    const Utf8CodePoint dc = DecodeUtf8CodePoint(word, i);
    if (!IsLexiconOrd(dc.code_point)) {
      return false;
    }
    i += dc.length;
  }
  return true;
}

bool IsUnicodeUpper(char32_t cp) {
  if (cp >= U'A' && cp <= U'Z') {
    return true;
  }
  return cp >= 0xC0 && cp <= 0xDE && cp != 0xD7;
}

bool IsUnicodeLower(char32_t cp) {
  if (cp >= U'a' && cp <= U'z') {
    return true;
  }
  return cp >= 0xDF && cp <= 0xFF && cp != 0xF7;
}

bool IsUnicodeLetter(char32_t cp) {
  return IsUnicodeUpper(cp) || IsUnicodeLower(cp) ||
         (cp >= 0x100 && cp <= 0x024F);
}

char32_t ToUnicodeUpper(char32_t cp) {
  if (cp >= U'a' && cp <= U'z') {
    return cp - (U'a' - U'A');
  }
  if (cp >= 0xE0 && cp <= 0xFE && cp != 0xF7) {
    return cp - 0x20;
  }
  return cp;
}

std::string NormalizeWordUtf8(absl::string_view word) {
  std::string out;
  out.reserve(word.size());
  for (size_t i = 0; i < word.size();) {
    const Utf8CodePoint dc = DecodeUtf8CodePoint(word, i);
    if (dc.code_point == U'‘' || dc.code_point == U'’') {
      out.push_back('\'');
    } else {
      AppendUtf8(dc.code_point, out);
    }
    i += dc.length;
  }
  return out;
}

std::string CapitalizeUtf8(absl::string_view word) {
  std::string out;
  out.reserve(word.size());
  bool first = true;
  for (size_t i = 0; i < word.size();) {
    const Utf8CodePoint dc = DecodeUtf8CodePoint(word, i);
    AppendUtf8(
        first ? ToUnicodeUpper(dc.code_point) : ToLowerLatin(dc.code_point),
        out);
    first = false;
    i += dc.length;
  }
  return out;
}

bool IsAllAlphaAscii(absl::string_view word) {
  if (word.empty()) {
    return false;
  }
  for (char c : word) {
    if (!absl::ascii_isalpha(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

bool IsAllUpperAlphaAscii(absl::string_view word) {
  if (word.empty()) {
    return false;
  }
  for (char c : word) {
    if (!absl::ascii_isupper(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

bool IsRestLowerUtf8(absl::string_view word) {
  bool first = true;
  for (size_t i = 0; i < word.size();) {
    const Utf8CodePoint dc = DecodeUtf8CodePoint(word, i);
    if (!first && IsUnicodeUpper(dc.code_point)) {
      return false;
    }
    first = false;
    i += dc.length;
  }
  return true;
}

bool IsRestUpperAscii(absl::string_view word) {
  if (word.size() <= 1) {
    return true;
  }
  for (size_t i = 1; i < word.size(); ++i) {
    const auto c = static_cast<unsigned char>(word[i]);
    if (absl::ascii_islower(c)) {
      return false;
    }
  }
  return true;
}

// Computes the alternate-case key matching `misaki.en.Lexicon.grow_dictionary`.
std::optional<std::string> AltCaseKey(absl::string_view key) {
  const std::u32string u32 = Utf8ToU32(key);
  if (u32.size() < 2) {
    return std::nullopt;
  }
  const std::string low = ToLowerUtf8(key);
  const std::string cap = CapitalizeUtf8(low);
  if (key == low) {
    if (key != cap) {
      return cap;
    }
  } else if (key == cap) {
    return low;
  }
  return std::nullopt;
}

struct PosChar {
  int order_x2 = 0;
  char32_t cp = 0;
};

bool ComparePosCharOrder(const PosChar& a, const PosChar& b) {
  return a.order_x2 < b.order_x2;
}

std::string RestressPhonemes(const std::u32string& ps) {
  std::vector<PosChar> items;
  items.reserve(ps.size());
  for (size_t i = 0; i < ps.size(); ++i) {
    int order_x2 = static_cast<int>(i) * 2;
    if (ps[i] == kPrimaryStress || ps[i] == kSecondaryStress) {
      for (size_t j = i; j < ps.size(); ++j) {
        if (IsVowelPhoneme(ps[j])) {
          order_x2 = static_cast<int>(j) * 2 - 1;
          break;
        }
      }
    }
    items.push_back({order_x2, ps[i]});
  }
  absl::c_stable_sort(items, ComparePosCharOrder);
  std::u32string out;
  out.reserve(items.size());
  for (const PosChar& item : items) {
    out.push_back(item.cp);
  }
  return CodePointsToUtf8(out);
}

std::optional<std::string> ApplyStress(std::optional<std::string> ps_opt,
                                       std::optional<double> stress) {
  if (!ps_opt.has_value() || !stress.has_value()) {
    return ps_opt;
  }
  const double s = *stress;
  const std::u32string ps = Utf8ToU32(*ps_opt);
  bool has_primary = false;
  bool has_secondary = false;
  bool has_vowel = false;
  for (char32_t cp : ps) {
    if (cp == kPrimaryStress) {
      has_primary = true;
    } else if (cp == kSecondaryStress) {
      has_secondary = true;
    } else if (IsVowelPhoneme(cp)) {
      has_vowel = true;
    }
  }
  if (s < -1.0) {
    std::u32string out;
    out.reserve(ps.size());
    for (char32_t cp : ps) {
      if (cp != kPrimaryStress && cp != kSecondaryStress) {
        out.push_back(cp);
      }
    }
    return CodePointsToUtf8(out);
  }
  if (s == -1.0 || ((s == 0.0 || s == -0.5) && has_primary)) {
    std::u32string out;
    out.reserve(ps.size());
    for (char32_t cp : ps) {
      if (cp == kSecondaryStress) {
        continue;
      }
      out.push_back(cp == kPrimaryStress ? kSecondaryStress : cp);
    }
    return CodePointsToUtf8(out);
  }
  if ((s == 0.0 || s == 0.5 || s == 1.0) && !has_primary && !has_secondary) {
    if (!has_vowel) {
      return ps_opt;
    }
    std::u32string prefixed;
    prefixed.reserve(ps.size() + 1);
    prefixed.push_back(kSecondaryStress);
    prefixed.append(ps);
    return RestressPhonemes(prefixed);
  }
  if (s >= 1.0 && !has_primary && has_secondary) {
    std::u32string out;
    out.reserve(ps.size());
    for (char32_t cp : ps) {
      out.push_back(cp == kSecondaryStress ? kPrimaryStress : cp);
    }
    return CodePointsToUtf8(out);
  }
  if (s > 1.0 && !has_primary && !has_secondary) {
    if (!has_vowel) {
      return ps_opt;
    }
    std::u32string prefixed;
    prefixed.reserve(ps.size() + 1);
    prefixed.push_back(kPrimaryStress);
    prefixed.append(ps);
    return RestressPhonemes(prefixed);
  }
  return ps_opt;
}

constexpr absl::string_view kOnes[] = {
    "zero",    "one",     "two",       "three",    "four",
    "five",    "six",     "seven",     "eight",    "nine",
    "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
};

constexpr absl::string_view kTens[] = {
    "",      "",      "twenty",  "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety",
};

struct WordPair {
  absl::string_view key;
  absl::string_view value;
};

constexpr WordPair kOrdinalExact[] = {
    {"zero", "zeroth"},    {"one", "first"},     {"two", "second"},
    {"three", "third"},    {"four", "fourth"},   {"five", "fifth"},
    {"six", "sixth"},      {"seven", "seventh"}, {"eight", "eighth"},
    {"nine", "ninth"},     {"ten", "tenth"},     {"eleven", "eleventh"},
    {"twelve", "twelfth"},
};

struct NumberScale {
  int64_t divisor;
  absl::string_view name;
};

constexpr NumberScale kNumberScales[] = {
    {1000000000000LL, "trillion"},
    {1000000000LL, "billion"},
    {1000000LL, "million"},
    {1000LL, "thousand"},
};

struct CurrencyUnitAmount {
  int64_t amount = 0;
  absl::string_view unit;
};

std::string IntToCardinal(int64_t n) {
  if (n < 0) {
    return absl::StrCat("minus ", IntToCardinal(-n));
  }
  if (n < 20) {
    return std::string(kOnes[n]);
  }
  if (n < 100) {
    const int64_t t = n / 10;
    const int64_t r = n % 10;
    return r == 0 ? std::string(kTens[t])
                  : absl::StrCat(kTens[t], "-", kOnes[r]);
  }
  if (n < 1000) {
    const int64_t h = n / 100;
    const int64_t r = n % 100;
    return r == 0 ? absl::StrCat(kOnes[h], " hundred")
                  : absl::StrCat(kOnes[h], " hundred and ", IntToCardinal(r));
  }
  for (const NumberScale& s : kNumberScales) {
    if (n >= s.divisor) {
      const int64_t q = n / s.divisor;
      const int64_t r = n % s.divisor;
      std::string head = absl::StrCat(IntToCardinal(q), " ", s.name);
      if (r == 0) {
        return head;
      }
      absl::StrAppend(&head, r < 100 ? " and " : ", ", IntToCardinal(r));
      return head;
    }
  }
  return absl::StrCat(n);
}

std::string IntToOrdinal(int64_t n) {
  std::string card = IntToCardinal(n);
  size_t split_pos = card.find_last_of(" -");
  std::string prefix =
      split_pos == std::string::npos ? "" : card.substr(0, split_pos + 1);
  absl::string_view last = split_pos == std::string::npos
                               ? absl::string_view(card)
                               : absl::string_view(card).substr(split_pos + 1);
  for (const WordPair& p : kOrdinalExact) {
    if (last == p.key) {
      return absl::StrCat(prefix, p.value);
    }
  }
  if (absl::EndsWith(last, "y")) {
    return absl::StrCat(prefix, last.substr(0, last.size() - 1), "ieth");
  }
  return absl::StrCat(prefix, last, "th");
}

std::string IntToYear(int64_t n) {
  if (n >= 1000 && n < 2000) {
    const int64_t hi = n / 100;
    const int64_t lo = n % 100;
    if (lo == 0) {
      return absl::StrCat(IntToCardinal(hi), " hundred");
    }
    if (lo < 10) {
      return absl::StrCat(IntToCardinal(hi), " oh-", kOnes[lo]);
    }
    return absl::StrCat(IntToCardinal(hi), " ", IntToCardinal(lo));
  }
  if (n >= 2000 && n < 2100) {
    const int64_t hi = n / 100;
    const int64_t lo = n % 100;
    if (lo == 0) {
      return "two thousand";
    }
    if (lo < 10) {
      return absl::StrCat("two thousand and ", kOnes[lo]);
    }
    return absl::StrCat(IntToCardinal(hi), " ", IntToCardinal(lo));
  }
  return IntToCardinal(n);
}

std::string FloatToWords(absl::string_view s) {
  const size_t dot = s.find('.');
  if (dot == absl::string_view::npos) {
    int64_t val = 0;
    if (absl::SimpleAtoi(s, &val)) {
      return IntToCardinal(val);
    }
    return std::string(s);
  }
  const absl::string_view int_part = s.substr(0, dot);
  absl::string_view frac_part = s.substr(dot + 1);
  while (!frac_part.empty() && frac_part.back() == '0') {
    frac_part.remove_suffix(1);
  }
  int64_t int_val = 0;
  if (!int_part.empty()) {
    (void)absl::SimpleAtoi(int_part, &int_val);
  }
  std::string out = absl::StrCat(IntToCardinal(int_val), " point");
  if (frac_part.empty()) {
    absl::StrAppend(&out, " zero");
    return out;
  }
  for (char c : frac_part) {
    if (c >= '0' && c <= '9') {
      absl::StrAppend(&out, " ", kOnes[c - '0']);
    }
  }
  return out;
}

struct TokenContext {
  std::optional<bool> future_vowel;
  bool future_to = false;
};

struct PhonemeResult {
  std::optional<std::string> phonemes;
  int rating = 0;
};

struct MToken {
  std::string text;
  std::string tag;
  std::string whitespace;
  bool is_head = true;
  std::optional<std::string> alias;
  std::optional<std::string> phonemes;
  std::optional<double> stress;
  std::string currency;
  std::string num_flags;
  bool prespace = false;
  std::optional<int> rating;

  bool IsTo() const {
    return text == "to" || text == "To" ||
           (text == "TO" && (tag == "TO" || tag == "IN"));
  }

  int StressWeight() const {
    if (!phonemes.has_value()) {
      return 0;
    }
    const std::u32string u32 = Utf8ToU32(*phonemes);
    int w = 0;
    for (char32_t cp : u32) {
      w += IsDiphthong(cp) ? 2 : 1;
    }
    return w;
  }
};

int TokenCaseScore(absl::string_view text) {
  int score = 0;
  for (size_t i = 0; i < text.size();) {
    const Utf8CodePoint dc = DecodeUtf8CodePoint(text, i);
    score += IsUnicodeUpper(dc.code_point) ? 2 : 1;
    i += dc.length;
  }
  return score;
}

MToken MergeTokens(const std::vector<MToken>& tokens,
                   std::optional<absl::string_view> unk) {
  MToken out;
  if (tokens.empty()) {
    return out;
  }
  size_t text_len = 0;
  for (const MToken& t : tokens) {
    text_len += t.text.size() + t.whitespace.size();
  }
  out.text.reserve(text_len);
  for (size_t i = 0; i + 1 < tokens.size(); ++i) {
    out.text.append(tokens[i].text);
    out.text.append(tokens[i].whitespace);
  }
  out.text.append(tokens.back().text);

  int best_score = -1;
  for (const MToken& t : tokens) {
    const int s = TokenCaseScore(t.text);
    if (s > best_score) {
      best_score = s;
      out.tag = t.tag;
    }
  }
  out.whitespace = tokens.back().whitespace;
  out.is_head = tokens.front().is_head;
  out.prespace = tokens.front().prespace;

  bool multiple_stresses = false;
  std::optional<double> unique_stress;
  for (const MToken& t : tokens) {
    if (t.stress.has_value()) {
      if (!unique_stress.has_value()) {
        unique_stress = t.stress;
      } else if (*unique_stress != *t.stress) {
        multiple_stresses = true;
      }
    }
    if (!t.currency.empty() && t.currency > out.currency) {
      out.currency = t.currency;
    }
    for (char c : t.num_flags) {
      if (!absl::StrContains(out.num_flags, c)) {
        out.num_flags.push_back(c);
      }
    }
  }
  absl::c_sort(out.num_flags);
  if (!multiple_stresses) {
    out.stress = unique_stress;
  }

  bool any_null_rating = false;
  int min_rating = 100;
  for (const MToken& t : tokens) {
    if (!t.rating.has_value()) {
      any_null_rating = true;
    } else if (*t.rating < min_rating) {
      min_rating = *t.rating;
    }
  }
  if (!any_null_rating && min_rating < 100) {
    out.rating = min_rating;
  }

  if (unk.has_value()) {
    std::string ph;
    ph.reserve(text_len * 2);
    for (const MToken& t : tokens) {
      if (t.prespace && !ph.empty() &&
          !absl::ascii_isspace(static_cast<unsigned char>(ph.back())) &&
          t.phonemes.has_value() && !t.phonemes->empty()) {
        ph.push_back(' ');
      }
      if (t.phonemes.has_value()) {
        ph.append(*t.phonemes);
      } else {
        ph.append(*unk);
      }
    }
    out.phonemes = std::move(ph);
  }
  return out;
}

absl::string_view GetParentTag(absl::string_view tag) {
  if (absl::StartsWith(tag, "VB")) {
    return "VERB";
  }
  if (absl::StartsWith(tag, "NN")) {
    return "NOUN";
  }
  if (absl::StartsWith(tag, "ADV") || absl::StartsWith(tag, "RB")) {
    return "ADV";
  }
  if (absl::StartsWith(tag, "ADJ") || absl::StartsWith(tag, "JJ")) {
    return "ADJ";
  }
  return tag;
}

std::optional<std::string> ResolveEncodedLexiconValue(absl::string_view payload,
                                                      absl::string_view tag,
                                                      const TokenContext* ctx) {
  if (!absl::StrContains(payload, '=')) {
    return std::string(payload);
  }
  absl::string_view default_val;
  bool has_none_key = false;
  std::optional<absl::string_view> none_val;
  std::optional<absl::string_view> exact_val;
  std::optional<absl::string_view> parent_val;
  const absl::string_view parent_tag = GetParentTag(tag);

  for (absl::string_view entry : absl::StrSplit(payload, '|')) {
    const size_t eq = entry.find('=');
    if (eq == absl::string_view::npos) {
      continue;
    }
    const absl::string_view k = entry.substr(0, eq);
    const absl::string_view v = entry.substr(eq + 1);
    if (k == "DEFAULT") {
      default_val = v;
    }
    if (k == "None") {
      has_none_key = true;
      if (!v.empty()) {
        none_val = v;
      }
    }
    if (!tag.empty() && k == tag) {
      exact_val = v;
    }
    if (!parent_tag.empty() && k == parent_tag) {
      parent_val = v;
    }
  }

  if (ctx != nullptr && !ctx->future_vowel.has_value() && has_none_key) {
    return none_val.has_value() ? std::optional<std::string>(*none_val)
                                : std::nullopt;
  }
  if (exact_val.has_value()) {
    return exact_val->empty() ? std::nullopt
                              : std::optional<std::string>(*exact_val);
  }
  if (parent_val.has_value()) {
    return parent_val->empty() ? std::nullopt
                               : std::optional<std::string>(*parent_val);
  }
  return default_val.empty() ? std::nullopt
                             : std::optional<std::string>(default_val);
}

class AccentLexiconTable {
 public:
  AccentLexiconTable(SortedStringIndex keys, StringValueTable values)
      : keys_(keys), values_(values) {}

  std::optional<absl::string_view> FindGoldRaw(absl::string_view word) const {
    if (const auto raw = FindWithRating(word, '4'); raw.has_value()) {
      return raw;
    }
    if (const auto alt = AltCaseKey(word); alt.has_value()) {
      return FindWithRating(*alt, '4');
    }
    return std::nullopt;
  }

  std::optional<absl::string_view> FindSilverRaw(absl::string_view word) const {
    if (const auto raw = FindWithRating(word, '3'); raw.has_value()) {
      return raw;
    }
    if (const auto alt = AltCaseKey(word); alt.has_value()) {
      return FindWithRating(*alt, '3');
    }
    return std::nullopt;
  }

  bool InGold(absl::string_view word) const {
    return FindGoldRaw(word).has_value();
  }

  bool InSilver(absl::string_view word) const {
    return FindSilverRaw(word).has_value();
  }

 private:
  std::optional<absl::string_view> FindWithRating(absl::string_view key,
                                                  char rating_char) const {
    const std::optional<int> idx = keys_.Find(key);
    if (!idx.has_value() || *idx < 0 || *idx >= values_.Size()) {
      return std::nullopt;
    }
    const absl::string_view val = values_.Value(*idx);
    if (val.empty() || val[0] != rating_char) {
      return std::nullopt;
    }
    return val.substr(1);
  }

  SortedStringIndex keys_;
  StringValueTable values_;
};

struct CurrencyUnits {
  absl::string_view symbol;
  absl::string_view major;
  absl::string_view minor;
};

constexpr CurrencyUnits kCurrencies[] = {
    {"$", "dollar", "cent"},
    {"£", "pound", "pence"},
    {"€", "euro", "cent"},
};

const CurrencyUnits* FindCurrency(absl::string_view sym) {
  for (const CurrencyUnits& c : kCurrencies) {
    if (c.symbol == sym) {
      return &c;
    }
  }
  return nullptr;
}

constexpr WordPair kAddSymbols[] = {
    {".", "dot"},
    {"/", "slash"},
};

constexpr WordPair kSymbols[] = {
    {"%", "percent"},
    {"&", "and"},
    {"+", "plus"},
    {"@", "at"},
};

std::optional<absl::string_view> FindInPairs(absl::string_view key,
                                             const WordPair* pairs,
                                             size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (pairs[i].key == key) {
      return pairs[i].value;
    }
  }
  return std::nullopt;
}

bool IsDoubledConsonantIng(absl::string_view word) {
  if (word.size() >= 6 && absl::EndsWith(word, "cking")) {
    return true;
  }
  if (word.size() < 5 || !absl::EndsWith(word, "ing")) {
    return false;
  }
  const char c1 = word[word.size() - 4];
  const char c2 = word[word.size() - 5];
  if (c1 != c2) {
    return false;
  }
  constexpr absl::string_view kDoubled = "bcdgklmnprstvxz";
  return absl::StrContains(kDoubled, c1);
}

bool IsDottedAcronym(absl::string_view word) {
  absl::string_view trimmed = word;
  while (!trimmed.empty() && trimmed.front() == '.') {
    trimmed.remove_prefix(1);
  }
  while (!trimmed.empty() && trimmed.back() == '.') {
    trimmed.remove_suffix(1);
  }
  if (!absl::StrContains(trimmed, '.')) {
    return false;
  }
  size_t max_part = 0;
  size_t cur_part = 0;
  bool any_alpha = false;
  for (char c : word) {
    if (c == '.') {
      max_part = std::max(max_part, cur_part);
      cur_part = 0;
    } else if (absl::ascii_isalpha(static_cast<unsigned char>(c))) {
      any_alpha = true;
      ++cur_part;
    } else {
      return false;
    }
  }
  max_part = std::max(max_part, cur_part);
  return any_alpha && max_part < 3;
}

class EnglishLexicon {
 public:
  EnglishLexicon(
      AccentLexiconTable table, bool british,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon)
      : table_(table), british_(british), custom_lexicon_(custom_lexicon) {}

  PhonemeResult GetNnp(absl::string_view word) const {
    std::string combined;
    combined.reserve(word.size() * 4);
    bool any_alpha = false;
    for (char c : word) {
      if (!absl::ascii_isalpha(static_cast<unsigned char>(c))) {
        continue;
      }
      any_alpha = true;
      const char upper_c = absl::ascii_toupper(static_cast<unsigned char>(c));
      const absl::string_view letter(&upper_c, 1);
      const std::optional<absl::string_view> raw = table_.FindGoldRaw(letter);
      if (!raw.has_value()) {
        return {std::nullopt, 0};
      }
      const std::optional<std::string> resolved =
          ResolveEncodedLexiconValue(*raw, "", nullptr);
      if (!resolved.has_value()) {
        return {std::nullopt, 0};
      }
      combined.append(*resolved);
    }
    if (!any_alpha) {
      return {std::nullopt, 0};
    }
    const std::optional<std::string> stressed = ApplyStress(combined, 0.0);
    if (!stressed.has_value()) {
      return {std::nullopt, 0};
    }
    const size_t pos = stressed->rfind(kSecondaryStressUtf8);
    if (pos == std::string::npos) {
      return {*stressed, 3};
    }
    std::string out;
    out.reserve(stressed->size());
    out.append(stressed->substr(0, pos));
    out.append(kPrimaryStressUtf8);
    out.append(stressed->substr(pos + kSecondaryStressUtf8.size()));
    return {std::move(out), 3};
  }

  PhonemeResult GetSpecialCase(absl::string_view word, absl::string_view tag,
                               std::optional<double> stress,
                               const TokenContext* ctx) const {
    if (tag == "ADD") {
      if (const auto sym = FindInPairs(word, kAddSymbols, 2); sym.has_value()) {
        return Lookup(*sym, "", -0.5, ctx);
      }
    }
    if (const auto sym = FindInPairs(word, kSymbols, 4); sym.has_value()) {
      return Lookup(*sym, "", std::nullopt, ctx);
    }
    if (IsDottedAcronym(word)) {
      return GetNnp(word);
    }
    if (word == "a" || (word == "A" && tag == "DT")) {
      return {"ɐ", 4};
    }
    if (word == "am" || word == "Am" || word == "AM") {
      if (absl::StartsWith(tag, "NN")) {
        return GetNnp(word);
      }
      if (ctx == nullptr || !ctx->future_vowel.has_value() || word != "am" ||
          (stress.has_value() && *stress > 0.0)) {
        const auto raw = table_.FindGoldRaw("am");
        if (raw.has_value()) {
          return {ResolveEncodedLexiconValue(*raw, "", nullptr), 4};
        }
      }
      return {"ɐm", 4};
    }
    if (word == "an" || word == "An" || word == "AN") {
      if (word == "AN" && absl::StartsWith(tag, "NN")) {
        return GetNnp(word);
      }
      return {"ɐn", 4};
    }
    if (word == "I" && tag == "PRP") {
      return {absl::StrCat(kSecondaryStressUtf8, "I"), 4};
    }
    if ((word == "by" || word == "By" || word == "BY") &&
        GetParentTag(tag) == "ADV") {
      return {"bˈI", 4};
    }
    if (word == "to" || word == "To" ||
        (word == "TO" && (tag == "TO" || tag == "IN"))) {
      if (ctx == nullptr || !ctx->future_vowel.has_value()) {
        const auto raw = table_.FindGoldRaw("to");
        if (raw.has_value()) {
          return {ResolveEncodedLexiconValue(*raw, "", nullptr), 4};
        }
        return {"tu", 4};
      }
      return {*ctx->future_vowel ? "tʊ" : "tə", 4};
    }
    if (word == "the" || word == "The" || (word == "THE" && tag == "DT")) {
      return {(ctx != nullptr && ctx->future_vowel.has_value() &&
               *ctx->future_vowel)
                  ? "ði"
                  : "ðə",
              4};
    }
    if (tag == "IN" && (absl::EqualsIgnoreCase(word, "vs") ||
                        absl::EqualsIgnoreCase(word, "vs."))) {
      return Lookup("versus", "", std::nullopt, ctx);
    }
    if (word == "used" || word == "Used" || word == "USED") {
      const absl::string_view sub_tag =
          ((tag == "VBD" || tag == "JJ") && ctx != nullptr && ctx->future_to)
              ? "VBD"
              : "DEFAULT";
      const auto raw = table_.FindGoldRaw("used");
      if (raw.has_value()) {
        return {ResolveEncodedLexiconValue(*raw, sub_tag, nullptr), 4};
      }
    }
    return {std::nullopt, 0};
  }

  bool IsKnown(absl::string_view word, absl::string_view /*tag*/) const {
    if (!custom_lexicon_.empty() &&
        custom_lexicon_.contains(ToLowerUtf8(word))) {
      return true;
    }
    if (table_.InGold(word) || FindInPairs(word, kSymbols, 4).has_value() ||
        table_.InSilver(word)) {
      return true;
    }
    if (!IsAllAlphaAscii(word) || !AllLexiconOrds(word)) {
      return false;
    }
    if (word.size() == 1) {
      return true;
    }
    const std::string low = absl::AsciiStrToLower(word);
    if (IsAllUpperAlphaAscii(word) && table_.InGold(low)) {
      return true;
    }
    return IsRestUpperAscii(word);
  }

  PhonemeResult Lookup(absl::string_view word, absl::string_view tag,
                       std::optional<double> stress,
                       const TokenContext* ctx) const {
    if (!custom_lexicon_.empty()) {
      const std::string low = ToLowerUtf8(word);
      if (const auto it = custom_lexicon_.find(low);
          it != custom_lexicon_.end()) {
        return {ApplyStress(it->second, stress), 4};
      }
    }
    std::string lookup_word(word);
    bool has_is_nnp = false;
    bool is_nnp = false;
    if (IsAllUpperAlphaAscii(lookup_word) && !table_.InGold(lookup_word)) {
      lookup_word = absl::AsciiStrToLower(lookup_word);
      has_is_nnp = true;
      is_nnp = (tag == "NNP");
    }
    std::optional<std::string> ps;
    int rating = 4;
    if (const auto raw_gold = table_.FindGoldRaw(lookup_word);
        raw_gold.has_value()) {
      ps = ResolveEncodedLexiconValue(*raw_gold, tag, ctx);
      rating = 4;
    } else if (!is_nnp) {
      if (const auto raw_silver = table_.FindSilverRaw(lookup_word);
          raw_silver.has_value()) {
        ps = ResolveEncodedLexiconValue(*raw_silver, tag, ctx);
        rating = 3;
      }
    }
    if (!ps.has_value() ||
        (has_is_nnp && is_nnp &&
         ps->find(kPrimaryStressUtf8) == std::string::npos)) {
      const PhonemeResult nnp = GetNnp(lookup_word);
      if (nnp.phonemes.has_value()) {
        return nnp;
      }
    }
    return {ApplyStress(ps, stress), rating};
  }

  std::optional<std::string> InflectS(
      const std::optional<std::string>& stem) const {
    if (!stem.has_value() || stem->empty()) {
      return std::nullopt;
    }
    const std::u32string u32 = Utf8ToU32(*stem);
    if (u32.empty()) {
      return std::nullopt;
    }
    const char32_t last = u32.back();
    if (last == U'p' || last == U't' || last == U'k' || last == U'f' ||
        last == U'θ') {
      return absl::StrCat(*stem, "s");
    }
    if (last == U's' || last == U'z' || last == U'ʃ' || last == U'ʒ' ||
        last == U'ʧ' || last == U'ʤ') {
      return absl::StrCat(*stem, british_ ? "ɪz" : "ᵻz");
    }
    return absl::StrCat(*stem, "z");
  }

  PhonemeResult StemS(absl::string_view word, absl::string_view tag,
                      std::optional<double> stress,
                      const TokenContext* ctx) const {
    std::string stem;
    if (word.size() > 2 && absl::EndsWith(word, "s") &&
        !absl::EndsWith(word, "ss") &&
        IsKnown(word.substr(0, word.size() - 1), tag)) {
      stem = std::string(word.substr(0, word.size() - 1));
    } else if ((absl::EndsWith(word, "'s") ||
                (word.size() > 4 && absl::EndsWith(word, "es"))) &&
               IsKnown(word.substr(0, word.size() - 2), tag)) {
      stem = std::string(word.substr(0, word.size() - 2));
    } else if (word.size() > 4 && absl::EndsWith(word, "ies") &&
               IsKnown(absl::StrCat(word.substr(0, word.size() - 3), "y"),
                       tag)) {
      stem = absl::StrCat(word.substr(0, word.size() - 3), "y");
    } else {
      return {std::nullopt, 0};
    }
    const PhonemeResult base = Lookup(stem, tag, stress, ctx);
    return {InflectS(base.phonemes), base.rating};
  }

  std::optional<std::string> InflectEd(
      const std::optional<std::string>& stem) const {
    if (!stem.has_value() || stem->empty()) {
      return std::nullopt;
    }
    const std::u32string u32 = Utf8ToU32(*stem);
    if (u32.empty()) {
      return std::nullopt;
    }
    const char32_t last = u32.back();
    if (last == U'p' || last == U'k' || last == U'f' || last == U'θ' ||
        last == U'ʃ' || last == U's' || last == U'ʧ') {
      return absl::StrCat(*stem, "t");
    }
    if (last == U'd') {
      return absl::StrCat(*stem, british_ ? "ɪd" : "ᵻd");
    }
    if (last != U't') {
      return absl::StrCat(*stem, "d");
    }
    if (british_ || u32.size() < 2) {
      return absl::StrCat(*stem, "ɪd");
    }
    if (IsUsTau(u32[u32.size() - 2])) {
      std::u32string prefix(u32.begin(), u32.end() - 1);
      return absl::StrCat(CodePointsToUtf8(prefix), "ɾᵻd");
    }
    return absl::StrCat(*stem, "ᵻd");
  }

  PhonemeResult StemEd(absl::string_view word, absl::string_view tag,
                       std::optional<double> stress,
                       const TokenContext* ctx) const {
    std::string stem;
    if (absl::EndsWith(word, "d") && !absl::EndsWith(word, "dd") &&
        IsKnown(word.substr(0, word.size() - 1), tag)) {
      stem = std::string(word.substr(0, word.size() - 1));
    } else if (absl::EndsWith(word, "ed") && !absl::EndsWith(word, "eed") &&
               IsKnown(word.substr(0, word.size() - 2), tag)) {
      stem = std::string(word.substr(0, word.size() - 2));
    } else {
      return {std::nullopt, 0};
    }
    const PhonemeResult base = Lookup(stem, tag, stress, ctx);
    return {InflectEd(base.phonemes), base.rating};
  }

  std::optional<std::string> InflectIng(
      const std::optional<std::string>& stem) const {
    if (!stem.has_value() || stem->empty()) {
      return std::nullopt;
    }
    const std::u32string u32 = Utf8ToU32(*stem);
    if (u32.empty()) {
      return std::nullopt;
    }
    const char32_t last = u32.back();
    if (british_) {
      if (last == U'ə' || last == U'ː') {
        return std::nullopt;
      }
    } else if (u32.size() > 1 && last == U't' && IsUsTau(u32[u32.size() - 2])) {
      std::u32string prefix(u32.begin(), u32.end() - 1);
      return absl::StrCat(CodePointsToUtf8(prefix), "ɾɪŋ");
    }
    return absl::StrCat(*stem, "ɪŋ");
  }

  PhonemeResult StemIng(absl::string_view word, absl::string_view tag,
                        std::optional<double> stress,
                        const TokenContext* ctx) const {
    std::string stem;
    if (word.size() > 3 && absl::EndsWith(word, "ing") &&
        IsKnown(word.substr(0, word.size() - 3), tag)) {
      stem = std::string(word.substr(0, word.size() - 3));
    } else if (word.size() > 3 && absl::EndsWith(word, "ing") &&
               IsKnown(absl::StrCat(word.substr(0, word.size() - 3), "e"),
                       tag)) {
      stem = absl::StrCat(word.substr(0, word.size() - 3), "e");
    } else if (IsDoubledConsonantIng(word) &&
               IsKnown(word.substr(0, word.size() - 4), tag)) {
      stem = std::string(word.substr(0, word.size() - 4));
    } else {
      return {std::nullopt, 0};
    }
    const PhonemeResult base = Lookup(stem, tag, stress, ctx);
    return {InflectIng(base.phonemes), base.rating};
  }

  PhonemeResult GetWord(absl::string_view word, absl::string_view tag,
                        std::optional<double> stress,
                        const TokenContext* ctx) const {
    const PhonemeResult sc = GetSpecialCase(word, tag, stress, ctx);
    if (sc.phonemes.has_value()) {
      return sc;
    }
    if (IsKnown(word, tag)) {
      return Lookup(word, tag, stress, ctx);
    }
    if (word.size() > 2 && absl::EndsWith(word, "s'") &&
        IsKnown(absl::StrCat(word.substr(0, word.size() - 2), "'s"), tag)) {
      return Lookup(absl::StrCat(word.substr(0, word.size() - 2), "'s"), tag,
                    stress, ctx);
    }
    if (word.size() > 1 && absl::EndsWith(word, "'") &&
        IsKnown(word.substr(0, word.size() - 1), tag)) {
      return Lookup(word.substr(0, word.size() - 1), tag, stress, ctx);
    }
    const PhonemeResult s_res = StemS(word, tag, stress, ctx);
    if (s_res.phonemes.has_value()) {
      return s_res;
    }
    const PhonemeResult ed_res = StemEd(word, tag, stress, ctx);
    if (ed_res.phonemes.has_value()) {
      return ed_res;
    }
    const PhonemeResult ing_res =
        StemIng(word, tag, stress.has_value() ? stress : 0.5, ctx);
    if (ing_res.phonemes.has_value()) {
      return ing_res;
    }
    return {std::nullopt, 0};
  }

  static bool IsCurrencyNumber(absl::string_view word) {
    const size_t first_dot = word.find('.');
    if (first_dot == absl::string_view::npos) {
      return true;
    }
    if (word.find('.', first_dot + 1) != absl::string_view::npos) {
      return false;
    }
    const absl::string_view cents = word.substr(first_dot + 1);
    if (cents.size() < 3) {
      return true;
    }
    for (char c : cents) {
      if (c != '0') {
        return false;
      }
    }
    return true;
  }

  void ExtendNumberWords(absl::string_view words_str, bool first, bool escape,
                         absl::string_view num_flags,
                         std::vector<PhonemeResult>& result) const {
    std::string expanded;
    if (escape) {
      expanded = std::string(words_str);
    } else {
      int64_t val = 0;
      if (absl::SimpleAtoi(words_str, &val)) {
        expanded = IntToCardinal(val);
      } else {
        expanded = std::string(words_str);
      }
    }
    std::vector<std::string> splits;
    std::string cur;
    for (char c : expanded) {
      if (c >= 'a' && c <= 'z') {
        cur.push_back(c);
      } else if (!cur.empty()) {
        splits.push_back(std::move(cur));
        cur.clear();
      }
    }
    if (!cur.empty()) {
      splits.push_back(std::move(cur));
    }
    for (size_t i = 0; i < splits.size(); ++i) {
      const std::string& w = splits[i];
      if (w != "and" || absl::StrContains(num_flags, '&')) {
        if (first && i == 0 && splits.size() > 1 && w == "one" &&
            absl::StrContains(num_flags, 'a')) {
          result.push_back({"ə", 4});
        } else {
          const std::optional<double> w_stress =
              (w == "point") ? std::optional<double>(-2.0) : std::nullopt;
          result.push_back(Lookup(w, "", w_stress, nullptr));
        }
      } else if (w == "and" && absl::StrContains(num_flags, 'n') &&
                 !result.empty() && result.back().phonemes.has_value()) {
        absl::StrAppend(&(*result.back().phonemes), "ən");
      }
    }
  }

  PhonemeResult GetNumber(absl::string_view raw_word,
                          absl::string_view currency, bool is_head,
                          absl::string_view num_flags) const {
    size_t suffix_pos = raw_word.size();
    while (suffix_pos > 0) {
      const char c = raw_word[suffix_pos - 1];
      if ((c >= 'a' && c <= 'z') || c == '\'') {
        --suffix_pos;
      } else {
        break;
      }
    }
    const absl::string_view suffix = raw_word.substr(suffix_pos);
    absl::string_view word = raw_word.substr(0, suffix_pos);

    std::vector<PhonemeResult> result;
    if (absl::StartsWith(word, "-")) {
      result.push_back(Lookup("minus", "", std::nullopt, nullptr));
      word.remove_prefix(1);
    }

    std::string no_commas;
    no_commas.reserve(word.size());
    for (char c : word) {
      if (c != ',') {
        no_commas.push_back(c);
      }
    }

    const size_t dot_count = std::count(word.begin(), word.end(), '.');
    const CurrencyUnits* curr_units = FindCurrency(currency);
    const bool is_ordinal =
        (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th");

    if (!is_head && dot_count == 0) {
      if (!no_commas.empty() && (no_commas[0] == '0' || no_commas.size() > 3)) {
        for (char c : no_commas) {
          ExtendNumberWords(absl::string_view(&c, 1), /*first=*/false,
                            /*escape=*/false, num_flags, result);
        }
      } else if (no_commas.size() == 3 && !absl::EndsWith(no_commas, "00")) {
        ExtendNumberWords(absl::string_view(no_commas).substr(0, 1),
                          /*first=*/true, /*escape=*/false, num_flags, result);
        if (no_commas[1] == '0') {
          result.push_back(Lookup("O", "", -2.0, nullptr));
          ExtendNumberWords(absl::string_view(no_commas).substr(2, 1),
                            /*first=*/false, /*escape=*/false, num_flags,
                            result);
        } else {
          ExtendNumberWords(absl::string_view(no_commas).substr(1, 2),
                            /*first=*/false, /*escape=*/false, num_flags,
                            result);
        }
      } else {
        ExtendNumberWords(no_commas, /*first=*/true, /*escape=*/false,
                          num_flags, result);
      }
    } else if (dot_count > 1 || !is_head) {
      bool first = true;
      for (absl::string_view part : absl::StrSplit(no_commas, '.')) {
        if (part.empty()) {
          first = false;
          continue;
        }
        bool any_nonzero_tail = false;
        for (size_t k = 1; k < part.size(); ++k) {
          if (part[k] != '0') {
            any_nonzero_tail = true;
            break;
          }
        }
        if (part[0] == '0' || (part.size() != 2 && any_nonzero_tail)) {
          for (char c : part) {
            ExtendNumberWords(absl::string_view(&c, 1), /*first=*/false,
                              /*escape=*/false, num_flags, result);
          }
        } else {
          ExtendNumberWords(part, /*first=*/first, /*escape=*/false, num_flags,
                            result);
        }
        first = false;
      }
    } else if (curr_units != nullptr && IsCurrencyNumber(word)) {
      std::vector<CurrencyUnitAmount> pairs;
      const size_t dot = no_commas.find('.');
      if (dot == std::string::npos) {
        int64_t major_val = 0;
        (void)absl::SimpleAtoi(no_commas, &major_val);
        pairs.push_back({major_val, curr_units->major});
      } else {
        int64_t major_val = 0;
        int64_t minor_val = 0;
        const absl::string_view p0 =
            absl::string_view(no_commas).substr(0, dot);
        const absl::string_view p1 =
            absl::string_view(no_commas).substr(dot + 1);
        if (!p0.empty()) {
          (void)absl::SimpleAtoi(p0, &major_val);
        }
        if (!p1.empty()) {
          (void)absl::SimpleAtoi(p1, &minor_val);
        }
        pairs.push_back({major_val, curr_units->major});
        pairs.push_back({minor_val, curr_units->minor});
        if (pairs[1].amount == 0) {
          pairs.pop_back();
        } else if (pairs[0].amount == 0) {
          pairs.erase(pairs.begin());
        }
      }
      for (size_t i = 0; i < pairs.size(); ++i) {
        if (i > 0) {
          result.push_back(Lookup("and", "", std::nullopt, nullptr));
        }
        ExtendNumberWords(absl::StrCat(pairs[i].amount), /*first=*/i == 0,
                          /*escape=*/false, num_flags, result);
        if (std::abs(pairs[i].amount) != 1 && pairs[i].unit != "pence") {
          result.push_back(StemS(absl::StrCat(pairs[i].unit, "s"), "",
                                 std::nullopt, nullptr));
        } else {
          result.push_back(Lookup(pairs[i].unit, "", std::nullopt, nullptr));
        }
      }
    } else {
      bool all_digits = !word.empty();
      for (char c : word) {
        if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
          all_digits = false;
          break;
        }
      }
      std::string word_str;
      if (all_digits) {
        int64_t val = 0;
        (void)absl::SimpleAtoi(word, &val);
        if (is_ordinal) {
          word_str = IntToOrdinal(val);
        } else if (result.empty() && word.size() == 4) {
          word_str = IntToYear(val);
        } else {
          word_str = IntToCardinal(val);
        }
      } else if (dot_count == 0) {
        int64_t val = 0;
        (void)absl::SimpleAtoi(no_commas, &val);
        word_str = is_ordinal ? IntToOrdinal(val) : IntToCardinal(val);
      } else if (!no_commas.empty() && no_commas[0] == '.') {
        word_str = "point";
        for (size_t k = 1; k < no_commas.size(); ++k) {
          const char c = no_commas[k];
          if (c >= '0' && c <= '9') {
            absl::StrAppend(&word_str, " ", kOnes[c - '0']);
          }
        }
      } else {
        word_str = FloatToWords(no_commas);
      }
      ExtendNumberWords(word_str, /*first=*/true, /*escape=*/true, num_flags,
                        result);
    }

    if (result.empty()) {
      return {std::nullopt, 0};
    }
    std::string joined;
    int min_rating = 4;
    for (const PhonemeResult& item : result) {
      if (!item.phonemes.has_value() || item.phonemes->empty()) {
        continue;
      }
      if (!joined.empty()) {
        joined.push_back(' ');
      }
      joined.append(*item.phonemes);
      min_rating = std::min(min_rating, item.rating);
    }
    if (joined.empty()) {
      return {std::nullopt, 0};
    }
    if (suffix == "s" || suffix == "'s") {
      return {InflectS(joined), min_rating};
    }
    if (suffix == "ed" || suffix == "'d") {
      return {InflectEd(joined), min_rating};
    }
    if (suffix == "ing") {
      return {InflectIng(joined), min_rating};
    }
    return {std::move(joined), min_rating};
  }

  std::optional<std::string> AppendCurrency(
      const std::optional<std::string>& ps, absl::string_view currency) const {
    if (!ps.has_value() || currency.empty()) {
      return ps;
    }
    const CurrencyUnits* curr = FindCurrency(currency);
    if (curr == nullptr) {
      return ps;
    }
    const PhonemeResult unit_ps =
        StemS(absl::StrCat(curr->major, "s"), "", std::nullopt, nullptr);
    if (!unit_ps.phonemes.has_value() || unit_ps.phonemes->empty()) {
      return ps;
    }
    return absl::StrCat(*ps, " ", *unit_ps.phonemes);
  }

  static bool IsNumber(absl::string_view word, bool is_head) {
    bool any_digit = false;
    for (char c : word) {
      if (absl::ascii_isdigit(static_cast<unsigned char>(c))) {
        any_digit = true;
        break;
      }
    }
    if (!any_digit) {
      return false;
    }
    constexpr absl::string_view kSuffixes[] = {"ing", "'d", "ed", "'s", "st",
                                               "nd",  "rd", "th", "s"};
    for (absl::string_view s : kSuffixes) {
      if (absl::EndsWith(word, s)) {
        word.remove_suffix(s.size());
        break;
      }
    }
    if (word.empty()) {
      return false;
    }
    for (size_t i = 0; i < word.size(); ++i) {
      const char c = word[i];
      if (absl::ascii_isdigit(static_cast<unsigned char>(c)) || c == ',' ||
          c == '.' || (is_head && i == 0 && c == '-')) {
        continue;
      }
      return false;
    }
    return true;
  }

  PhonemeResult LookupToken(const MToken& t, const TokenContext& ctx) const {
    const std::string word =
        NormalizeWordUtf8(t.alias.has_value() ? *t.alias : t.text);
    const std::string low = ToLowerUtf8(word);
    std::optional<double> stress;
    if (word != low) {
      stress = (word == absl::AsciiStrToUpper(word)) ? 2.0 : 0.5;
    }
    PhonemeResult res = GetWord(word, t.tag, stress, &ctx);
    if (res.phonemes.has_value()) {
      return {ApplyStress(AppendCurrency(res.phonemes, t.currency), t.stress),
              res.rating};
    }
    if (IsNumber(word, t.is_head)) {
      res = GetNumber(word, t.currency, t.is_head, t.num_flags);
      return {ApplyStress(res.phonemes, t.stress), res.rating};
    }
    if (!AllLexiconOrds(word)) {
      return {std::nullopt, 0};
    }
    if (word != low &&
        (word == absl::AsciiStrToUpper(word) || IsRestLowerUtf8(word))) {
      res = GetWord(low, t.tag, stress, &ctx);
      if (res.phonemes.has_value()) {
        return {ApplyStress(AppendCurrency(res.phonemes, t.currency), t.stress),
                res.rating};
      }
    }
    return {std::nullopt, 0};
  }

 private:
  AccentLexiconTable table_;
  bool british_;
  const absl::flat_hash_map<std::string, std::string>& custom_lexicon_;
};

bool IsApostropheCp(char32_t cp) {
  return cp == U'\'' || cp == U'‘' || cp == U'’';
}

bool IsAsciiDigitCp(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

std::vector<std::string> Subtokenize(absl::string_view word) {
  const std::u32string u32 = Utf8ToU32(word);
  const size_t n = u32.size();
  std::vector<std::string> out;
  size_t i = 0;
  while (i < n) {
    if (i == 0 && IsApostropheCp(u32[i])) {
      size_t j = i;
      while (j < n && IsApostropheCp(u32[j])) {
        ++j;
      }
      out.push_back(CodePointsToUtf8(u32.substr(i, j - i)));
      i = j;
      continue;
    }
    if (IsUnicodeUpper(u32[i]) && i + 2 < n && IsUnicodeUpper(u32[i + 1]) &&
        IsUnicodeLower(u32[i + 2])) {
      out.push_back(CodePointsToUtf8(u32.substr(i, 1)));
      ++i;
      continue;
    }
    size_t j = i;
    if (j == 0 && u32[j] == U'-') {
      ++j;
    }
    bool matched_num = false;
    while (j < n) {
      if (IsAsciiDigitCp(u32[j])) {
        ++j;
        matched_num = true;
      } else if ((u32[j] == U',' || u32[j] == U'.') && j + 1 < n &&
                 IsAsciiDigitCp(u32[j + 1])) {
        j += 2;
        matched_num = true;
      } else {
        break;
      }
    }
    if (matched_num) {
      out.push_back(CodePointsToUtf8(u32.substr(i, j - i)));
      i = j;
      continue;
    }
    if (u32[i] == U'-' || u32[i] == U'_') {
      j = i;
      while (j < n && (u32[j] == U'-' || u32[j] == U'_')) {
        ++j;
      }
      out.push_back(CodePointsToUtf8(u32.substr(i, j - i)));
      i = j;
      continue;
    }
    if (IsApostropheCp(u32[i])) {
      j = i;
      while (j < n && IsApostropheCp(u32[j])) {
        ++j;
      }
      if (j - i >= 2 || j == n) {
        out.push_back(CodePointsToUtf8(u32.substr(i, j - i)));
      }
      i = j;
      continue;
    }
    if (IsUnicodeLetter(u32[i])) {
      j = i;
      while (j < n) {
        if (IsUnicodeLetter(u32[j])) {
          if (IsUnicodeLower(u32[j]) && j + 1 < n &&
              IsUnicodeUpper(u32[j + 1])) {
            ++j;
            break;
          }
          ++j;
        } else if (IsApostropheCp(u32[j]) && j + 1 < n &&
                   IsUnicodeLetter(u32[j + 1])) {
          if (IsUnicodeLower(u32[j + 1]) && j + 2 < n &&
              IsUnicodeUpper(u32[j + 2])) {
            j += 2;
            break;
          }
          j += 2;
        } else {
          break;
        }
      }
      out.push_back(CodePointsToUtf8(u32.substr(i, j - i)));
      i = j;
      continue;
    }
    out.push_back(CodePointsToUtf8(u32.substr(i, 1)));
    ++i;
  }
  return out;
}

bool InWordSet(absl::string_view word, const absl::string_view* set,
               size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (set[i] == word) {
      return true;
    }
  }
  return false;
}

constexpr absl::string_view kDottedAbbrs[] = {
    "dr.",   "mr.",   "mrs.", "ms.",  "prof.", "st.",   "jr.",   "sr.",
    "rev.",  "gen.",  "col.", "lt.",  "sgt.",  "capt.", "gov.",  "sen.",
    "rep.",  "jan.",  "feb.", "mar.", "apr.",  "jun.",  "jul.",  "aug.",
    "sep.",  "sept.", "oct.", "nov.", "dec.",  "vs.",   "v.",    "p.",
    "pp.",   "no.",   "nos.", "co.",  "inc.",  "ltd.",  "corp.", "bros.",
    "ph.d.", "m.d.",  "b.a.", "m.a.", "d.c.",  "u.s.",  "u.k.",  "u.n.",
    "e.u.",  "a.m.",  "p.m.", "e.g.", "i.e.",  "cf.",   "al.",
};

constexpr absl::string_view kSplitNoStopAbbrs[] = {
    "etc.",
    "vol.",
};

struct ContractionSplit {
  absl::string_view full;
  absl::string_view first;
  absl::string_view second;
};

constexpr ContractionSplit kContractions[] = {
    {"can't", "ca", "n't"},   {"cannot", "can", "not"}, {"won't", "wo", "n't"},
    {"shan't", "sha", "n't"}, {"ain't", "ai", "n't"},
};

struct PieceToken {
  std::string text;
  std::string tag;
};

bool IsClockTimeToken(absl::string_view s) {
  const size_t colon = s.find(':');
  if (colon == absl::string_view::npos || colon == 0 || colon > 2 ||
      colon + 3 != s.size()) {
    return false;
  }
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != colon && !absl::ascii_isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  return true;
}

bool IsMultiDotAcronym(absl::string_view low) {
  if (low.size() < 4 || low.back() != '.') {
    return false;
  }
  for (size_t i = 0; i < low.size(); i += 2) {
    if (i + 1 >= low.size() || low[i] < 'a' || low[i] > 'z' ||
        low[i + 1] != '.') {
      return false;
    }
  }
  return true;
}

bool ParseDegreeToken(absl::string_view sc, std::string& num_out,
                      std::string& unit_out) {
  // Matches ^(-?\d+)°([FCfc])$ where '°' is "\xC2\xB0" in UTF-8.
  constexpr absl::string_view kDegUtf8 = "°";
  const size_t deg_pos = sc.find(kDegUtf8);
  if (deg_pos == absl::string_view::npos || deg_pos == 0 ||
      deg_pos + kDegUtf8.size() + 1 != sc.size()) {
    return false;
  }
  const char u = sc.back();
  if (u != 'F' && u != 'C' && u != 'f' && u != 'c') {
    return false;
  }
  const absl::string_view num_part = sc.substr(0, deg_pos);
  for (size_t i = 0; i < num_part.size(); ++i) {
    if (i == 0 && num_part[i] == '-' && num_part.size() > 1) {
      continue;
    }
    if (!absl::ascii_isdigit(static_cast<unsigned char>(num_part[i]))) {
      return false;
    }
  }
  num_out = std::string(num_part);
  unit_out = std::string(1, u);
  return true;
}

bool MatchApostropheClitic(absl::string_view sc_norm, std::string& stem_out,
                           std::string& suf_out) {
  constexpr absl::string_view kClitics[] = {"'s",  "'ll", "'ve",
                                            "'re", "'d",  "'m"};
  const std::string low = ToLowerUtf8(sc_norm);
  for (absl::string_view clitic : kClitics) {
    if (low.size() > clitic.size() && absl::EndsWith(low, clitic)) {
      const size_t cut = sc_norm.size() - clitic.size();
      stem_out = std::string(sc_norm.substr(0, cut));
      suf_out = std::string(sc_norm.substr(cut));
      return true;
    }
  }
  return false;
}

std::vector<MToken> RuleTokenizeRaw(absl::string_view text) {
  std::vector<MToken> tokens;
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    if (absl::ascii_isspace(static_cast<unsigned char>(text[i]))) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < n && !absl::ascii_isspace(static_cast<unsigned char>(text[j]))) {
      ++j;
    }
    size_t ws_end = j;
    while (ws_end < n &&
           absl::ascii_isspace(static_cast<unsigned char>(text[ws_end]))) {
      ++ws_end;
    }
    const std::string chunk_ws = (ws_end > j) ? " " : "";
    absl::string_view chunk = text.substr(i, j - i);
    i = ws_end;

    std::vector<PieceToken> lead_toks;
    while (!chunk.empty()) {
      if (absl::StartsWith(chunk, "...")) {
        lead_toks.push_back({"...", ":"});
        chunk.remove_prefix(3);
      } else if (chunk[0] == '(' || chunk[0] == '[' || chunk[0] == '{') {
        lead_toks.push_back({std::string(1, chunk[0]), "-LRB-"});
        chunk.remove_prefix(1);
      } else if (chunk[0] == '"') {
        lead_toks.push_back({"\"", "``"});
        chunk.remove_prefix(1);
      } else if (absl::StartsWith(chunk, "“")) {
        lead_toks.push_back({"“", "``"});
        chunk.remove_prefix(3);
      } else if (chunk[0] == '$') {
        lead_toks.push_back({"$", "$"});
        chunk.remove_prefix(1);
      } else if (absl::StartsWith(chunk, "£")) {
        lead_toks.push_back({"£", "$"});
        chunk.remove_prefix(2);
      } else if (absl::StartsWith(chunk, "€")) {
        lead_toks.push_back({"€", "$"});
        chunk.remove_prefix(3);
      } else if (chunk[0] == '\'' || absl::StartsWith(chunk, "‘")) {
        const size_t p_len = chunk[0] == '\'' ? 1 : 3;
        if (chunk.size() > p_len) {
          lead_toks.push_back({"'", "``"});
          chunk.remove_prefix(p_len);
        } else {
          break;
        }
      } else {
        break;
      }
    }

    std::vector<PieceToken> trail_toks;
    while (!chunk.empty()) {
      if (absl::EndsWith(chunk, "...")) {
        trail_toks.push_back({"...", ":"});
        chunk.remove_suffix(3);
      } else if (absl::EndsWith(chunk, "…")) {
        trail_toks.push_back({"…", ":"});
        chunk.remove_suffix(3);
      } else if (chunk.back() == ')' || chunk.back() == ']' ||
                 chunk.back() == '}') {
        trail_toks.push_back({std::string(1, chunk.back()), "-RRB-"});
        chunk.remove_suffix(1);
      } else if (chunk.back() == '"') {
        trail_toks.push_back({"\"", "''"});
        chunk.remove_suffix(1);
      } else if (absl::EndsWith(chunk, "”")) {
        trail_toks.push_back({"”", "''"});
        chunk.remove_suffix(3);
      } else if (chunk.back() == ',') {
        trail_toks.push_back({",", ","});
        chunk.remove_suffix(1);
      } else if (chunk.back() == ';' || chunk.back() == ':') {
        if (chunk.back() == ':' && IsClockTimeToken(chunk)) {
          break;
        }
        if (chunk.back() == ':' && chunk.size() > 1 &&
            IsClockTimeToken(chunk.substr(0, chunk.size() - 1))) {
          trail_toks.push_back({":", ":"});
          chunk.remove_suffix(1);
        } else if (!IsClockTimeToken(chunk)) {
          trail_toks.push_back({std::string(1, chunk.back()), ":"});
          chunk.remove_suffix(1);
        } else {
          break;
        }
      } else if (chunk.back() == '!' || chunk.back() == '?') {
        trail_toks.push_back({std::string(1, chunk.back()), "."});
        chunk.remove_suffix(1);
      } else if (chunk.back() == '.' && chunk.size() > 1) {
        const std::string low_c = ToLowerUtf8(chunk);
        if (InWordSet(low_c, kDottedAbbrs,
                      sizeof(kDottedAbbrs) / sizeof(kDottedAbbrs[0])) ||
            IsMultiDotAcronym(low_c)) {
          break;
        }
        if (InWordSet(
                low_c, kSplitNoStopAbbrs,
                sizeof(kSplitNoStopAbbrs) / sizeof(kSplitNoStopAbbrs[0])) &&
            !trail_toks.empty()) {
          trail_toks.push_back({".", "FW"});
          chunk.remove_suffix(1);
        } else {
          trail_toks.push_back({".", "."});
          chunk.remove_suffix(1);
        }
      } else if ((chunk.back() == '\'' || absl::EndsWith(chunk, "’")) &&
                 chunk.size() > 1 &&
                 !absl::EndsWith(ToLowerUtf8(chunk), "s'") &&
                 !absl::EndsWith(ToLowerUtf8(chunk), "s’")) {
        const size_t s_len = chunk.back() == '\'' ? 1 : 3;
        if (chunk.size() > s_len) {
          trail_toks.push_back({"'", "''"});
          chunk.remove_suffix(s_len);
        } else {
          break;
        }
      } else {
        break;
      }
    }
    std::reverse(trail_toks.begin(), trail_toks.end());

    std::vector<PieceToken> mid_parts;
    if (!chunk.empty()) {
      constexpr absl::string_view kEmDash = "—";
      size_t start = 0;
      while (start <= chunk.size()) {
        const size_t dash_pos = chunk.find(kEmDash, start);
        const absl::string_view sc =
            dash_pos == absl::string_view::npos
                ? chunk.substr(start)
                : chunk.substr(start, dash_pos - start);
        if (!sc.empty()) {
          std::string deg_num;
          std::string deg_unit;
          if (ParseDegreeToken(sc, deg_num, deg_unit)) {
            if (!deg_num.empty() && deg_num[0] == '-') {
              mid_parts.push_back({deg_num, "HYPH"});
              mid_parts.push_back({"°", "``"});
              mid_parts.push_back({deg_unit, "NNP"});
            } else {
              mid_parts.push_back({deg_num, "CD"});
              mid_parts.push_back({"°", "CD"});
              mid_parts.push_back({deg_unit, "NNP"});
            }
          } else if (sc.size() > 1 && sc.back() == '%') {
            mid_parts.push_back(
                {std::string(sc.substr(0, sc.size() - 1)), "XX"});
            mid_parts.push_back({"%", "NN"});
          } else {
            const std::string sc_norm = NormalizeWordUtf8(sc);
            const std::string sc_low = ToLowerUtf8(sc_norm);
            bool handled_contraction = false;
            for (const ContractionSplit& cs : kContractions) {
              if (sc_low == cs.full) {
                std::string p1(cs.first);
                if (!sc_norm.empty() &&
                    absl::ascii_isupper(
                        static_cast<unsigned char>(sc_norm[0]))) {
                  p1 = CapitalizeUtf8(p1);
                }
                mid_parts.push_back({std::move(p1), "XX"});
                mid_parts.push_back({std::string(cs.second), "RB"});
                handled_contraction = true;
                break;
              }
            }
            if (!handled_contraction && sc_norm.size() > 3 &&
                absl::EndsWith(sc_low, "n't")) {
              mid_parts.push_back(
                  {sc_norm.substr(0, sc_norm.size() - 3), "XX"});
              mid_parts.push_back({sc_norm.substr(sc_norm.size() - 3), "RB"});
              handled_contraction = true;
            }
            std::string clitic_stem;
            std::string clitic_suf;
            if (!handled_contraction &&
                MatchApostropheClitic(sc_norm, clitic_stem, clitic_suf)) {
              const std::string suf_tag =
                  absl::EqualsIgnoreCase(clitic_suf, "'s") ? "POS" : "VBP";
              mid_parts.push_back({std::move(clitic_stem), "XX"});
              mid_parts.push_back({std::move(clitic_suf), suf_tag});
              handled_contraction = true;
            }
            if (!handled_contraction && absl::StrContains(sc, '-') &&
                sc.front() != '-' && sc.back() != '-' &&
                !absl::StrContains(sc, "--")) {
              std::vector<absl::string_view> hparts = absl::StrSplit(sc, '-');
              bool all_digits = true;
              bool all_alpha = true;
              for (absl::string_view hp : hparts) {
                if (hp.empty()) {
                  all_digits = false;
                  all_alpha = false;
                  break;
                }
                for (char c : hp) {
                  if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
                    all_digits = false;
                  }
                  if (!absl::ascii_isalpha(static_cast<unsigned char>(c)) &&
                      c != '\'') {
                    all_alpha = false;
                  }
                }
              }
              if (all_digits) {
                for (size_t h_idx = 0; h_idx < hparts.size(); ++h_idx) {
                  if (h_idx > 0) {
                    mid_parts.push_back(
                        {"-", h_idx + 1 == hparts.size() ? "SYM" : "CD"});
                  }
                  mid_parts.push_back({std::string(hparts[h_idx]), "CD"});
                }
                handled_contraction = true;
              } else if (all_alpha) {
                for (size_t h_idx = 0; h_idx < hparts.size(); ++h_idx) {
                  if (h_idx > 0) {
                    mid_parts.push_back({"-", "HYPH"});
                  }
                  mid_parts.push_back({std::string(hparts[h_idx]), "XX"});
                }
                handled_contraction = true;
              }
            }
            if (!handled_contraction) {
              mid_parts.push_back({std::string(sc), "XX"});
            }
          }
        }
        if (dash_pos == absl::string_view::npos) {
          break;
        }
        mid_parts.push_back({"—", ":"});
        start = dash_pos + kEmDash.size();
      }
    }

    std::vector<PieceToken> all_pieces;
    all_pieces.reserve(lead_toks.size() + mid_parts.size() + trail_toks.size());
    for (PieceToken& pt : lead_toks) {
      all_pieces.push_back(std::move(pt));
    }
    for (PieceToken& pt : mid_parts) {
      all_pieces.push_back(std::move(pt));
    }
    for (PieceToken& pt : trail_toks) {
      all_pieces.push_back(std::move(pt));
    }
    for (size_t p_idx = 0; p_idx < all_pieces.size(); ++p_idx) {
      MToken mt;
      mt.text = std::move(all_pieces[p_idx].text);
      mt.tag = std::move(all_pieces[p_idx].tag);
      mt.whitespace = (p_idx + 1 == all_pieces.size()) ? chunk_ws : "";
      tokens.push_back(std::move(mt));
    }
  }
  return tokens;
}

constexpr absl::string_view kInfVerbCues[] = {
    "to",     "will",  "would", "can", "could", "should", "must",
    "may",    "might", "did",   "do",  "does",  "not",    "n't",
    "please", "let",   "ca",    "wo",  "'ll",   "'d",
};

constexpr absl::string_view kFiniteVerbCuesVbp[] = {
    "i",
    "we",
    "you",
    "they",
};

constexpr absl::string_view kFiniteVerbCuesVbz[] = {
    "he",
    "she",
    "it",
    "who",
};

constexpr absl::string_view kNounCues[] = {
    "the",  "a",     "an",     "this", "that",      "these", "those", "his",
    "her",  "my",    "your",   "our",  "their",     "its",   "no",    "every",
    "each", "some",  "any",    "of",   "new",       "old",   "good",  "bad",
    "big",  "small", "strict", "ugly", "beautiful",
};

constexpr absl::string_view kCopulaCues[] = {
    "is",     "was",    "were",    "are",      "be",   "been",
    "became", "become", "becomes", "declared", "seem", "seems",
    "seemed", "feel",   "feels",   "felt",
};

constexpr absl::string_view kDegreeAdvCues[] = {
    "too",  "very",  "so",  "quite", "rather",    "more",   "most",
    "less", "least", "how", "as",    "extremely", "really",
};

constexpr absl::string_view kPredAdjHeteronyms[] = {
    "invalid",    "content", "subject", "close",       "separate", "elaborate",
    "deliberate", "minute",  "compact", "appropriate", "moderate", "intimate",
};

constexpr absl::string_view kAttrAdjHeteronyms[] = {
    "invalid",  "separate",   "elaborate",   "deliberate",  "minute",
    "compact",  "converse",   "animate",     "consummate",  "incorporate",
    "intimate", "moderate",   "appropriate", "approximate", "duplicate",
    "graduate", "alternate",  "articulate",  "associate",   "delegate",
    "desolate", "degenerate",
};

constexpr absl::string_view kNonNounFollowExtra[] = {
    "in",       "on",      "at",        "by",      "for",      "with",
    "from",     "into",    "onto",      "upon",    "about",    "above",
    "below",    "under",   "over",      "after",   "before",   "between",
    "among",    "through", "during",    "without", "within",   "against",
    "and",      "or",      "but",       "nor",     "yet",      "if",
    "when",     "while",   "because",   "since",   "until",    "unless",
    "although", "though",  "whether",   "which",   "what",     "where",
    "why",      "whom",    "whose",     "than",    "then",     "there",
    "here",     "now",     "yesterday", "today",   "tomorrow", "up",
    "down",     "out",     "off",       "away",    "back",     "only",
};

constexpr absl::string_view kKeepTags[] = {
    ".", ",", "-LRB-", "-RRB-", "``",   "\"\"", "''",  ":",
    "$", "#", "NFP",   "CD",    "HYPH", "POS",  "SYM", "FW",
};

bool IsNonNounFollow(absl::string_view lw) {
  return InWordSet(lw, kInfVerbCues,
                   sizeof(kInfVerbCues) / sizeof(kInfVerbCues[0])) ||
         InWordSet(
             lw, kFiniteVerbCuesVbp,
             sizeof(kFiniteVerbCuesVbp) / sizeof(kFiniteVerbCuesVbp[0])) ||
         InWordSet(
             lw, kFiniteVerbCuesVbz,
             sizeof(kFiniteVerbCuesVbz) / sizeof(kFiniteVerbCuesVbz[0])) ||
         InWordSet(lw, kNounCues, sizeof(kNounCues) / sizeof(kNounCues[0])) ||
         InWordSet(lw, kCopulaCues,
                   sizeof(kCopulaCues) / sizeof(kCopulaCues[0])) ||
         InWordSet(lw, kDegreeAdvCues,
                   sizeof(kDegreeAdvCues) / sizeof(kDegreeAdvCues[0])) ||
         InWordSet(
             lw, kNonNounFollowExtra,
             sizeof(kNonNounFollowExtra) / sizeof(kNonNounFollowExtra[0]));
}

void AssignPosTags(std::vector<MToken>& toks) {
  std::string prev;
  bool has_prev = false;
  bool prev_is_propn = false;
  const size_t n = toks.size();
  for (size_t idx = 0; idx < n; ++idx) {
    MToken& t = toks[idx];
    if (InWordSet(t.tag, kKeepTags, sizeof(kKeepTags) / sizeof(kKeepTags[0]))) {
      if (t.tag != "HYPH") {
        has_prev = false;
        prev.clear();
        prev_is_propn = false;
      }
      continue;
    }
    const std::string& w = t.text;
    const std::string lw = ToLowerUtf8(w);
    std::string next_lw;
    bool has_next_lw = false;
    if (idx + 1 < n && !InWordSet(toks[idx + 1].tag, kKeepTags,
                                  sizeof(kKeepTags) / sizeof(kKeepTags[0]))) {
      next_lw = ToLowerUtf8(toks[idx + 1].text);
      has_next_lw = true;
    }

    if (EnglishLexicon::IsNumber(w, /*is_head=*/true)) {
      t.tag = "CD";
    } else if (w == "I") {
      t.tag = "PRP";
    } else if (lw == "a" || lw == "an" || lw == "the") {
      t.tag = "DT";
    } else if (lw == "to") {
      t.tag = "TO";
    } else if (lw == "vs" || lw == "vs.") {
      t.tag = "IN";
    } else if (w.size() > 1 && IsAllUpperAlphaAscii(w)) {
      t.tag = "NNP";
    } else if (lw == "used" && has_next_lw && next_lw == "to" && has_prev) {
      t.tag = "VBD";
    } else if (has_prev &&
               InWordSet(prev, kDegreeAdvCues,
                         sizeof(kDegreeAdvCues) / sizeof(kDegreeAdvCues[0]))) {
      t.tag = "JJ";
    } else if (has_prev &&
               InWordSet(prev, kCopulaCues,
                         sizeof(kCopulaCues) / sizeof(kCopulaCues[0])) &&
               InWordSet(lw, kPredAdjHeteronyms,
                         sizeof(kPredAdjHeteronyms) /
                             sizeof(kPredAdjHeteronyms[0]))) {
      t.tag = "JJ";
    } else if (has_prev &&
               InWordSet(prev, kInfVerbCues,
                         sizeof(kInfVerbCues) / sizeof(kInfVerbCues[0]))) {
      t.tag = "VB";
    } else if (has_prev && InWordSet(prev, kFiniteVerbCuesVbp,
                                     sizeof(kFiniteVerbCuesVbp) /
                                         sizeof(kFiniteVerbCuesVbp[0]))) {
      t.tag = "VBP";
    } else if (has_prev && InWordSet(prev, kFiniteVerbCuesVbz,
                                     sizeof(kFiniteVerbCuesVbz) /
                                         sizeof(kFiniteVerbCuesVbz[0]))) {
      t.tag = (lw == "read" || lw == "reread" || lw == "wound") ? "VBD" : "VBZ";
    } else if (prev_is_propn && lw == "lives") {
      t.tag = "VBZ";
    } else if (has_prev &&
               InWordSet(prev, kNounCues,
                         sizeof(kNounCues) / sizeof(kNounCues[0]))) {
      if (InWordSet(
              lw, kAttrAdjHeteronyms,
              sizeof(kAttrAdjHeteronyms) / sizeof(kAttrAdjHeteronyms[0])) &&
          has_next_lw && !IsNonNounFollow(next_lw)) {
        t.tag = "JJ";
      } else {
        t.tag = "NN";
      }
    } else if (idx == 0 && (lw == "wait" || lw == "call" || lw == "please")) {
      t.tag = "VB";
    } else {
      t.tag = "XX";
    }

    prev = lw;
    has_prev = true;
    const bool is_cap_word =
        !w.empty() && absl::ascii_isupper(static_cast<unsigned char>(w[0])) &&
        IsRestLowerUtf8(w);
    prev_is_propn = (idx > 0 && is_cap_word) || lw == "dr." || lw == "mr." ||
                    lw == "mrs." || lw == "ms." || lw == "prof.";
  }
}

bool IsPunctTag(absl::string_view tag) {
  constexpr absl::string_view kPunctTags[] = {
      ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
  };
  return InWordSet(tag, kPunctTags, sizeof(kPunctTags) / sizeof(kPunctTags[0]));
}

std::string PunctTagPhonemes(absl::string_view tag, absl::string_view text) {
  if (tag == "-LRB-") {
    return "(";
  }
  if (tag == "-RRB-") {
    return ")";
  }
  if (tag == "``") {
    return "“";
  }
  if (tag == "\"\"" || tag == "''") {
    return "”";
  }
  const std::u32string u32 = Utf8ToU32(text);
  std::u32string out;
  out.reserve(u32.size());
  for (char32_t cp : u32) {
    if (IsPunctPhoneme(cp)) {
      out.push_back(cp);
    }
  }
  return CodePointsToUtf8(out);
}

bool IsAsciiVowel(char c) {
  switch (absl::ascii_tolower(static_cast<unsigned char>(c))) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
      return true;
    default:
      return false;
  }
}

bool IsAsciiVowelOrY(char c) {
  return IsAsciiVowel(c) ||
         absl::ascii_tolower(static_cast<unsigned char>(c)) == 'y';
}

bool HasAnyAsciiVowelOrY(absl::string_view s) {
  for (char c : s) {
    if (IsAsciiVowelOrY(c)) {
      return true;
    }
  }
  return false;
}

void EmitLtsVowel(absl::string_view v, bool can_stress, bool& stressed,
                  std::string& out) {
  if (can_stress && !stressed) {
    out.append(kPrimaryStressUtf8);
    out.append(v);
    stressed = true;
  } else {
    out.append(v);
  }
}

std::string EnglishLts(absl::string_view raw_word, bool british) {
  std::string w = ToLowerUtf8(raw_word);
  while (!w.empty() &&
         (w.front() == '\'' || w.front() == '-' || w.front() == '.')) {
    w.erase(w.begin());
  }
  while (!w.empty() &&
         (w.back() == '\'' || w.back() == '-' || w.back() == '.')) {
    w.pop_back();
  }
  if (w.empty()) {
    return "";
  }

  bool magic_e = false;
  if (w.size() >= 3 && w.back() == 'e' && !IsAsciiVowelOrY(w[w.size() - 2]) &&
      IsAsciiVowel(w[w.size() - 3]) &&
      (w.size() == 3 || !IsAsciiVowel(w[w.size() - 4]))) {
    const absl::string_view tail2 = absl::string_view(w).substr(w.size() - 2);
    if (tail2 != "le" && tail2 != "re") {
      magic_e = true;
    } else if (w.size() >= 4 && !IsAsciiVowel(w[w.size() - 4])) {
      magic_e = true;
    }
  }

  const size_t limit = magic_e ? (w.size() - 1) : w.size();
  std::string out;
  out.reserve(w.size() * 2 + 4);
  bool stressed = false;
  size_t i = 0;

  while (i < limit) {
    const absl::string_view rem = absl::string_view(w).substr(i, limit - i);
    if (rem == "ia" && i > 0) {
      EmitLtsVowel("i", /*can_stress=*/false, stressed, out);
      EmitLtsVowel("ə", /*can_stress=*/false, stressed, out);
      i += 2;
      continue;
    }
    if (rem == "ey" && i > 0) {
      EmitLtsVowel("i", /*can_stress=*/false, stressed, out);
      i += 2;
      continue;
    }
    if (rem == "er" && i > 0) {
      EmitLtsVowel(british ? "ə" : "əɹ", /*can_stress=*/false, stressed, out);
      i += 2;
      continue;
    }
    if (rem == "mond" && i > 1 && stressed) {
      out.append("mənd");
      i += 4;
      continue;
    }
    if ((rem == "on" || rem == "an" || rem == "en") && i > 1 && stressed) {
      EmitLtsVowel("ə", /*can_stress=*/false, stressed, out);
      out.push_back('n');
      i += 2;
      continue;
    }
    if (absl::StartsWith(rem, "tion")) {
      out.append("ʃən");
      i += 4;
      continue;
    }
    if (absl::StartsWith(rem, "sion")) {
      out.append("ʒən");
      i += 4;
      continue;
    }
    if (absl::StartsWith(rem, "ture")) {
      out.append(british ? "ʧə" : "ʧəɹ");
      i += 4;
      continue;
    }
    if (absl::StartsWith(rem, "igh")) {
      EmitLtsVowel("I", /*can_stress=*/true, stressed, out);
      i += 3;
      continue;
    }
    if (absl::StartsWith(rem, "tch")) {
      out.append("ʧ");
      i += 3;
      continue;
    }
    if (absl::StartsWith(rem, "dge")) {
      out.append("ʤ");
      i += 3;
      continue;
    }
    if (absl::StartsWith(rem, "ñ")) {
      out.append("nj");
      i += 2;
      continue;
    }
    if (absl::StartsWith(rem, "é") || absl::StartsWith(rem, "è") ||
        absl::StartsWith(rem, "ê")) {
      EmitLtsVowel("A", /*can_stress=*/true, stressed, out);
      i += 2;
      continue;
    }
    if (rem.size() >= 2) {
      const absl::string_view dg = rem.substr(0, 2);
      if (dg == "sh") {
        out.append("ʃ");
        i += 2;
        continue;
      }
      if (dg == "ch") {
        out.append("ʧ");
        i += 2;
        continue;
      }
      if (dg == "th") {
        out.append("θ");
        i += 2;
        continue;
      }
      if (dg == "ph") {
        out.push_back('f');
        i += 2;
        continue;
      }
      if (dg == "ck") {
        out.push_back('k');
        i += 2;
        continue;
      }
      if (dg == "qu") {
        out.append("kw");
        i += 2;
        continue;
      }
      if (dg == "ng") {
        out.append("ŋ");
        i += 2;
        continue;
      }
      if (dg == "wh") {
        out.push_back('w');
        i += 2;
        continue;
      }
      if ((dg == "kn" || dg == "gn" || dg == "wr") && i == 0) {
        if (dg[1] == 'r') {
          out.append("ɹ");
        } else {
          out.push_back(dg[1]);
        }
        i += 2;
        continue;
      }
      if (dg == "ee" || dg == "ea") {
        EmitLtsVowel(british ? "iː" : "i", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "oo") {
        EmitLtsVowel(british ? "uː" : "u", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "ai" || dg == "ay") {
        EmitLtsVowel("A", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "ei" || dg == "ey") {
        EmitLtsVowel(dg == "ei" ? "I" : "A", /*can_stress=*/true, stressed,
                     out);
        i += 2;
        continue;
      }
      if (dg == "oa" || dg == "ow") {
        EmitLtsVowel(british ? "Q" : "O", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "ou") {
        EmitLtsVowel("W", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "oi" || dg == "oy") {
        EmitLtsVowel("Y", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "au" || dg == "aw") {
        EmitLtsVowel(british ? "ɔː" : "ɔ", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "ar") {
        EmitLtsVowel(british ? "ɑː" : "ɑɹ", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "er" || dg == "ir" || dg == "ur") {
        EmitLtsVowel(british ? "ɜː" : "ɜɹ", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      if (dg == "or") {
        EmitLtsVowel(british ? "ɔː" : "ɔɹ", /*can_stress=*/true, stressed, out);
        i += 2;
        continue;
      }
      constexpr absl::string_view kDoubles = "bcdfgklmnprstvz";
      if (dg[0] == dg[1] && absl::StrContains(kDoubles, dg[0])) {
        if (dg[0] == 'g') {
          out.append("ɡ");
        } else if (dg[0] == 'r') {
          out.append("ɹ");
        } else {
          out.push_back(dg[0]);
        }
        i += 2;
        continue;
      }
    }

    const char c = w[i];
    const bool is_magic_pos = magic_e && (i + 2 == limit);
    if (c == 'a') {
      EmitLtsVowel(is_magic_pos ? "A" : (british ? "a" : "æ"),
                   /*can_stress=*/true, stressed, out);
    } else if (c == 'e') {
      EmitLtsVowel(is_magic_pos ? (british ? "iː" : "i") : "ɛ",
                   /*can_stress=*/true, stressed, out);
    } else if (c == 'i') {
      EmitLtsVowel(is_magic_pos ? "I" : "ɪ", /*can_stress=*/true, stressed,
                   out);
    } else if (c == 'o') {
      EmitLtsVowel(is_magic_pos ? (british ? "Q" : "O") : (british ? "ɒ" : "ɑ"),
                   /*can_stress=*/true, stressed, out);
    } else if (c == 'u') {
      EmitLtsVowel(is_magic_pos ? (british ? "uː" : "u") : "ʌ",
                   /*can_stress=*/true, stressed, out);
    } else if (c == 'y') {
      if (i == 0) {
        out.push_back('j');
      } else if (i + 1 == limit) {
        EmitLtsVowel("i", /*can_stress=*/!stressed, stressed, out);
      } else {
        EmitLtsVowel("ɪ", /*can_stress=*/true, stressed, out);
      }
    } else if (c == 'c') {
      const char nxt = (i + 1 < w.size()) ? w[i + 1] : '\0';
      out.push_back((nxt == 'e' || nxt == 'i' || nxt == 'y') ? 's' : 'k');
    } else if (c == 'g') {
      const char nxt = (i + 1 < w.size()) ? w[i + 1] : '\0';
      if (nxt == 'e' || nxt == 'i' || nxt == 'y') {
        out.append("ʤ");
      } else {
        out.append("ɡ");
      }
    } else if (c == 'j') {
      out.append("ʤ");
    } else if (c == 'q') {
      out.push_back('k');
    } else if (c == 'r') {
      out.append("ɹ");
    } else if (c == 'x') {
      out.append("ks");
    } else {
      constexpr absl::string_view kPlainCons = "bdfhklmnpstvwz";
      if (absl::StrContains(kPlainCons, c)) {
        out.push_back(c);
      }
    }
    ++i;
  }
  return out;
}

constexpr WordPair kAbbrExpand[] = {
    {"lbs", "pounds"},      {"lb", "pound"},     {"tbsp", "tbsp"},
    {"oz", "oz"},           {"st.", "st."},      {"jr.", "jr."},
    {"sr", "sr"},           {"sr.", "sr."},      {"feb.", "feb"},
    {"jan.", "jan"},        {"mar.", "mar"},     {"apr.", "apr"},
    {"jun.", "jun"},        {"jul.", "jul"},     {"aug.", "aug"},
    {"sep.", "sep"},        {"sept.", "sept"},   {"oct.", "oct"},
    {"nov.", "nov"},        {"dec.", "dec"},     {"vol.", "vol"},
    {"fig.", "fig"},        {"mon.", "mon"},     {"dr.", "doctor"},
    {"mr.", "mister"},      {"mrs.", "missus"},  {"ms.", "miss"},
    {"prof.", "professor"}, {"vs.", "versus"},   {"etc.", "etcetera"},
    {"etc", "etcetera"},    {"ext", "ext"},      {"ext.", "ext"},
    {"corp.", "corp"},      {"=", "equals"},     {"km/h", "KM slash H"},
    {"°f", "degrees F"},    {"°c", "degrees C"},
};

constexpr WordPair kDirectOovUs[] = {
    {"st.", "sənt."},
    {"jr.", "ʤˌAˈɑɹ."},
    {"sr.", "ˌɛsˈɑɹ."},
    {"sr", "ˌɛsˈɑɹ"},
    {"oz", "ˈɑz"},
    {"tbsp", "tˈAbᵊlspˌun"},
    {"ext", "ˈɛkst"},
    {"fig", "fˈɪɡ"},
    {"vol", "vˈɑl"},
    {"mon", "mˈOn"},
    {"feb", "fˈɛb"},
    {"litert", "lˌItˌɑɹtˈi"},
    {"microsoft", "mˈIkɹəsˌɑft"},
    {"jude", "ʤˈud"},
    {"jalapeño", "ʤˌæləpˈinjO"},
    {"entendre", "ɑntˈɑndɹə"},
    {"d'état", "dɪtˈæt"},
    {"état", "ɪtˈæt"},
    {"hors", "hˈɔɹz"},
    {"d'oeuvres", "dˈɜɹvɹəz"},
    {"oeuvres", "ˈɜɹvɹəz"},
    {"je", "ʤˈi"},
    {"ne", "nˈi"},
    {"sais", "sˈIz"},
    {"quoi", "kwˈɑ"},
};

constexpr WordPair kDirectOovGb[] = {
    {"st.", "sənt."},
    {"jr.", "ʤˌAˈɑː."},
    {"sr.", "ˌɛsˈɑː."},
    {"sr", "ˌɛsˈɑː"},
    {"oz", "ˈɒz"},
    {"tbsp", "tˈAbᵊlspˌuːn"},
    {"ext", "ˈɛkst"},
    {"fig", "fˈɪɡ"},
    {"vol", "vˈɒl"},
    {"mon", "mˈɒn"},
    {"feb", "fˈɛb"},
    {"litert", "lˌItˌɑɹtˈi"},
    {"microsoft", "mˈIkɹəsˌɒft"},
    {"jude", "ʤˈuːd"},
    {"jalapeño", "ʤˌaləpˈiːnjQ"},
    {"croissants", "kwˈɑːsɑs"},
    {"entendre", "ɒntˈɒndɹə"},
    {"d'état", "dɪtˈat"},
    {"état", "ɪtˈat"},
    {"hors", "hˈɔːz"},
    {"d'oeuvres", "dˈɜːvɹəz"},
    {"oeuvres", "ˈɜːvɹəz"},
    {"je", "ʤˈiː"},
    {"ne", "nˈiː"},
    {"sais", "sˈIz"},
    {"quoi", "kwˈɑː"},
};

constexpr absl::string_view kPrefixes[] = {
    "micro", "multi",   "inter",  "super", "under",  "over",  "anti",
    "auto",  "semi",    "mini",   "mega",  "hyper",  "cyber", "trans",
    "ultra", "counter", "pseudo", "tele",  "photo",  "bio",   "geo",
    "neuro", "electro", "thermo", "astro", "crypto", "macro", "nano",
    "pico",  "poly",    "mono",   "omni",  "uni",    "tri",   "bi",
    "co",    "de",      "dis",    "mis",   "non",    "out",   "post",
    "pre",   "pro",     "re",     "sub",   "un",     "up",
};

constexpr WordPair kSpellingAlts[] = {
    {"our", "or"},       {"ours", "ors"},    {"oured", "ored"},
    {"ouring", "oring"}, {"ise", "ize"},     {"ises", "izes"},
    {"ised", "ized"},    {"ising", "izing"}, {"re", "er"},
};

enum class FeatureKind {
  kNone,
  kStress,
  kPhonemes,
  kNumFlags,
};

struct LinkFeature {
  FeatureKind kind = FeatureKind::kNone;
  double stress = 0.0;
  std::string str_value;
};

struct PreprocessOutput {
  std::string clean_text;
  std::vector<std::string> ws_tokens;
  absl::flat_hash_map<size_t, LinkFeature> features;
};

void AppendWhitespaceSplit(absl::string_view segment,
                           std::vector<std::string>& out) {
  size_t i = 0;
  const size_t n = segment.size();
  while (i < n) {
    while (i < n &&
           absl::ascii_isspace(static_cast<unsigned char>(segment[i]))) {
      ++i;
    }
    if (i >= n) {
      break;
    }
    size_t j = i;
    while (j < n &&
           !absl::ascii_isspace(static_cast<unsigned char>(segment[j]))) {
      ++j;
    }
    out.push_back(std::string(segment.substr(i, j - i)));
    i = j;
  }
}

LinkFeature ParseLinkFeature(absl::string_view f) {
  LinkFeature out;
  if (f.empty()) {
    return out;
  }
  if (f == "0.5" || f == "+0.5") {
    out.kind = FeatureKind::kStress;
    out.stress = 0.5;
    return out;
  }
  if (f == "-0.5") {
    out.kind = FeatureKind::kStress;
    out.stress = -0.5;
    return out;
  }
  const absl::string_view digits_part =
      (f[0] == '-' || f[0] == '+') ? f.substr(1) : f;
  if (!digits_part.empty()) {
    bool all_digits = true;
    for (char c : digits_part) {
      if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
        all_digits = false;
        break;
      }
    }
    if (all_digits) {
      int val = 0;
      if (absl::SimpleAtoi(f, &val)) {
        out.kind = FeatureKind::kStress;
        out.stress = static_cast<double>(val);
        return out;
      }
    }
  }
  if (f.size() > 1 && f.front() == '/' && f.back() == '/') {
    absl::string_view body = f.substr(1);
    while (!body.empty() && body.back() == '/') {
      body.remove_suffix(1);
    }
    out.kind = FeatureKind::kPhonemes;
    out.str_value = std::string(body);
    return out;
  }
  if (f.size() > 1 && f.front() == '#' && f.back() == '#') {
    absl::string_view body = f.substr(1);
    while (!body.empty() && body.back() == '#') {
      body.remove_suffix(1);
    }
    out.kind = FeatureKind::kNumFlags;
    out.str_value = std::string(body);
    return out;
  }
  return out;
}

PreprocessOutput PreprocessMarkdownLinks(absl::string_view raw_text) {
  PreprocessOutput out;
  absl::string_view text = absl::StripLeadingAsciiWhitespace(raw_text);
  out.clean_text.reserve(text.size());
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t open_br = text.find('[', pos);
    if (open_br == absl::string_view::npos) {
      break;
    }
    const size_t close_br = text.find(']', open_br + 1);
    if (close_br == absl::string_view::npos || close_br + 1 >= text.size() ||
        text[close_br + 1] != '(') {
      pos = open_br + 1;
      continue;
    }
    // Disallow nested '[' inside the link text so we match [^\]]+.
    const absl::string_view link_word =
        text.substr(open_br + 1, close_br - open_br - 1);
    if (link_word.empty() || absl::StrContains(link_word, '[')) {
      pos = open_br + 1;
      continue;
    }
    const size_t close_paren = text.find(')', close_br + 2);
    if (close_paren == absl::string_view::npos) {
      break;
    }
    const absl::string_view prefix = text.substr(pos, open_br - pos);
    out.clean_text.append(prefix);
    AppendWhitespaceSplit(prefix, out.ws_tokens);

    const absl::string_view feat_str =
        text.substr(close_br + 2, close_paren - close_br - 2);
    const LinkFeature feat = ParseLinkFeature(feat_str);
    if (feat.kind != FeatureKind::kNone) {
      out.features[out.ws_tokens.size()] = feat;
    }
    out.clean_text.append(link_word);
    out.ws_tokens.push_back(std::string(link_word));
    pos = close_paren + 1;
  }
  if (pos < text.size()) {
    const absl::string_view rem = text.substr(pos);
    out.clean_text.append(rem);
    AppendWhitespaceSplit(rem, out.ws_tokens);
  }
  return out;
}

struct TokenOrGroup {
  bool is_group = false;
  MToken single;
  std::vector<MToken> group;
};

struct StressCandidate {
  bool has_primary = false;
  int weight = 0;
  size_t index = 0;
};

bool CompareStressCandidate(const StressCandidate& a,
                            const StressCandidate& b) {
  if (a.has_primary != b.has_primary) {
    return static_cast<int>(a.has_primary) < static_cast<int>(b.has_primary);
  }
  if (a.weight != b.weight) {
    return a.weight < b.weight;
  }
  return a.index < b.index;
}

class EnglishG2pEngine {
 public:
  EnglishG2pEngine(
      AccentLexiconTable table, bool british,
      const absl::flat_hash_map<std::string, std::string>& custom_lexicon)
      : lexicon_(table, british, custom_lexicon), british_(british) {}

  std::string Run(absl::string_view text, bool preprocess) const {
    PreprocessOutput prep;
    if (preprocess) {
      prep = PreprocessMarkdownLinks(text);
    } else {
      prep.clean_text = std::string(text);
    }
    std::vector<MToken> tokens = Tokenize(prep);
    tokens = FoldLeft(tokens);
    std::vector<TokenOrGroup> groups = Retokenize(tokens);

    TokenContext ctx;
    for (size_t rev = groups.size(); rev > 0; --rev) {
      TokenOrGroup& wg = groups[rev - 1];
      if (!wg.is_group) {
        MToken& w = wg.single;
        if (!w.phonemes.has_value()) {
          const PhonemeResult lex_res = lexicon_.LookupToken(w, ctx);
          w.phonemes = lex_res.phonemes;
          if (lex_res.phonemes.has_value()) {
            w.rating = lex_res.rating;
          }
        }
        if (!w.phonemes.has_value()) {
          const PhonemeResult fb_res = Fallback(w);
          w.phonemes = fb_res.phonemes;
          w.rating = fb_res.rating;
        }
        ctx = UpdateTokenContext(ctx, w.phonemes, w);
        continue;
      }

      std::vector<MToken>& w = wg.group;
      size_t left = 0;
      size_t right = w.size();
      bool should_fallback = false;
      while (left < right) {
        bool any_preset = false;
        for (size_t k = left; k < right; ++k) {
          if (w[k].alias.has_value() || w[k].phonemes.has_value()) {
            any_preset = true;
            break;
          }
        }
        PhonemeResult res;
        MToken merged_span;
        if (!any_preset) {
          std::vector<MToken> span(w.begin() + left, w.begin() + right);
          merged_span = MergeTokens(span, std::nullopt);
          res = lexicon_.LookupToken(merged_span, ctx);
        }
        if (res.phonemes.has_value()) {
          w[left].phonemes = res.phonemes;
          w[left].rating = res.rating;
          for (size_t k = left + 1; k < right; ++k) {
            w[k].phonemes = "";
            w[k].rating = res.rating;
          }
          ctx = UpdateTokenContext(ctx, res.phonemes, merged_span);
          right = left;
          left = 0;
        } else if (left + 1 < right) {
          ++left;
        } else {
          --right;
          MToken& t = w[right];
          if (!t.phonemes.has_value()) {
            const std::u32string u32 = Utf8ToU32(t.text);
            bool all_junk = !u32.empty();
            for (char32_t cp : u32) {
              if (!IsSubtokenJunk(cp)) {
                all_junk = false;
                break;
              }
            }
            if (all_junk) {
              t.phonemes = "";
              t.rating = 3;
            } else {
              should_fallback = true;
              break;
            }
          }
          left = 0;
        }
      }
      if (should_fallback) {
        const MToken merged_all = MergeTokens(w, std::nullopt);
        const PhonemeResult fb_res = Fallback(merged_all);
        w[0].phonemes = fb_res.phonemes;
        w[0].rating = fb_res.rating;
        for (size_t j = 1; j < w.size(); ++j) {
          w[j].phonemes = "";
          w[j].rating = fb_res.rating;
        }
      } else {
        ResolveTokens(w);
      }
    }

    std::string result;
    result.reserve(text.size() * 2);
    for (const TokenOrGroup& wg : groups) {
      const MToken final_tok =
          wg.is_group ? MergeTokens(wg.group, kUnkUtf8) : wg.single;
      if (final_tok.phonemes.has_value()) {
        result.append(*final_tok.phonemes);
      } else {
        result.append(kUnkUtf8);
      }
      result.append(final_tok.whitespace);
    }
    return result;
  }

 private:
  static std::vector<MToken> Tokenize(const PreprocessOutput& prep) {
    std::vector<MToken> mutable_tokens = RuleTokenizeRaw(prep.clean_text);
    AssignPosTags(mutable_tokens);
    if (prep.features.empty()) {
      return mutable_tokens;
    }
    size_t ws_idx = 0;
    size_t sub_idx = 0;
    for (MToken& mt : mutable_tokens) {
      if (const auto it = prep.features.find(ws_idx);
          it != prep.features.end()) {
        const LinkFeature& f = it->second;
        if (f.kind == FeatureKind::kStress) {
          mt.stress = f.stress;
        } else if (f.kind == FeatureKind::kPhonemes) {
          mt.is_head = (sub_idx == 0);
          mt.phonemes = (sub_idx == 0) ? f.str_value : "";
          mt.rating = 5;
        } else if (f.kind == FeatureKind::kNumFlags) {
          mt.num_flags = f.str_value;
        }
      }
      if (!mt.whitespace.empty()) {
        ++ws_idx;
        sub_idx = 0;
      } else {
        ++sub_idx;
      }
    }
    return mutable_tokens;
  }

  static std::vector<MToken> FoldLeft(const std::vector<MToken>& tokens) {
    std::vector<MToken> result;
    result.reserve(tokens.size());
    for (const MToken& t : tokens) {
      if (!result.empty() && !t.is_head) {
        std::vector<MToken> pair = {std::move(result.back()), t};
        result.pop_back();
        result.push_back(MergeTokens(pair, kUnkUtf8));
      } else {
        result.push_back(t);
      }
    }
    return result;
  }

  static std::vector<TokenOrGroup> Retokenize(
      const std::vector<MToken>& tokens) {
    std::vector<TokenOrGroup> words;
    words.reserve(tokens.size());
    std::string currency;
    for (size_t i = 0; i < tokens.size(); ++i) {
      const MToken& token = tokens[i];
      std::vector<MToken> ts;
      if (!token.alias.has_value() && !token.phonemes.has_value()) {
        const std::vector<std::string> subs = Subtokenize(token.text);
        ts.reserve(subs.size());
        for (const std::string& sub : subs) {
          MToken copy = token;
          copy.text = sub;
          copy.whitespace.clear();
          ts.push_back(std::move(copy));
        }
      } else {
        ts.push_back(token);
      }
      if (ts.empty()) {
        continue;
      }
      ts.back().whitespace = token.whitespace;

      for (size_t j = 0; j < ts.size(); ++j) {
        MToken& t = ts[j];
        if (t.alias.has_value() || t.phonemes.has_value()) {
          // Keep pre-assigned alias or phonemes.
        } else if (t.tag == "$" && FindCurrency(t.text) != nullptr) {
          currency = t.text;
          t.phonemes = "";
          t.rating = 4;
        } else if (t.tag == ":" && (t.text == "-" || t.text == "–")) {
          t.phonemes = "—";
          t.rating = 3;
        } else if (IsPunctTag(t.tag)) {
          t.phonemes = PunctTagPhonemes(t.tag, t.text);
          t.rating = 4;
        } else if (!currency.empty()) {
          if (t.tag != "CD") {
            currency.clear();
          } else if (j + 1 == ts.size() &&
                     (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) {
            t.currency = currency;
          }
        } else if (j > 0 && j + 1 < ts.size() && t.text == "2" &&
                   !ts[j - 1].text.empty() && !ts[j + 1].text.empty() &&
                   absl::ascii_isalpha(
                       static_cast<unsigned char>(ts[j - 1].text.back())) &&
                   absl::ascii_isalpha(
                       static_cast<unsigned char>(ts[j + 1].text.front()))) {
          t.alias = "to";
        }

        if (t.alias.has_value() || t.phonemes.has_value()) {
          words.push_back({/*is_group=*/false, std::move(t), {}});
        } else if (!words.empty() && words.back().is_group &&
                   !words.back().group.empty() &&
                   words.back().group.back().whitespace.empty()) {
          t.is_head = false;
          words.back().group.push_back(std::move(t));
        } else if (!t.whitespace.empty()) {
          words.push_back({/*is_group=*/false, std::move(t), {}});
        } else {
          words.push_back({/*is_group=*/true, {}, {std::move(t)}});
        }
      }
    }
    for (TokenOrGroup& wg : words) {
      if (wg.is_group && wg.group.size() == 1) {
        wg.single = std::move(wg.group[0]);
        wg.group.clear();
        wg.is_group = false;
      }
    }
    return words;
  }

  static TokenContext UpdateTokenContext(const TokenContext& ctx,
                                         const std::optional<std::string>& ps,
                                         const MToken& token) {
    std::optional<bool> vowel = ctx.future_vowel;
    if (ps.has_value() && !ps->empty()) {
      const std::u32string u32 = Utf8ToU32(*ps);
      for (char32_t cp : u32) {
        if (IsVowelPhoneme(cp) || IsConsonantPhoneme(cp) ||
            IsNonQuotePunctPhoneme(cp)) {
          if (IsNonQuotePunctPhoneme(cp)) {
            vowel = std::nullopt;
          } else {
            vowel = IsVowelPhoneme(cp);
          }
          break;
        }
      }
    }
    return {vowel, token.IsTo()};
  }

  static void ResolveTokens(std::vector<MToken>& tokens) {
    if (tokens.empty()) {
      return;
    }
    std::string text;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      text.append(tokens[i].text);
      text.append(tokens[i].whitespace);
    }
    text.append(tokens.back().text);

    bool has_space_or_slash = false;
    int char_classes_mask = 0;
    const std::u32string u32 = Utf8ToU32(text);
    for (char32_t cp : u32) {
      if (cp == U' ' || cp == U'/') {
        has_space_or_slash = true;
      }
      if (!IsSubtokenJunk(cp)) {
        if (IsUnicodeLetter(cp)) {
          char_classes_mask |= 1;
        } else if (IsAsciiDigitCp(cp)) {
          char_classes_mask |= 2;
        } else {
          char_classes_mask |= 4;
        }
      }
    }
    const int distinct_classes = ((char_classes_mask & 1) != 0) +
                                 ((char_classes_mask & 2) != 0) +
                                 ((char_classes_mask & 4) != 0);
    const bool prespace = has_space_or_slash || (distinct_classes > 1);

    for (size_t i = 0; i < tokens.size(); ++i) {
      MToken& t = tokens[i];
      if (!t.phonemes.has_value()) {
        const std::u32string t_u32 = Utf8ToU32(t.text);
        if (i + 1 == tokens.size() && t_u32.size() == 1 &&
            IsNonQuotePunctPhoneme(t_u32[0])) {
          t.phonemes = t.text;
          t.rating = 3;
        } else {
          bool all_junk = !t_u32.empty();
          for (char32_t cp : t_u32) {
            if (!IsSubtokenJunk(cp)) {
              all_junk = false;
              break;
            }
          }
          if (all_junk) {
            t.phonemes = "";
            t.rating = 3;
          }
        }
      } else if (i > 0) {
        t.prespace = prespace;
      }
    }
    if (prespace) {
      return;
    }

    std::vector<StressCandidate> indices;
    size_t primary_count = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i].phonemes.has_value() && !tokens[i].phonemes->empty()) {
        const bool has_p =
            tokens[i].phonemes->find(kPrimaryStressUtf8) != std::string::npos;
        if (has_p) {
          ++primary_count;
        }
        indices.push_back({has_p, tokens[i].StressWeight(), i});
      }
    }
    if (indices.size() == 2 &&
        Utf8ToU32(tokens[indices[0].index].text).size() == 1) {
      const size_t idx = indices[1].index;
      tokens[idx].phonemes = ApplyStress(tokens[idx].phonemes, -0.5);
      return;
    }
    if (indices.size() < 2 || primary_count <= (indices.size() + 1) / 2) {
      return;
    }
    std::sort(indices.begin(), indices.end(), CompareStressCandidate);
    const size_t half = indices.size() / 2;
    for (size_t k = 0; k < half; ++k) {
      const size_t idx = indices[k].index;
      tokens[idx].phonemes = ApplyStress(tokens[idx].phonemes, -0.5);
    }
  }

  std::string SubPhonemize(absl::string_view sub_text) const {
    std::string out = Run(sub_text, /*preprocess=*/false);
    return std::string(absl::StripAsciiWhitespace(out));
  }

  std::optional<absl::string_view> FindDirectOov(absl::string_view key) const {
    if (british_) {
      return FindInPairs(key, kDirectOovGb,
                         sizeof(kDirectOovGb) / sizeof(kDirectOovGb[0]));
    }
    return FindInPairs(key, kDirectOovUs,
                       sizeof(kDirectOovUs) / sizeof(kDirectOovUs[0]));
  }

  PhonemeResult Fallback(const MToken& token) const {
    const std::string t = NormalizeWordUtf8(token.text);
    std::string deg_num;
    std::string deg_unit;
    if (ParseDegreeToken(t, deg_num, deg_unit)) {
      const std::string num_ipa = SubPhonemize(deg_num);
      const absl::string_view deg_ipa = british_ ? "dɪɡɹˈiːz" : "dᵻɡɹˈiz";
      const std::string unit_ipa =
          SubPhonemize(absl::AsciiStrToUpper(deg_unit));
      return {absl::StrCat(num_ipa, " ", deg_ipa, " ", unit_ipa), 2};
    }

    const std::string low = ToLowerUtf8(t);
    if (const auto direct = FindDirectOov(low); direct.has_value()) {
      return {std::string(*direct), 2};
    }

    if (IsClockTimeToken(t)) {
      const size_t colon = t.find(':');
      const absl::string_view h_str = absl::string_view(t).substr(0, colon);
      const absl::string_view mm = absl::string_view(t).substr(colon + 1);
      constexpr absl::string_view kZeroIpa = "zˈiəɹO";
      const absl::string_view zero_ipa = british_ ? "zˈiəɹQ" : kZeroIpa;
      std::string h_ipa;
      if (h_str.size() == 2 && h_str[0] == '0') {
        h_ipa = absl::StrCat(zero_ipa, " ", SubPhonemize(h_str.substr(1)));
      } else {
        int h_val = 0;
        (void)absl::SimpleAtoi(h_str, &h_val);
        h_ipa = SubPhonemize(absl::StrCat(h_val));
      }
      std::string m_ipa;
      if (mm == "00") {
        m_ipa = absl::StrCat(zero_ipa, " ", zero_ipa);
      } else if (mm[0] == '0') {
        m_ipa = absl::StrCat(zero_ipa, " ", SubPhonemize(mm.substr(1)));
      } else if (mm == "15") {
        m_ipa = british_ ? "fˈɪftiːn" : "fˈɪftin";
      } else {
        m_ipa = SubPhonemize(mm);
      }
      return {absl::StrCat(h_ipa, ":", m_ipa), 2};
    }

    if (const auto exp = FindInPairs(
            low, kAbbrExpand, sizeof(kAbbrExpand) / sizeof(kAbbrExpand[0]));
        exp.has_value()) {
      std::string out;
      if (const auto direct = FindDirectOov(*exp); direct.has_value()) {
        out = std::string(*direct);
      } else {
        out = SubPhonemize(*exp);
      }
      if (absl::EndsWith(t, ".") && !absl::EndsWith(out, ".")) {
        out.push_back('.');
      }
      return {std::move(out), 2};
    }

    absl::string_view core(t);
    while (!core.empty() && core.back() == '.') {
      core.remove_suffix(1);
    }
    const std::string core_low = ToLowerUtf8(core);
    if (const auto direct = FindDirectOov(core_low); direct.has_value()) {
      std::string out(*direct);
      if (absl::EndsWith(t, ".") && !absl::EndsWith(out, ".")) {
        out.push_back('.');
      }
      return {std::move(out), 2};
    }

    // Comparative -er on known stems (e.g. nobler -> noble + er).
    if (core_low.size() > 4 && absl::EndsWith(core_low, "er")) {
      const std::string cand1 = core_low.substr(0, core_low.size() - 2);
      const std::string cand2 = core_low.substr(0, core_low.size() - 1);
      const std::string cand3 =
          absl::StrCat(core_low.substr(0, core_low.size() - 3), "y");
      const std::string* cands[] = {&cand1, &cand2, &cand3};
      for (const std::string* cand : cands) {
        if (lexicon_.IsKnown(*cand, "")) {
          const PhonemeResult stem_res =
              lexicon_.Lookup(*cand, "", std::nullopt, nullptr);
          if (stem_res.phonemes.has_value() && !stem_res.phonemes->empty()) {
            std::string stem_ps = *stem_res.phonemes;
            if (absl::EndsWith(stem_ps, "ᵊl")) {
              stem_ps.resize(stem_ps.size() - std::string("ᵊl").size());
              stem_ps.push_back('l');
            }
            absl::StrAppend(&stem_ps, british_ ? "ə" : "əɹ");
            if (absl::EndsWith(t, ".")) {
              stem_ps.push_back('.');
            }
            return {std::move(stem_ps), 2};
          }
        }
      }
    }

    // British <-> American spelling normalization (e.g. honour -> honor).
    for (const WordPair& sp : kSpellingAlts) {
      if (core_low.size() > sp.key.size() + 1 &&
          absl::EndsWith(core_low, sp.key)) {
        const std::string alt_w = absl::StrCat(
            core_low.substr(0, core_low.size() - sp.key.size()), sp.value);
        const PhonemeResult alt_res =
            lexicon_.GetWord(alt_w, "", std::nullopt, nullptr);
        if (alt_res.phonemes.has_value()) {
          std::string out = *alt_res.phonemes;
          if (absl::EndsWith(t, ".")) {
            out.push_back('.');
          }
          return {std::move(out), 2};
        }
      }
    }

    // Vowelless short acronym rule (<= 4 letters, no vowel).
    if (IsAllAlphaAscii(core) && core.size() <= 4 &&
        !HasAnyAsciiVowelOrY(core)) {
      const PhonemeResult nnp_res =
          lexicon_.GetNnp(absl::AsciiStrToUpper(core));
      if (nnp_res.phonemes.has_value()) {
        std::string out = *nnp_res.phonemes;
        if (absl::EndsWith(t, ".")) {
          out.push_back('.');
        }
        return {std::move(out), 2};
      }
    }

    // Productive prefix + known stem decomposition (e.g. Microsoft ->
    // micro+soft).
    for (absl::string_view pref : kPrefixes) {
      if (core_low.size() > pref.size() + 2 &&
          absl::StartsWith(core_low, pref)) {
        const absl::string_view rem =
            absl::string_view(core_low).substr(pref.size());
        if (lexicon_.IsKnown(rem, "")) {
          const PhonemeResult p_res =
              lexicon_.Lookup(pref, "", std::nullopt, nullptr);
          const PhonemeResult r_res = lexicon_.Lookup(rem, "", -0.5, nullptr);
          if (p_res.phonemes.has_value() && r_res.phonemes.has_value()) {
            std::string out = absl::StrCat(*p_res.phonemes, *r_res.phonemes);
            if (absl::EndsWith(t, ".")) {
              out.push_back('.');
            }
            return {std::move(out), 2};
          }
        }
      }
    }

    std::string lts = EnglishLts(core, british_);
    if (!lts.empty()) {
      if (absl::EndsWith(t, ".")) {
        lts.push_back('.');
      }
      return {std::move(lts), 1};
    }
    return {std::string(kUnkUtf8), 1};
  }

  EnglishLexicon lexicon_;
  bool british_;
};

}  // namespace

EnglishG2p::EnglishG2p(SortedStringIndex us_keys, StringValueTable us_values,
                       SortedStringIndex gb_keys, StringValueTable gb_values)
    : us_keys_(us_keys),
      us_values_(us_values),
      gb_keys_(gb_keys),
      gb_values_(gb_values) {}

absl::StatusOr<EnglishG2p> EnglishG2p::Create() {
  static constexpr char kEmptyIndexBytes[4] = {0, 0, 0, 0};
  constexpr absl::string_view kEmptyIndex(kEmptyIndexBytes, 4);
  ABSL_ASSIGN_OR_RETURN(SortedStringIndex us_keys,
                        SortedStringIndex::Create(kEmptyIndex, ""));
  ABSL_ASSIGN_OR_RETURN(StringValueTable us_values,
                        StringValueTable::Create(kEmptyIndex, ""));
  ABSL_ASSIGN_OR_RETURN(SortedStringIndex gb_keys,
                        SortedStringIndex::Create(kEmptyIndex, ""));
  ABSL_ASSIGN_OR_RETURN(StringValueTable gb_values,
                        StringValueTable::Create(kEmptyIndex, ""));
  return EnglishG2p(us_keys, us_values, gb_keys, gb_values);
}

absl::StatusOr<EnglishG2p> EnglishG2p::Create(const CjkBlob& blob) {
  ABSL_ASSIGN_OR_RETURN(const absl::string_view us_idx, blob.Section("us-idx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view us_key, blob.Section("us-key"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view us_vidx,
                        blob.Section("us-vidx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view us_val, blob.Section("us-val"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view gb_idx, blob.Section("gb-idx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view gb_key, blob.Section("gb-key"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view gb_vidx,
                        blob.Section("gb-vidx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view gb_val, blob.Section("gb-val"));

  ABSL_ASSIGN_OR_RETURN(SortedStringIndex us_keys,
                        SortedStringIndex::Create(us_idx, us_key));
  ABSL_ASSIGN_OR_RETURN(StringValueTable us_values,
                        StringValueTable::Create(us_vidx, us_val));
  if (us_keys.Size() != us_values.Size()) {
    return absl::InvalidArgumentError(
        "us-idx and us-vidx entry counts do not match");
  }
  ABSL_ASSIGN_OR_RETURN(SortedStringIndex gb_keys,
                        SortedStringIndex::Create(gb_idx, gb_key));
  ABSL_ASSIGN_OR_RETURN(StringValueTable gb_values,
                        StringValueTable::Create(gb_vidx, gb_val));
  if (gb_keys.Size() != gb_values.Size()) {
    return absl::InvalidArgumentError(
        "gb-idx and gb-vidx entry counts do not match");
  }

  return EnglishG2p(us_keys, us_values, gb_keys, gb_values);
}

absl::StatusOr<std::string> EnglishG2p::TextToIpa(
    absl::string_view text, bool british,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  if (absl::StripAsciiWhitespace(text).empty()) {
    return std::string();
  }
  const AccentLexiconTable table =
      british ? AccentLexiconTable(gb_keys_, gb_values_)
              : AccentLexiconTable(us_keys_, us_values_);
  const EnglishG2pEngine engine(table, british, custom_lexicon);
  std::string out = engine.Run(text, /*preprocess=*/true);
  return std::string(absl::StripAsciiWhitespace(out));
}

absl::StatusOr<std::string> EnglishG2p::WordToIpa(absl::string_view word,
                                                  bool british) const {
  return TextToIpa(word, british);
}

}  // namespace litert::omni::tts::kokoro
