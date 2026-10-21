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

#include "omni/text2speech/kokoro/spanish_g2p.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/unicode_utils.h"

namespace litert::omni::tts::kokoro {
namespace {

// Decodes UTF-8 `s`, composing a vowel, u or n followed by a combining
// acute accent, diaeresis or tilde into the precomposed Spanish letter.
std::u32string Utf8ToComposedU32(absl::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  size_t pos = 0;
  while (pos < s.size()) {
    const Utf8CodePoint dec = DecodeUtf8CodePoint(s, pos);
    if (dec.code_point == 0x0301 && !out.empty()) {
      // Combining acute accent -> precomposed Spanish vowel.
      switch (out.back()) {
        case U'a':
          out.back() = U'á';
          pos += dec.length;
          continue;
        case U'e':
          out.back() = U'é';
          pos += dec.length;
          continue;
        case U'i':
          out.back() = U'í';
          pos += dec.length;
          continue;
        case U'o':
          out.back() = U'ó';
          pos += dec.length;
          continue;
        case U'u':
          out.back() = U'ú';
          pos += dec.length;
          continue;
        case U'A':
          out.back() = U'Á';
          pos += dec.length;
          continue;
        case U'E':
          out.back() = U'É';
          pos += dec.length;
          continue;
        case U'I':
          out.back() = U'Í';
          pos += dec.length;
          continue;
        case U'O':
          out.back() = U'Ó';
          pos += dec.length;
          continue;
        case U'U':
          out.back() = U'Ú';
          pos += dec.length;
          continue;
        default:
          break;
      }
    } else if (dec.code_point == 0x0308 && !out.empty()) {
      if (out.back() == U'u') {
        out.back() = U'ü';
        pos += dec.length;
        continue;
      }
      if (out.back() == U'U') {
        out.back() = U'Ü';
        pos += dec.length;
        continue;
      }
    } else if (dec.code_point == 0x0303 && !out.empty()) {
      if (out.back() == U'n') {
        out.back() = U'ñ';
        pos += dec.length;
        continue;
      }
      if (out.back() == U'N') {
        out.back() = U'Ñ';
        pos += dec.length;
        continue;
      }
    }
    out.push_back(dec.code_point);
    pos += dec.length;
  }
  return out;
}

bool IsUpperSpanish(char32_t cp) {
  return (cp >= U'A' && cp <= U'Z') || cp == U'Á' || cp == U'É' || cp == U'Í' ||
         cp == U'Ó' || cp == U'Ú' || cp == U'Ü' || cp == U'Ñ';
}

bool IsSpanishLetter(char32_t cp) {
  const char32_t lc = ToLowerLatin(cp);
  return (lc >= U'a' && lc <= U'z') || lc == U'á' || lc == U'é' || lc == U'í' ||
         lc == U'ó' || lc == U'ú' || lc == U'ü' || lc == U'ñ' || lc == U'ª' ||
         lc == U'º';
}

bool IsAsciiDigitCp(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

bool IsWhitespaceCp(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' ||
         cp == 0x00A0;
}

std::u32string LowercaseU32(const std::u32string& s) {
  std::u32string out;
  out.reserve(s.size());
  for (char32_t cp : s) {
    out.push_back(ToLowerLatin(cp));
  }
  return out;
}

std::string LowercaseUtf8(absl::string_view s) {
  return CodePointsToUtf8(LowercaseU32(Utf8ToComposedU32(s)));
}

bool IsAllUppercaseWord(const std::u32string& w) {
  bool any_upper = false;
  for (char32_t cp : w) {
    if (IsUpperSpanish(cp)) {
      any_upper = true;
    } else if (IsSpanishLetter(cp)) {
      return false;
    }
  }
  return any_upper;
}

bool IsStrongVowel(char32_t cp) {
  return cp == U'a' || cp == U'e' || cp == U'o' || cp == U'á' || cp == U'é' ||
         cp == U'ó';
}

bool IsWeakVowel(char32_t cp) {
  return cp == U'i' || cp == U'u' || cp == U'ü' || cp == U'Y';
}

bool IsAccentedVowel(char32_t cp) {
  return cp == U'á' || cp == U'é' || cp == U'í' || cp == U'ó' || cp == U'ú';
}

bool IsVowelLetter(char32_t cp) {
  return IsStrongVowel(cp) || cp == U'i' || cp == U'u' || cp == U'ü' ||
         cp == U'í' || cp == U'ú';
}

char32_t UnaccentVowel(char32_t cp) {
  switch (cp) {
    case U'á':
      return U'a';
    case U'é':
      return U'e';
    case U'í':
      return U'i';
    case U'ó':
      return U'o';
    case U'ú':
    case U'ü':
      return U'u';
    default:
      return cp;
  }
}

enum class UnitKind { kVowel, kConsonant, kSilentH };

struct GraphemeUnit {
  UnitKind kind;
  char32_t vowel = 0;
  std::string cons;
};

enum class VowelRole { kGlide, kHiatusPre, kNucleus, kOffglide };

struct NucleusChar {
  char32_t ch;
  VowelRole role;
};

struct Segment {
  bool is_nucleus = false;
  std::string cons;
  std::vector<NucleusChar> nucleus;
};

enum class PhoneKind { kConsonant, kVowel, kLiteral };

struct PhoneToken {
  PhoneKind kind = PhoneKind::kConsonant;
  std::string phone;   // For kConsonant / kLiteral.
  std::string glides;  // For kVowel.
  std::string body;    // For kVowel.
  std::string stress;  // "", "ˈ", or "ˌ" for kVowel.
  bool no_open_e = false;
};

PhoneToken MakeCons(absl::string_view ph) {
  PhoneToken tok;
  tok.kind = PhoneKind::kConsonant;
  tok.phone = std::string(ph);
  return tok;
}

PhoneToken MakeVowel(absl::string_view glides, absl::string_view body,
                     absl::string_view stress, bool no_open_e = false) {
  PhoneToken tok;
  tok.kind = PhoneKind::kVowel;
  tok.glides = std::string(glides);
  tok.body = std::string(body);
  tok.stress = std::string(stress);
  tok.no_open_e = no_open_e;
  return tok;
}

PhoneToken MakeLiteral(absl::string_view ipa) {
  PhoneToken tok;
  tok.kind = PhoneKind::kLiteral;
  tok.phone = std::string(ipa);
  return tok;
}

bool IsAtonaMono(absl::string_view w) {
  static const auto* const kSet = new absl::flat_hash_set<std::string>({
      "el",  "la",  "los",  "las",   "lo", "al",   "del", "a",   "con",
      "de",  "en",  "por",  "sin",   "so", "tras", "y",   "e",   "ni",
      "o",   "u",   "que",  "mas",   "si", "pues", "me",  "te",  "se",
      "nos", "os",  "le",   "les",   "mi", "mis",  "tu",  "tus", "su",
      "sus", "tan", "cual", "quien", "do",
  });
  return kSet->contains(w);
}

bool IsAtonaPoly(absl::string_view w) {
  static const auto* const kSet = new absl::flat_hash_set<std::string>({
      "para",     "pero",    "porque",   "como",     "cuando",  "donde",
      "mientras", "aunque",  "sino",     "desde",    "hasta",   "hacia",
      "entre",    "sobre",   "contra",   "ante",     "bajo",    "mediante",
      "nuestro",  "nuestra", "nuestros", "nuestras", "vuestro", "vuestra",
      "cuanto",   "cuanta",  "cuantos",  "cuantas",  "según",   "aun",
      "luego",
  });
  return kSet->contains(w);
}

std::optional<int> LookupRomanNumeral(absl::string_view w) {
  static const auto* const kMap = new absl::flat_hash_map<std::string, int>({
      {"ii", 2},     {"iii", 3},     {"iv", 4},    {"v", 5},    {"vi", 6},
      {"vii", 7},    {"viii", 8},    {"ix", 9},    {"x", 10},   {"xi", 11},
      {"xii", 12},   {"xiii", 13},   {"xiv", 14},  {"xv", 15},  {"xvi", 16},
      {"xvii", 17},  {"xviii", 18},  {"xix", 19},  {"xx", 20},  {"xxi", 21},
      {"xxii", 22},  {"xxiii", 23},  {"xxiv", 24}, {"xxv", 25}, {"xxvi", 26},
      {"xxvii", 27}, {"xxviii", 28}, {"xxix", 29}, {"xxx", 30},
  });
  if (const auto it = kMap->find(w); it != kMap->end()) {
    return it->second;
  }
  return std::nullopt;
}

const absl::flat_hash_map<std::string, std::vector<PhoneToken>>&
GetSpanishLexicon() {
  static const auto* const kLex = new absl::flat_hash_map<
      std::string, std::vector<PhoneToken>>({
      {"méxico",
       {MakeCons("m"), MakeVowel("", "ɛ", "ˈ", true), MakeCons("x"),
        MakeVowel("", "i", ""), MakeCons("k"), MakeVowel("", "o", "")}},
      {"mexicano",
       {MakeCons("m"), MakeVowel("", "e", "ˌ"), MakeCons("x"),
        MakeVowel("", "i", ""), MakeCons("k"), MakeVowel("", "a", "ˈ"),
        MakeCons("n"), MakeVowel("", "o", "")}},
      {"mexicana",
       {MakeCons("m"), MakeVowel("", "e", "ˌ"), MakeCons("x"),
        MakeVowel("", "i", ""), MakeCons("k"), MakeVowel("", "a", "ˈ"),
        MakeCons("n"), MakeVowel("", "a", "")}},
      {"mexicanos",
       {MakeCons("m"), MakeVowel("", "e", "ˌ"), MakeCons("x"),
        MakeVowel("", "i", ""), MakeCons("k"), MakeVowel("", "a", "ˈ"),
        MakeCons("n"), MakeVowel("", "o", ""), MakeCons("s")}},
      {"mexicanas",
       {MakeCons("m"), MakeVowel("", "e", "ˌ"), MakeCons("x"),
        MakeVowel("", "i", ""), MakeCons("k"), MakeVowel("", "a", "ˈ"),
        MakeCons("n"), MakeVowel("", "a", ""), MakeCons("s")}},
      {"ser", {MakeCons("s"), MakeVowel("", "e", "ˈ", true), MakeCons("r")}},
      {"whisky",
       {MakeCons("w"), MakeVowel("", "i", "ˈ"), MakeCons("s"), MakeCons("k"),
        MakeVowel("", "i", "")}},
      {"software",
       {MakeCons("s"), MakeVowel("", "o", "ˈ"), MakeCons("f"), MakeCons("t"),
        MakeCons("w"), MakeVowel("", "e", ""), MakeCons("ɾ")}},
      {"jazz", {MakeCons("ʤ"), MakeVowel("", "a", "ˈ"), MakeCons("s")}},
      {"show", {MakeCons("ʃ"), MakeVowel("", "o", "ˈ"), MakeCons("w")}},
      {"marketing",
       {MakeCons("m"), MakeVowel("", "a", ""), MakeCons("ɾ"), MakeCons("k"),
        MakeVowel("", "e", "ˈ", true), MakeCons("t"), MakeVowel("", "i", ""),
        MakeCons("ŋ")}},
      {"parking",
       {MakeCons("p"), MakeVowel("", "a", "ˈ"), MakeCons("ɾ"), MakeCons("k"),
        MakeVowel("", "i", ""), MakeCons("ŋ")}},
      {"ranking",
       {MakeCons("r"), MakeVowel("", "a", "ˈ"), MakeCons("ŋ"), MakeCons("k"),
        MakeVowel("", "i", ""), MakeCons("ŋ")}},
      {"racing",
       {MakeCons("r"), MakeVowel("", "a", "ˈ"), MakeCons("θ"),
        MakeVowel("", "i", ""), MakeCons("ŋ")}},
      {"twitter",
       {MakeCons("t"), MakeCons("w"), MakeVowel("", "i", "ˈ"), MakeCons("t"),
        MakeVowel("", "e", ""), MakeCons("ɾ")}},
      {"blog",
       {MakeCons("B"), MakeCons("l"), MakeVowel("", "o", "ˈ"), MakeCons("ɡ")}},
      {"big", {MakeCons("B"), MakeVowel("", "i", "ˈ"), MakeCons("ɡ")}},
      {"link",
       {MakeCons("l"), MakeVowel("", "i", "ˈ"), MakeCons("ŋ"), MakeCons("k")}},
      {"frank",
       {MakeCons("f"), MakeCons("ɾ"), MakeVowel("", "a", "ˈ"), MakeCons("ŋ"),
        MakeCons("k")}},
      {"hong", {MakeVowel("", "o", "ˈ"), MakeCons("ŋ")}},
      {"kong", {MakeCons("k"), MakeVowel("", "o", "ˈ"), MakeCons("ŋ")}},
      {"youtube",
       {MakeCons("ʝ"), MakeVowel("", "ow", ""), MakeCons("t"),
        MakeVowel("", "u", "ˈ"), MakeCons("B"), MakeVowel("", "e", "")}},
      {"you", {MakeCons("ʝ"), MakeVowel("", "ow", "ˈ")}},
      {"harry",
       {MakeVowel("", "a", "ˈ"), MakeCons("r"), MakeVowel("", "i", "")}},
      {"mary",
       {MakeCons("m"), MakeVowel("", "a", "ˈ"), MakeCons("ɾ"),
        MakeVowel("", "i", "")}},
      {"city",
       {MakeCons("θ"), MakeVowel("", "i", "ˈ"), MakeCons("t"),
        MakeVowel("", "i", "")}},
      {"sexy",
       {MakeCons("s"), MakeVowel("", "e", "ˈ", true), MakeCons("k"),
        MakeCons("s"), MakeVowel("", "i", "")}},
      {"android",
       {MakeVowel("", "a", "ˈ"), MakeCons("n"), MakeCons("D"), MakeCons("ɾ"),
        MakeVowel("", "oɪ", ""), MakeCons("d")}},
      {"william",
       {MakeCons("w"), MakeVowel("", "i", "ˈ"), MakeCons("ʎ"),
        MakeVowel("", "i", ""), MakeVowel("", "a", ""), MakeCons("m")}},
      {"apple",
       {MakeVowel("", "a", "ˈ"), MakeCons("p"), MakeVowel("", "ə", ""),
        MakeCons("l")}},
      {"mac", {MakeCons("m"), MakeVowel("", "a", "ˈ"), MakeCons("k")}},
      {"free", {MakeCons("f"), MakeCons("ɾ"), MakeVowel("", "i", "ˈ")}},
      {"click",
       {MakeCons("k"), MakeCons("l"), MakeVowel("", "i", "ˈ"), MakeCons("k")}},
      {"live", {MakeCons("l"), MakeVowel("", "I", "ˈ"), MakeCons("B")}},
      {"team",
       {MakeCons("t"), MakeVowel("", "e", "ˈ"), MakeVowel("", "a", ""),
        MakeCons("m")}},
      {"junior",
       {MakeCons("ʤ"), MakeVowel("", "u", "ˈ"), MakeCons("n"),
        MakeVowel("j", "o", ""), MakeCons("ɾ")}},
      {"flash",
       {MakeCons("f"), MakeCons("l"), MakeVowel("", "a", "ˈ"), MakeCons("ʃ")}},
      {"ing", {MakeVowel("", "i", "ˈ"), MakeCons("ŋ")}},
      {"pág", {MakeCons("p"), MakeVowel("", "a", "ˈ"), MakeCons("ɡ")}},
      {"lic", {MakeCons("l"), MakeVowel("", "i", "ˈ"), MakeCons("k")}},
      {"dr",
       {MakeCons("D"), MakeVowel("", "o", ""), MakeCons("k"), MakeCons("t"),
        MakeVowel("", "o", "ˈ"), MakeCons("ɾ")}},
      {"dra",
       {MakeCons("D"), MakeVowel("", "o", ""), MakeCons("k"), MakeCons("t"),
        MakeVowel("", "o", "ˈ"), MakeCons("ɾ"), MakeVowel("", "a", "")}},
      {"sra",
       {MakeCons("s"), MakeVowel("", "e", ""), MakeCons("ɲ"),
        MakeVowel("", "o", "ˈ"), MakeCons("ɾ"), MakeVowel("", "a", "")}},
      {"sr",
       {MakeCons("s"), MakeVowel("", "e", ""), MakeCons("ɲ"),
        MakeVowel("", "o", "ˈ"), MakeCons("ɾ")}},
      {"etc",
       {MakeVowel("", "e", ""), MakeCons("t"), MakeCons("θ"),
        MakeVowel("", "e", "ˈ", true), MakeCons("t"), MakeVowel("", "e", ""),
        MakeCons("ɾ"), MakeVowel("", "a", "")}},
      {"dª", {MakeCons("D"), MakeVowel("", "e", "ˌ"), MakeVowel("", "a", "ˈ")}},
      {"kg",
       {MakeCons("k"), MakeVowel("", "i", "ˌ"), MakeCons("l"),
        MakeVowel("", "o", ""), MakeCons("G"), MakeCons("ɾ"),
        MakeVowel("", "a", "ˈ"), MakeCons("m"), MakeVowel("", "o", "")}},
      {"av", {MakeVowel("", "a", "ˈ"), MakeCons("b")}},
      {"uds", {MakeVowel("", "u", "ˈ"), MakeCons("d"), MakeCons("s")}},
      {"abc", {MakeVowel("", "a", "ˈ"), MakeCons("b"), MakeCons("k")}},
  });
  return *kLex;
}

std::vector<GraphemeUnit> ParseUnits(const std::u32string& w) {
  std::vector<GraphemeUnit> out;
  out.reserve(w.size());
  size_t i = 0;
  const size_t n = w.size();
  while (i < n) {
    const char32_t c = w[i];
    const char32_t nx = (i + 1 < n) ? w[i + 1] : 0;
    const char32_t nx2 = (i + 2 < n) ? w[i + 2] : 0;
    if (IsVowelLetter(c)) {
      out.push_back({UnitKind::kVowel, c, ""});
      ++i;
      continue;
    }
    if (c == U'h') {
      if (!out.empty() && out.back().kind == UnitKind::kVowel &&
          IsVowelLetter(nx)) {
        out.push_back({UnitKind::kSilentH, 0, ""});
      }
      ++i;
      continue;
    }
    if (c == U's' && nx == U'h') {
      out.push_back({UnitKind::kConsonant, 0, "ʃ"});
      i += 2;
      continue;
    }
    if (c == U'c') {
      if (nx == U'h') {
        out.push_back({UnitKind::kConsonant, 0, "ʧ"});
        i += 2;
        continue;
      }
      if (nx == U'e' || nx == U'é' || nx == U'i' || nx == U'í') {
        out.push_back({UnitKind::kConsonant, 0, "θ"});
        ++i;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "k"});
      ++i;
      continue;
    }
    if (c == U'q') {
      out.push_back({UnitKind::kConsonant, 0, "k"});
      i += (nx == U'u') ? 2 : 1;
      continue;
    }
    if (c == U'g') {
      if (nx == U'u' &&
          (nx2 == U'e' || nx2 == U'é' || nx2 == U'i' || nx2 == U'í')) {
        out.push_back({UnitKind::kConsonant, 0, "G"});
        i += 2;
        continue;
      }
      if (nx == U'e' || nx == U'é' || nx == U'i' || nx == U'í') {
        out.push_back({UnitKind::kConsonant, 0, "x"});
        ++i;
        continue;
      }
      out.push_back({UnitKind::kConsonant, 0, "G"});
      ++i;
      continue;
    }
    if (c == U'l' && nx == U'l') {
      out.push_back({UnitKind::kConsonant, 0, "ʎ"});
      i += 2;
      continue;
    }
    if (c == U'r') {
      if (nx == U'r') {
        out.push_back({UnitKind::kConsonant, 0, "r"});
        i += 2;
        continue;
      }
      const absl::string_view prev =
          (!out.empty() && out.back().kind == UnitKind::kConsonant)
              ? absl::string_view(out.back().cons)
              : absl::string_view();
      if (i == 0 || prev == "n" || prev == "l" || prev == "s") {
        out.push_back({UnitKind::kConsonant, 0, "r"});
      } else {
        out.push_back({UnitKind::kConsonant, 0, "ɾ"});
      }
      ++i;
      continue;
    }
    if (c == U'y') {
      if (nx != 0 && IsVowelLetter(nx)) {
        out.push_back({UnitKind::kConsonant, 0, "ʝ"});
      } else if (!out.empty() && out.back().kind == UnitKind::kVowel) {
        out.push_back({UnitKind::kVowel, U'Y', ""});
      } else {
        out.push_back({UnitKind::kVowel, U'i', ""});
      }
      ++i;
      continue;
    }
    if (c == U'ñ') {
      out.push_back({UnitKind::kConsonant, 0, "ɲ"});
      ++i;
      continue;
    }
    if (c == U'j') {
      out.push_back({UnitKind::kConsonant, 0, "x"});
      ++i;
      continue;
    }
    if (c == U'z') {
      out.push_back({UnitKind::kConsonant, 0, "θ"});
      ++i;
      continue;
    }
    if (c == U'x') {
      out.push_back({UnitKind::kConsonant, 0, "k"});
      out.push_back({UnitKind::kConsonant, 0, "s"});
      ++i;
      continue;
    }
    if (c == U'b' || c == U'v') {
      out.push_back({UnitKind::kConsonant, 0, "B"});
      ++i;
      continue;
    }
    if (c == U'd') {
      out.push_back({UnitKind::kConsonant, 0, "D"});
      ++i;
      continue;
    }
    if (c == U'w') {
      out.push_back({UnitKind::kConsonant, 0, "w"});
      ++i;
      continue;
    }
    if (c == U'k') {
      out.push_back({UnitKind::kConsonant, 0, "k"});
      ++i;
      continue;
    }
    std::string single_cons;
    AppendUtf8(c, single_cons);
    out.push_back({UnitKind::kConsonant, 0, std::move(single_cons)});
    ++i;
  }
  return out;
}

std::vector<Segment> Syllabify(const std::vector<GraphemeUnit>& units) {
  std::vector<Segment> segs;
  segs.reserve(units.size());
  size_t i = 0;
  while (i < units.size()) {
    const GraphemeUnit& u = units[i];
    if (u.kind == UnitKind::kSilentH) {
      ++i;
      continue;
    }
    if (u.kind == UnitKind::kConsonant) {
      segs.push_back({false, u.cons, {}});
      ++i;
      continue;
    }
    const absl::string_view c1 = (!segs.empty() && !segs.back().is_nucleus)
                                     ? absl::string_view(segs.back().cons)
                                     : absl::string_view();
    const absl::string_view c2 =
        (segs.size() > 1 && !segs[segs.size() - 2].is_nucleus)
            ? absl::string_view(segs[segs.size() - 2].cons)
            : absl::string_view();
    bool hiatus_ctx =
        (c1 == "r") || ((c1 == "ɾ" || c1 == "l") && !c2.empty() && c2 != "ɾ" &&
                        c2 != "l" && c2 != "r" && c2 != "n" && c2 != "s");

    std::vector<char32_t> run;
    while (i < units.size() && units[i].kind == UnitKind::kVowel) {
      run.push_back(units[i].vowel);
      ++i;
    }
    if (!run.empty() &&
        ((run[0] == U'u' && (c1 == "l" || c1 == "r" || c1 == "ɾ")) ||
         (run[0] == U'ü' && c1 == "G"))) {
      hiatus_ctx = true;
    }

    size_t j = 0;
    while (j < run.size()) {
      const char32_t ch = run[j];
      const char32_t nxt = (j + 1 < run.size()) ? run[j + 1] : 0;
      const bool is_weak = IsWeakVowel(ch);
      if (j == 0 && hiatus_ctx && is_weak && nxt != 0 &&
          (IsStrongVowel(nxt) || nxt == U'í' || nxt == U'ú' ||
           (IsWeakVowel(nxt) && nxt != ch))) {
        std::vector<NucleusChar> nuc = {{ch, VowelRole::kHiatusPre},
                                        {nxt, VowelRole::kNucleus}};
        j += 2;
        if (j < run.size() && IsWeakVowel(run[j]) && IsStrongVowel(nxt) &&
            (j + 1 >= run.size() || !IsStrongVowel(run[j + 1]))) {
          nuc.push_back({run[j], VowelRole::kOffglide});
          ++j;
        }
        segs.push_back({true, "", std::move(nuc)});
        continue;
      }
      if (ch == U'u' && nxt == U'Y') {
        segs.push_back(
            {true,
             "",
             {{U'u', VowelRole::kNucleus}, {U'Y', VowelRole::kOffglide}}});
        j += 2;
        continue;
      }
      if (is_weak && nxt != 0 &&
          (IsStrongVowel(nxt) || (IsWeakVowel(nxt) && nxt != ch))) {
        std::vector<NucleusChar> nuc = {{ch, VowelRole::kGlide},
                                        {nxt, VowelRole::kNucleus}};
        j += 2;
        if (j < run.size() && IsWeakVowel(run[j]) && IsStrongVowel(nxt) &&
            (j + 1 >= run.size() || !IsStrongVowel(run[j + 1]))) {
          nuc.push_back({run[j], VowelRole::kOffglide});
          ++j;
        }
        segs.push_back({true, "", std::move(nuc)});
        continue;
      }
      std::vector<NucleusChar> nuc = {{ch, VowelRole::kNucleus}};
      ++j;
      if (IsStrongVowel(ch) && j < run.size() && IsWeakVowel(run[j]) &&
          (j + 1 >= run.size() || !IsStrongVowel(run[j + 1]))) {
        nuc.push_back({run[j], VowelRole::kOffglide});
        ++j;
      }
      segs.push_back({true, "", std::move(nuc)});
    }
  }
  return segs;
}

struct NucleusIpa {
  std::string pre;
  std::string body;
};

NucleusIpa RenderNucleusIpa(const std::vector<NucleusChar>& nuc) {
  std::string pre;
  std::string body;
  bool accented_strong = false;
  for (const NucleusChar& nc : nuc) {
    if ((nc.ch == U'á' || nc.ch == U'é' || nc.ch == U'ó') &&
        nc.role == VowelRole::kNucleus) {
      accented_strong = true;
    }
    const char32_t base = UnaccentVowel(nc.ch);
    if (nc.role == VowelRole::kGlide) {
      pre.push_back((base == U'i') ? 'j' : 'w');
    } else if (nc.role == VowelRole::kHiatusPre) {
      AppendUtf8(base, pre);
    } else if (nc.role == VowelRole::kNucleus) {
      AppendUtf8(base, body);
    } else {
      AppendUtf8((base == U'Y') ? U'i' : base, body);
    }
  }
  if (body == "ui") {
    return {std::move(pre), "uj"};
  }
  if (accented_strong) {
    return {std::move(pre), std::move(body)};
  }
  if (body == "ai") {
    body = "I";
  } else if (body == "au") {
    body = "W";
  } else if (body == "ei") {
    body = "A";
  } else if (body == "oi") {
    body = "oɪ";
  } else if (body == "eu") {
    body = "eʊ";
  } else if (body == "ou") {
    body = "oʊ";
  }
  return {std::move(pre), std::move(body)};
}

std::vector<PhoneToken> WordPhones(const std::u32string& raw_word);

absl::string_view LetterName(char32_t ch) {
  switch (ch) {
    case U'a':
      return "a";
    case U'b':
      return "be";
    case U'c':
      return "ce";
    case U'd':
      return "de";
    case U'e':
      return "e";
    case U'f':
      return "efe";
    case U'g':
      return "ge";
    case U'h':
      return "hache";
    case U'i':
      return "i";
    case U'j':
      return "jota";
    case U'k':
      return "ka";
    case U'l':
      return "ele";
    case U'm':
      return "eme";
    case U'n':
      return "ene";
    case U'ñ':
      return "eñe";
    case U'o':
      return "o";
    case U'p':
      return "pe";
    case U'q':
      return "cu";
    case U'r':
      return "erre";
    case U's':
      return "ese";
    case U't':
      return "te";
    case U'u':
      return "u";
    case U'v':
      return "uve";
    case U'w':
      return "uve";
    case U'x':
      return "equis";
    case U'y':
      return "ye";
    case U'z':
      return "zeta";
    default:
      return "";
  }
}

std::vector<PhoneToken> SpellWord(const std::u32string& word) {
  if (word.size() == 1 && word[0] == U'w') {
    std::vector<PhoneToken> out = WordPhones(Utf8ToComposedU32("uve"));
    for (PhoneToken& t : out) {
      if (t.kind == PhoneKind::kVowel && t.stress == "ˈ") {
        t.stress = "ˌ";
      }
    }
    const std::vector<PhoneToken> doble = WordPhones(Utf8ToComposedU32("doble"));
    out.insert(out.end(), doble.begin(), doble.end());
    return out;
  }
  std::vector<std::vector<PhoneToken>> parts;
  parts.reserve(word.size());
  for (char32_t ch : word) {
    const absl::string_view name = LetterName(ch);
    if (name.empty()) {
      continue;
    }
    parts.push_back(WordPhones(Utf8ToComposedU32(name)));
  }
  std::vector<PhoneToken> out;
  for (size_t k = 0; k < parts.size(); ++k) {
    std::vector<PhoneToken>& toks = parts[k];
    for (size_t idx = 0; idx < toks.size(); ++idx) {
      PhoneToken& t = toks[idx];
      if (t.kind == PhoneKind::kVowel && t.stress == "ˈ" &&
          k + 1 != parts.size()) {
        t.stress = "ˌ";
      }
      if (t.kind == PhoneKind::kVowel && t.stress == "ˈ" && t.body == "e" &&
          idx + 1 != toks.size()) {
        t.body = "ɛ";
        t.no_open_e = true;
      }
    }
    out.insert(out.end(), toks.begin(), toks.end());
  }
  return out;
}

std::vector<PhoneToken> WordPhones(const std::u32string& raw_word) {
  const std::u32string word = LowercaseU32(raw_word);
  const std::string word_utf8 = CodePointsToUtf8(word);
  const auto& lexicon = GetSpanishLexicon();
  if (const auto it = lexicon.find(word_utf8); it != lexicon.end()) {
    return it->second;
  }

  bool any_vowel_or_y = false;
  for (char32_t c : word) {
    if (IsVowelLetter(c) || c == U'y') {
      any_vowel_or_y = true;
      break;
    }
  }
  if (!any_vowel_or_y && word.size() <= 5) {
    return SpellWord(word);
  }

  // Adverbs in `-mente` carry two primary stresses (on the adjective base and
  // on `-mente`).
  if (word.size() > 6 && absl::EndsWith(word_utf8, "mente")) {
    const std::u32string stem = word.substr(0, word.size() - 5);
    bool stem_has_vowel = false;
    for (char32_t c : stem) {
      if (IsVowelLetter(c)) {
        stem_has_vowel = true;
        break;
      }
    }
    if (stem_has_vowel) {
      std::vector<PhoneToken> base = WordPhones(stem);
      std::vector<PhoneToken> suffix = WordPhones(Utf8ToComposedU32("mente"));
      for (PhoneToken& t : suffix) {
        if (t.kind == PhoneKind::kVowel) {
          t.no_open_e = true;
        }
      }
      base.insert(base.end(), suffix.begin(), suffix.end());
      return base;
    }
  }

  const std::vector<GraphemeUnit> units = ParseUnits(word);
  const std::vector<Segment> segs = Syllabify(units);

  std::vector<size_t> nuc_idx;
  for (size_t i = 0; i < segs.size(); ++i) {
    if (segs[i].is_nucleus) {
      nuc_idx.push_back(i);
    }
  }
  if (nuc_idx.empty()) {
    std::vector<PhoneToken> toks;
    toks.reserve(segs.size());
    for (const Segment& s : segs) {
      toks.push_back(MakeCons(s.cons));
    }
    return toks;
  }

  size_t st = 0;
  bool found_accent = false;
  for (size_t k = 0; k < nuc_idx.size(); ++k) {
    for (const NucleusChar& nc : segs[nuc_idx[k]].nucleus) {
      if (IsAccentedVowel(nc.ch)) {
        st = k;
        found_accent = true;
        break;
      }
    }
    if (found_accent) {
      break;
    }
  }
  if (!found_accent) {
    if (nuc_idx.size() == 1) {
      st = 0;
    } else {
      const char32_t last = word.back();
      if (IsVowelLetter(last) || last == U'n' || last == U's') {
        st = nuc_idx.size() - 2;
      } else {
        st = nuc_idx.size() - 1;
      }
    }
  }

  std::vector<int> syl_pos;
  syl_pos.reserve(nuc_idx.size());
  int pos = 0;
  for (size_t idx : nuc_idx) {
    bool has_hiatus_pre = false;
    for (const NucleusChar& nc : segs[idx].nucleus) {
      if (nc.role == VowelRole::kHiatusPre) {
        has_hiatus_pre = true;
        break;
      }
    }
    if (has_hiatus_pre) {
      ++pos;
    }
    syl_pos.push_back(pos);
    ++pos;
  }

  std::vector<PhoneToken> toks;
  toks.reserve(segs.size());
  size_t nuc_k = 0;
  for (size_t i = 0; i < segs.size(); ++i) {
    if (!segs[i].is_nucleus) {
      toks.push_back(MakeCons(segs[i].cons));
      continue;
    }
    const size_t k = nuc_k++;
    const NucleusIpa n_ipa = RenderNucleusIpa(segs[i].nucleus);
    absl::string_view stress = "";
    if (k == st) {
      stress = "ˈ";
    } else if ((syl_pos[st] >= 2 && syl_pos[k] < syl_pos[st] - 1 &&
                syl_pos[k] % 2 == 0) ||
               (syl_pos[st] >= 3 && k == 0 && syl_pos[k] == 1)) {
      stress = "ˌ";
    }
    bool has_acute_e = false;
    for (const NucleusChar& nc : segs[i].nucleus) {
      if (nc.ch == U'é') {
        has_acute_e = true;
        break;
      }
    }
    toks.push_back(MakeVowel(n_ipa.pre, n_ipa.body, stress, has_acute_e));
  }
  return toks;
}

std::vector<PhoneToken> NumberUnder100Tokens(int n) {
  switch (n) {
    case 0:
      return {MakeCons("θ"), MakeVowel("", "e", "ˈ", true), MakeCons("ɾ"),
              MakeVowel("", "o", "")};
    case 1:
      return {MakeVowel("", "u", "ˈ"), MakeCons("n"), MakeVowel("", "o", "")};
    case 2:
      return {MakeCons("D"), MakeVowel("", "o", "ˈ"), MakeCons("s")};
    case 3:
      return {MakeCons("t"), MakeCons("ɾ"), MakeVowel("", "e", "ˈ", true),
              MakeCons("s")};
    case 4:
      return {MakeCons("k"), MakeVowel("w", "a", "ˈ"), MakeCons("t"),
              MakeCons("ɾ"), MakeVowel("", "o", "")};
    case 5:
      return {MakeCons("θ"), MakeVowel("", "i", "ˈ"), MakeCons("n"),
              MakeCons("k"), MakeVowel("", "o", "")};
    case 6:
      return {MakeCons("s"), MakeVowel("", "A", "ˈ"), MakeCons("s")};
    case 7:
      return {MakeCons("s"), MakeVowel("j", "e", "ˈ", true), MakeCons("t"),
              MakeVowel("", "e", "")};
    case 8:
      return {MakeVowel("", "o", "ˈ"), MakeCons("ʧ"), MakeVowel("", "o", "")};
    case 9:
      return {MakeCons("n"), MakeVowel("w", "e", "ˈ", true), MakeCons("B"),
              MakeVowel("", "e", "")};
    case 10:
      return {MakeCons("D"), MakeVowel("j", "ɛ", "ˈ", true), MakeCons("θ")};
    case 11:
      return {MakeVowel("", "o", "ˈ"), MakeCons("n"), MakeCons("θ"),
              MakeVowel("", "e", "")};
    case 12:
      return {MakeCons("D"), MakeVowel("", "o", "ˈ"), MakeCons("θ"),
              MakeVowel("", "e", "")};
    case 13:
      return {MakeCons("t"), MakeCons("ɾ"), MakeVowel("", "e", "ˈ", true),
              MakeCons("θ"), MakeVowel("", "e", "")};
    case 14:
      return {MakeCons("k"),           MakeVowel("", "a", ""), MakeCons("t"),
              MakeVowel("", "o", "ˈ"), MakeCons("ɾ"),          MakeCons("θ"),
              MakeVowel("", "e", "")};
    case 15:
      return {MakeCons("k"), MakeVowel("", "i", "ˈ"), MakeCons("n"),
              MakeCons("θ"), MakeVowel("", "e", "")};
    case 16:
      return {MakeCons("D"), MakeVowel("j", "e", "ˌ"),
              MakeCons("θ"), MakeVowel("", "i", ""),
              MakeCons("s"), MakeVowel("", "A", "ˈ"),
              MakeCons("s")};
    case 17:
      return {MakeCons("D"), MakeVowel("j", "e", "ˌ"),
              MakeCons("θ"), MakeVowel("", "i", ""),
              MakeCons("s"), MakeVowel("j", "e", "ˈ", true),
              MakeCons("t"), MakeVowel("", "e", "")};
    case 18:
      return {MakeCons("D"),           MakeVowel("j", "ɛ", "ˌ", true),
              MakeCons("θ"),           MakeVowel("", "i", ""),
              MakeVowel("", "o", "ˈ"), MakeCons("ʧ"),
              MakeVowel("", "o", "")};
    case 19:
      return {MakeCons("D"), MakeVowel("j", "e", "ˌ"),
              MakeCons("θ"), MakeVowel("", "i", ""),
              MakeCons("n"), MakeVowel("w", "e", "ˈ", true),
              MakeCons("B"), MakeVowel("", "e", "")};
    case 20:
      return {MakeCons("B"), MakeVowel("", "A", "ˈ"), MakeCons("n"),
              MakeCons("t"), MakeVowel("", "e", "")};
    case 26:
      return {MakeCons("B"),           MakeVowel("", "A", ""), MakeCons("n"),
              MakeCons("t"),           MakeVowel("", "i", ""), MakeCons("s"),
              MakeVowel("", "A", "ˈ"), MakeCons("s")};
    default:
      break;
  }
  if (n >= 21 && n <= 29) {
    std::vector<PhoneToken> out = {MakeCons("B"), MakeVowel("", "A", "ˌ"),
                                   MakeCons("n"), MakeCons("t"),
                                   MakeVowel("", "i", "")};
    const std::vector<PhoneToken> unit = NumberUnder100Tokens(n - 20);
    out.insert(out.end(), unit.begin(), unit.end());
    return out;
  }
  const int tens = (n / 10) * 10;
  const int unit = n % 10;
  std::vector<PhoneToken> base_tens;
  switch (tens) {
    case 30:
      base_tens = {MakeCons("t"), MakeCons("ɾ"), MakeVowel("", "A", "ˈ"),
                   MakeCons("n"), MakeCons("t"), MakeVowel("", "a", "")};
      break;
    case 40:
      base_tens = {MakeCons("k"),         MakeVowel("w", "a", ""),
                   MakeCons("ɾ"),         MakeVowel("", "ɛ", "ˈ", true),
                   MakeCons("n"),         MakeCons("t"),
                   MakeVowel("", "a", "")};
      break;
    case 50:
      base_tens = {MakeCons("θ"), MakeVowel("", "i", ""),         MakeCons("n"),
                   MakeCons("k"), MakeVowel("w", "ɛ", "ˈ", true), MakeCons("n"),
                   MakeCons("t"), MakeVowel("", "a", "")};
      break;
    case 60:
      base_tens = {MakeCons("s"),         MakeVowel("", "ɛ", "", true),
                   MakeCons("s"),         MakeVowel("", "ɛ", "ˈ", true),
                   MakeCons("n"),         MakeCons("t"),
                   MakeVowel("", "a", "")};
      break;
    case 70:
      base_tens = {MakeCons("s"),         MakeVowel("", "e", ""),
                   MakeCons("t"),         MakeVowel("", "ɛ", "ˈ", true),
                   MakeCons("n"),         MakeCons("t"),
                   MakeVowel("", "a", "")};
      break;
    case 80:
      base_tens = {
          MakeVowel("", "o", ""), MakeCons("ʧ"), MakeVowel("", "ɛ", "ˈ", true),
          MakeCons("n"),          MakeCons("t"), MakeVowel("", "a", "")};
      break;
    case 90:
    default:
      base_tens = {MakeCons("n"),         MakeVowel("", "o", ""),
                   MakeCons("B"),         MakeVowel("", "ɛ", "ˈ", true),
                   MakeCons("n"),         MakeCons("t"),
                   MakeVowel("", "a", "")};
      break;
  }
  if (unit == 0) {
    return base_tens;
  }
  for (PhoneToken& t : base_tens) {
    if (t.kind == PhoneKind::kVowel && t.stress == "ˈ") {
      t.stress = "ˌ";
    }
  }
  base_tens.back().body = "ai";
  const std::vector<PhoneToken> unit_toks = NumberUnder100Tokens(unit);
  base_tens.insert(base_tens.end(), unit_toks.begin(), unit_toks.end());
  return base_tens;
}

std::vector<PhoneToken> HundredsTokens(int h) {
  switch (h) {
    case 100:
      return {MakeCons("θ"), MakeVowel("j", "e", "ˈ", true), MakeCons("n")};
    case 101:
      return {MakeCons("θ"), MakeVowel("j", "e", "ˈ", true), MakeCons("n"),
              MakeCons("t"), MakeVowel("", "o", "")};
    case 200:
      return {MakeCons("D"), MakeVowel("", "o", "ˌ"),        MakeCons("s"),
              MakeCons("θ"), MakeVowel("j", "ɛ", "ˈ", true), MakeCons("n"),
              MakeCons("t"), MakeVowel("", "o", ""),         MakeCons("s")};
    case 300:
      return {MakeCons("t"), MakeCons("ɾ"), MakeVowel("", "e", "ˌ"),
              MakeCons("s"), MakeCons("θ"), MakeVowel("j", "ɛ", "ˈ", true),
              MakeCons("n"), MakeCons("t"), MakeVowel("", "o", ""),
              MakeCons("s")};
    case 400:
      return {MakeCons("k"),
              MakeVowel("w", "a", "ˌ"),
              MakeCons("t"),
              MakeCons("ɾ"),
              MakeVowel("", "o", ""),
              MakeCons("θ"),
              MakeVowel("j", "ɛ", "ˈ", true),
              MakeCons("n"),
              MakeCons("t"),
              MakeVowel("", "o", ""),
              MakeCons("s")};
    case 500:
      return {MakeCons("k"),          MakeVowel("", "i", ""),
              MakeCons("n"),          MakeVowel("j", "ɛ", "ˈ", true),
              MakeCons("n"),          MakeCons("t"),
              MakeVowel("", "o", ""), MakeCons("s")};
    case 600:
      return {MakeCons("s"), MakeVowel("", "A", "ˌ"),        MakeCons("s"),
              MakeCons("θ"), MakeVowel("j", "ɛ", "ˈ", true), MakeCons("n"),
              MakeCons("t"), MakeVowel("", "o", ""),         MakeCons("s")};
    case 700:
      return {MakeCons("s"),          MakeVowel("", "e", "ˌ"),
              MakeCons("t"),          MakeVowel("", "e", ""),
              MakeCons("θ"),          MakeVowel("j", "ɛ", "ˈ", true),
              MakeCons("n"),          MakeCons("t"),
              MakeVowel("", "o", ""), MakeCons("s")};
    case 800:
      return {MakeVowel("", "o", "ˌ"),
              MakeCons("ʧ"),
              MakeVowel("", "o", ""),
              MakeCons("θ"),
              MakeVowel("j", "ɛ", "ˈ", true),
              MakeCons("n"),
              MakeCons("t"),
              MakeVowel("", "o", ""),
              MakeCons("s")};
    case 900:
    default:
      return {MakeCons("n"),          MakeVowel("", "o", "ˌ"),
              MakeCons("B"),          MakeVowel("", "e", ""),
              MakeCons("θ"),          MakeVowel("j", "ɛ", "ˈ", true),
              MakeCons("n"),          MakeCons("t"),
              MakeVowel("", "o", ""), MakeCons("s")};
  }
}

std::vector<std::vector<PhoneToken>> NumberWordsTokens(absl::string_view s) {
  if (s == "00") {
    const std::vector<PhoneToken> z = NumberUnder100Tokens(0);
    return {z, z};
  }
  if (s == "000") {
    const std::vector<PhoneToken> z = NumberUnder100Tokens(0);
    std::vector<PhoneToken> zz = z;
    zz.insert(zz.end(), z.begin(), z.end());
    return {std::move(zz), z};
  }
  if (s.size() > 1 && s.front() == '0') {
    std::vector<std::vector<PhoneToken>> out;
    out.reserve(s.size());
    for (char c : s) {
      out.push_back(NumberUnder100Tokens(c - '0'));
    }
    return out;
  }
  int64_t n = 0;
  if (!absl::SimpleAtoi(s, &n) || n >= 1000000000LL) {
    std::vector<std::vector<PhoneToken>> out;
    out.reserve(s.size());
    for (char c : s) {
      if (absl::ascii_isdigit(static_cast<unsigned char>(c))) {
        out.push_back(NumberUnder100Tokens(c - '0'));
      }
    }
    return out;
  }
  if (n < 100) {
    return {NumberUnder100Tokens(static_cast<int>(n))};
  }
  if (n < 1000) {
    const int h = static_cast<int>((n / 100) * 100);
    const int rem = static_cast<int>(n % 100);
    if (rem == 0) {
      return {HundredsTokens(h)};
    }
    std::vector<std::vector<PhoneToken>> out = {
        HundredsTokens(h == 100 ? 101 : h)};
    std::vector<std::vector<PhoneToken>> rem_toks =
        NumberWordsTokens(absl::StrCat(rem));
    out.insert(out.end(), rem_toks.begin(), rem_toks.end());
    return out;
  }
  if (n < 1000000) {
    const int64_t th = n / 1000;
    const int64_t rem = n % 1000;
    std::vector<std::vector<PhoneToken>> out;
    if (th > 1) {
      out = NumberWordsTokens(absl::StrCat(th));
    }
    out.push_back({MakeCons("m"), MakeVowel("", "i", "ˈ"), MakeCons("l")});
    if (rem > 0) {
      std::vector<std::vector<PhoneToken>> rem_toks =
          NumberWordsTokens(absl::StrCat(rem));
      out.insert(out.end(), rem_toks.begin(), rem_toks.end());
    }
    return out;
  }
  const int64_t ml = n / 1000000;
  const int64_t rem = n % 1000000;
  std::vector<std::vector<PhoneToken>> out;
  if (ml == 1) {
    out.push_back({MakeVowel("", "u", "ˈ"), MakeCons("n")});
    out.push_back(WordPhones(Utf8ToComposedU32("millón")));
  } else {
    out = NumberWordsTokens(absl::StrCat(ml));
    out.push_back(WordPhones(Utf8ToComposedU32("millones")));
  }
  if (rem > 0) {
    std::vector<std::vector<PhoneToken>> rem_toks =
        NumberWordsTokens(absl::StrCat(rem));
    out.insert(out.end(), rem_toks.begin(), rem_toks.end());
  }
  return out;
}

enum class ContextClass { kPause, kVowel, kNasal, kLateral, kConsonant };

ContextClass ClassifyConsonant(absl::string_view ph) {
  if (ph == "n" || ph == "m" || ph == "ŋ") {
    return ContextClass::kNasal;
  }
  if (ph == "l") {
    return ContextClass::kLateral;
  }
  return ContextClass::kConsonant;
}

ContextClass EndClass(const std::vector<PhoneToken>& toks) {
  if (toks.empty()) {
    return ContextClass::kPause;
  }
  const PhoneToken& last = toks.back();
  if (last.kind == PhoneKind::kVowel) {
    return ContextClass::kVowel;
  }
  return ClassifyConsonant(last.phone);
}

std::vector<PhoneToken> RealizeWord(const std::vector<PhoneToken>& toks,
                                    ContextClass prev_end, bool next_is_vowel,
                                    bool pause_after) {
  std::vector<PhoneToken> out;
  out.reserve(toks.size());
  for (size_t i = 0; i < toks.size(); ++i) {
    const PhoneToken& t = toks[i];
    if (t.kind != PhoneKind::kConsonant) {
      out.push_back(t);
      continue;
    }
    std::string ph = t.phone;
    ContextClass left = prev_end;
    if (i > 0) {
      const PhoneToken& prev = toks[i - 1];
      left = (prev.kind == PhoneKind::kVowel) ? ContextClass::kVowel
                                              : ClassifyConsonant(prev.phone);
    }
    // Textbook Spanish also keeps /d/ a stop after /l/ ("falda"), but Host
    // Kokoro spirantizes it ("fˈalða", "el ðˈeðo") and the Kokoro acoustic
    // model was trained on that output, so only a pause or a nasal blocks
    // spirantization here.
    const bool after_pause_or_nasal =
        (left == ContextClass::kPause || left == ContextClass::kNasal);
    if (ph == "B") {
      ph = after_pause_or_nasal ? "b" : "β";
    } else if (ph == "D") {
      ph = after_pause_or_nasal ? "d" : "ð";
      if (i + 1 == toks.size() && (pause_after || !next_is_vowel)) {
        ph = "d";
      }
    } else if (ph == "G") {
      ph = after_pause_or_nasal ? "ɡ" : "ɣ";
    }
    if ((t.phone == "B" || t.phone == "D" || t.phone == "G") &&
        i + 1 < toks.size() && toks[i + 1].kind == PhoneKind::kConsonant) {
      const absl::string_view nx = toks[i + 1].phone;
      if (nx != "l" && nx != "ɾ" && nx != "r" && nx != "s" && nx != "x") {
        ph = (t.phone == "B") ? "b" : ((t.phone == "D") ? "d" : "ɡ");
      }
    }
    if (ph == "p" && i + 1 < toks.size() &&
        toks[i + 1].kind == PhoneKind::kConsonant && toks[i + 1].phone == "t") {
      ph = "pː";
    }
    out.push_back(MakeCons(ph));
  }

  // Nasal place assimilation inside the word.
  if (out.size() >= 2) {
    for (size_t i = 0; i + 1 < out.size(); ++i) {
      if (out[i].kind == PhoneKind::kConsonant && out[i].phone == "n" &&
          out[i + 1].kind == PhoneKind::kConsonant) {
        const absl::string_view nx = out[i + 1].phone;
        const absl::string_view nx2 =
            (i + 2 < out.size() && out[i + 2].kind == PhoneKind::kConsonant)
                ? absl::string_view(out[i + 2].phone)
                : absl::string_view();
        if (nx == "b" || nx == "β" || nx == "p" ||
            (nx == "f" && (nx2 == "l" || nx2 == "ɾ"))) {
          out[i].phone = "m";
          if (nx == "β") {
            out[i + 1].phone = "b";
          }
        } else if (nx == "ɡ" || nx == "x" || nx == "ɣ") {
          out[i].phone = "ŋ";
          if (nx == "ɣ") {
            out[i + 1].phone = "ɡ";
          }
        }
      }
    }
  }

  // Unaccented stressed /e/ before nasal + consonant opens to /ɛ/.
  if (out.size() >= 3) {
    for (size_t i = 0; i + 2 < out.size(); ++i) {
      if (out[i].kind == PhoneKind::kVowel && !out[i].no_open_e &&
          out[i].stress == "ˈ" && out[i].body == "e" &&
          out[i + 1].kind == PhoneKind::kConsonant &&
          (out[i + 1].phone == "n" || out[i + 1].phone == "ŋ") &&
          out[i + 2].kind == PhoneKind::kConsonant) {
        out[i].body = "ɛ";
      }
    }
  }
  return out;
}

std::string RenderTokens(const std::vector<PhoneToken>& toks) {
  size_t total_len = 0;
  for (const PhoneToken& t : toks) {
    if (t.kind != PhoneKind::kVowel) {
      total_len += t.phone.size();
    } else {
      total_len += t.glides.size() + t.stress.size() + t.body.size();
    }
  }
  std::string out;
  out.reserve(total_len);
  for (const PhoneToken& t : toks) {
    if (t.kind != PhoneKind::kVowel) {
      out.append(t.phone);
    } else {
      out.append(t.glides);
      out.append(t.stress);
      out.append(t.body);
    }
  }
  return out;
}

enum class RawTokenKind { kSpace, kDigits, kWord, kPunct };

struct RawToken {
  RawTokenKind kind;
  std::u32string u32;
  std::string utf8;
};

std::vector<RawToken> TokenizeSpanish(absl::string_view text) {
  const std::u32string u32 = Utf8ToComposedU32(text);
  std::vector<RawToken> out;
  size_t i = 0;
  while (i < u32.size()) {
    const char32_t cp = u32[i];
    if (IsWhitespaceCp(cp)) {
      std::u32string span;
      while (i < u32.size() && IsWhitespaceCp(u32[i])) {
        span.push_back(u32[i]);
        ++i;
      }
      out.push_back({RawTokenKind::kSpace, span, CodePointsToUtf8(span)});
      continue;
    }
    if (IsAsciiDigitCp(cp)) {
      std::u32string span;
      while (i < u32.size() && IsAsciiDigitCp(u32[i])) {
        span.push_back(u32[i]);
        ++i;
      }
      out.push_back({RawTokenKind::kDigits, span, CodePointsToUtf8(span)});
      continue;
    }
    if (IsSpanishLetter(cp)) {
      std::u32string span;
      while (i < u32.size() && IsSpanishLetter(u32[i])) {
        span.push_back(u32[i]);
        ++i;
      }
      out.push_back({RawTokenKind::kWord, span, CodePointsToUtf8(span)});
      continue;
    }
    const std::u32string single(1, cp);
    out.push_back({RawTokenKind::kPunct, single, CodePointsToUtf8(single)});
    ++i;
  }
  return out;
}

struct SentenceItem {
  bool is_word = false;
  std::vector<PhoneToken> toks;
  std::string punct;
  bool tight_before = false;
};

std::string PhonemizeSpanish(
    absl::string_view text, bool isolated,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) {
  const std::vector<RawToken> raw_tokens = TokenizeSpanish(text);
  std::vector<SentenceItem> items;
  items.reserve(raw_tokens.size());

  size_t i = 0;
  const size_t n = raw_tokens.size();
  while (i < n) {
    const RawToken& tok = raw_tokens[i];
    if (tok.kind == RawTokenKind::kSpace) {
      ++i;
      continue;
    }
    const RawToken prev_tok =
        (i > 0) ? raw_tokens[i - 1] : RawToken{RawTokenKind::kSpace, U" ", " "};
    const RawToken next_tok = (i + 1 < n)
                                  ? raw_tokens[i + 1]
                                  : RawToken{RawTokenKind::kSpace, U" ", " "};
    const bool tight_before = (prev_tok.kind != RawTokenKind::kSpace) &&
                              !items.empty() && !items.back().is_word &&
                              items.back().punct != "¡" &&
                              items.back().punct != "¿";

    if (tok.kind == RawTokenKind::kDigits) {
      const std::vector<std::vector<PhoneToken>> num_words =
          NumberWordsTokens(tok.utf8);
      for (size_t k = 0; k < num_words.size(); ++k) {
        items.push_back({true, num_words[k], "", tight_before && (k == 0)});
      }
      ++i;
      continue;
    }

    if (tok.kind == RawTokenKind::kPunct) {
      if (tok.utf8 == "+" && next_tok.kind == RawTokenKind::kDigits) {
        items.push_back(
            {true,
             {MakeCons("m"), MakeVowel("", "a", "ˈ"), MakeCons("s")},
             "",
             tight_before});
        ++i;
        continue;
      }
      if (tok.utf8 == "%" && prev_tok.kind == RawTokenKind::kDigits) {
        items.push_back({true,
                         {MakeCons("p"), MakeVowel("", "o", ""), MakeCons("ɾ"),
                          MakeCons("θ"), MakeVowel("j", "ɛ", "ˈ", true),
                          MakeCons("n"), MakeCons("t"), MakeVowel("", "o", "")},
                         "",
                         false});
        ++i;
        continue;
      }
      if (tok.utf8 == "€" && next_tok.kind == RawTokenKind::kDigits) {
        items.push_back(
            {true,
             {MakeVowel("", "eʊ", "ˈ"), MakeCons("ɾ"), MakeVowel("", "o", "")},
             "",
             tight_before});
        ++i;
        continue;
      }
      if (tok.utf8 == "$" && next_tok.kind == RawTokenKind::kDigits) {
        items.push_back({true,
                         {MakeCons("D"), MakeVowel("", "o", "ˈ"), MakeCons("l"),
                          MakeVowel("", "a", ""), MakeCons("ɾ")},
                         "",
                         tight_before});
        ++i;
        continue;
      }
      if (tok.utf8 == "/" && prev_tok.kind == RawTokenKind::kDigits &&
          next_tok.kind == RawTokenKind::kDigits) {
        items.push_back({true,
                         {MakeCons("B"), MakeVowel("", "a", "ˈ"), MakeCons("r"),
                          MakeVowel("", "a", "")},
                         "",
                         false});
        ++i;
        continue;
      }
      if (tok.utf8 == "°") {
        items.push_back({true,
                         {MakeCons("G"), MakeCons("ɾ"), MakeVowel("", "a", "ˈ"),
                          MakeCons("D"), MakeVowel("", "o", ""), MakeCons("s")},
                         "",
                         false});
        ++i;
        continue;
      }
      if (tok.utf8 == "-" && prev_tok.kind == RawTokenKind::kSpace &&
          next_tok.kind == RawTokenKind::kDigits) {
        ++i;
        continue;
      }
      items.push_back({false, {}, tok.utf8, false});
      ++i;
      continue;
    }

    // Word token.
    const std::u32string lw_u32 = LowercaseU32(tok.u32);
    const std::string lw = CodePointsToUtf8(lw_u32);

    if (!custom_lexicon.empty()) {
      if (const auto it = custom_lexicon.find(lw); it != custom_lexicon.end()) {
        items.push_back({true, {MakeLiteral(it->second)}, "", tight_before});
        ++i;
        continue;
      }
    }

    if (lw == "º") {
      items.push_back({true, {MakeVowel("", "o", "ˈ")}, "", tight_before});
      ++i;
      continue;
    }

    if (lw == "ee" && i + 2 < n && raw_tokens[i + 1].utf8 == "." &&
        LowercaseUtf8(raw_tokens[i + 2].utf8) == "uu") {
      size_t word_count = 0;
      for (const SentenceItem& p_it : items) {
        if (p_it.is_word) {
          ++word_count;
        }
      }
      std::vector<PhoneToken> ee_toks =
          (word_count >= 2)
              ? std::vector<PhoneToken>{MakeVowel("", "e", "ˌ"),
                                        MakeVowel("", "e", "ˈ", true)}
              : std::vector<PhoneToken>{MakeVowel("", "e", "ˈ", true),
                                        MakeVowel("", "e", "")};
      items.push_back({true, std::move(ee_toks), "", tight_before});
      items.push_back({false, {}, ".", false});
      items.push_back(
          {true, {MakeVowel("", "u", "ˈ"), MakeVowel("", "u", "")}, "", true});
      i += 3;
      continue;
    }

    if ((isolated || (IsAllUppercaseWord(tok.u32) && !tight_before &&
                      (lw_u32.size() >= 2 || next_tok.utf8 != "."))) &&
        LookupRomanNumeral(lw).has_value()) {
      const int roman_val = *LookupRomanNumeral(lw);
      const std::vector<std::vector<PhoneToken>> num_words =
          NumberWordsTokens(absl::StrCat(roman_val));
      for (size_t k = 0; k < num_words.size(); ++k) {
        items.push_back({true, num_words[k], "", tight_before && (k == 0)});
      }
      ++i;
      continue;
    }

    std::vector<PhoneToken> toks;
    const auto& lexicon = GetSpanishLexicon();
    if (lw == "y") {
      toks = {MakeVowel("", "i", "")};
    } else if (lw == "a" && (tight_before || next_tok.utf8 == ".")) {
      toks = {MakeVowel("", "a", "ˈ")};
    } else if (!lexicon.contains(lw) && lw_u32.size() > 4 &&
               absl::EndsWith(lw, "ing")) {
      std::vector<PhoneToken> stem_toks =
          WordPhones(lw_u32.substr(0, lw_u32.size() - 3));
      for (PhoneToken& t : stem_toks) {
        if (t.kind == PhoneKind::kVowel) {
          t.stress.clear();
        }
      }
      for (size_t idx = stem_toks.size(); idx > 0; --idx) {
        if (stem_toks[idx - 1].kind == PhoneKind::kVowel) {
          stem_toks[idx - 1].stress = "ˈ";
          break;
        }
      }
      stem_toks.push_back(MakeVowel("", "i", ""));
      stem_toks.push_back(MakeCons("ŋ"));
      toks = std::move(stem_toks);
    } else {
      toks = WordPhones(lw_u32);
      if (!isolated && !(tight_before || next_tok.utf8 == ".")) {
        if (IsAtonaMono(lw)) {
          for (PhoneToken& t : toks) {
            if (t.kind == PhoneKind::kVowel) {
              t.stress.clear();
            }
          }
        } else if (IsAtonaPoly(lw)) {
          for (PhoneToken& t : toks) {
            if (t.kind == PhoneKind::kVowel) {
              if (t.stress == "ˈ") {
                t.stress = "ˌ";
              } else {
                t.stress.clear();
              }
            }
          }
        }
      }
    }
    items.push_back({true, std::move(toks), "", tight_before});
    ++i;
  }

  std::string out;
  out.reserve(text.size() * 2);
  ContextClass prev_end = ContextClass::kPause;
  for (size_t idx = 0; idx < items.size(); ++idx) {
    const SentenceItem& it = items[idx];
    if (!it.is_word) {
      if (it.punct != "¡" && it.punct != "¿") {
        if (!out.empty() && out.back() == ' ') {
          out.pop_back();
        }
        out.append(it.punct);
        out.push_back(' ');
      }
      prev_end = ContextClass::kPause;
      continue;
    }
    const SentenceItem* nxt =
        (idx + 1 < items.size()) ? &items[idx + 1] : nullptr;
    const bool pause_after = (nxt == nullptr || !nxt->is_word);
    const bool next_is_vowel =
        (nxt != nullptr && nxt->is_word && !nxt->toks.empty() &&
         nxt->toks.front().kind == PhoneKind::kVowel);

    std::vector<PhoneToken> realized =
        RealizeWord(it.toks, prev_end, next_is_vowel, pause_after);

    if (nxt != nullptr && nxt->is_word && !nxt->toks.empty() &&
        !realized.empty() && realized.back().kind == PhoneKind::kConsonant &&
        realized.back().phone == "n" &&
        nxt->toks.front().kind == PhoneKind::kConsonant) {
      const absl::string_view first_ph = nxt->toks.front().phone;
      if (first_ph == "B" || first_ph == "p") {
        realized.back().phone = "m";
      } else if (first_ph == "G") {
        realized.back().phone = "ŋ";
      }
    }
    prev_end = EndClass(realized);
    if (it.tight_before && !out.empty() && out.back() == ' ') {
      out.pop_back();
    }
    out.append(RenderTokens(realized));
    out.push_back(' ');
  }
  while (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

}  // namespace

SpanishG2p::SpanishG2p(int /*unused*/) {}

absl::StatusOr<SpanishG2p> SpanishG2p::Create() { return SpanishG2p(0); }

absl::StatusOr<std::string> SpanishG2p::DoTextToIpa(
    absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  return PhonemizeSpanish(text, /*isolated=*/false, custom_lexicon);
}

absl::StatusOr<std::string> SpanishG2p::DoWordToIpa(
    absl::string_view word,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  return PhonemizeSpanish(word, /*isolated=*/true, custom_lexicon);
}

}  // namespace litert::omni::tts::kokoro
