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

#include "omni/text2speech/kokoro/hindi_g2p.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/unicode_utils.h"

namespace litert::omni::tts::kokoro {
namespace {

constexpr char32_t kHalant = 0x094D;
constexpr char32_t kNukta = 0x093C;
constexpr char32_t kAnusvara = 0x0902;
constexpr char32_t kChandrabindu = 0x0901;
constexpr char32_t kVisarga = 0x0903;

// Normalizes precomposed Devanagari nukta codepoints (U+0958..U+095F) to their
// canonical base consonant + combining nukta (U+093C) sequence so that lookup
// keys and parser units see a uniform representation.
std::u32string NormalizeDevanagariU32(const std::u32string& u32) {
  std::u32string out;
  out.reserve(u32.size() + 4);
  for (char32_t cp : u32) {
    switch (cp) {
      case 0x0958:
        out.push_back(U'क');
        out.push_back(kNukta);
        break;
      case 0x0959:
        out.push_back(U'ख');
        out.push_back(kNukta);
        break;
      case 0x095A:
        out.push_back(U'ग');
        out.push_back(kNukta);
        break;
      case 0x095B:
        out.push_back(U'ज');
        out.push_back(kNukta);
        break;
      case 0x095C:
        out.push_back(U'ड');
        out.push_back(kNukta);
        break;
      case 0x095D:
        out.push_back(U'ढ');
        out.push_back(kNukta);
        break;
      case 0x095E:
        out.push_back(U'फ');
        out.push_back(kNukta);
        break;
      case 0x095F:
        out.push_back(U'य');
        out.push_back(kNukta);
        break;
      default:
        out.push_back(cp);
        break;
    }
  }
  return out;
}

absl::string_view ConsonantIpa(char32_t cp) {
  switch (cp) {
    case U'क':
      return "k";
    case U'ख':
      return "kʰ";
    case U'ग':
      return "ɡ";
    case U'घ':
      return "ɡʰ";
    case U'ङ':
      return "ŋ";
    case U'च':
      return "c";
    case U'छ':
      return "cʰ";
    case U'ज':
      return "ɟ";
    case U'झ':
      return "ɟʰ";
    case U'ञ':
      return "ɲ";
    case U'ट':
      return "ʈ";
    case U'ठ':
      return "ʈʰ";
    case U'ड':
      return "ɖ";
    case U'ढ':
      return "ɖʰ";
    case U'ण':
      return "ɳ";
    case U'त':
      return "t";
    case U'थ':
      return "tʰ";
    case U'द':
      return "d";
    case U'ध':
      return "dʰ";
    case U'न':
      return "n";
    case U'प':
      return "p";
    case U'फ':
      return "pʰ";
    case U'ब':
      return "b";
    case U'भ':
      return "bʰ";
    case U'म':
      return "m";
    case U'य':
      return "j";
    case U'र':
      return "ɾ";
    case U'ल':
      return "l";
    case U'व':
      return "ʋ";
    case U'ळ':
      return "ɭ";
    case U'श':
      return "ʃ";
    case U'ष':
      return "ʂ";
    case U'स':
      return "s";
    case U'ह':
      return "h";
    default:
      return "";
  }
}

absl::string_view NuktaConsonantIpa(char32_t base_cp) {
  switch (base_cp) {
    case U'क':
      return "q";
    case U'ख':
      return "x";
    case U'ग':
      return "ɣ";
    case U'ज':
      return "z";
    case U'फ':
      return "f";
    case U'ड':
      return "r.";
    case U'ढ':
      return "r.h";
    case U'य':
      return "j";
    case U'र':
      return "ɾ";
    case U'न':
      return "n";
    default:
      return ConsonantIpa(base_cp);
  }
}

absl::string_view IndependentVowelId(char32_t cp) {
  switch (cp) {
    case U'अ':
      return "a";
    case U'आ':
      return "aa";
    case U'इ':
      return "i";
    case U'ई':
      return "ii";
    case U'उ':
      return "u";
    case U'ऊ':
      return "U";
    case U'ए':
      return "e";
    case U'ऐ':
      return "ai";
    case U'ओ':
      return "o";
    case U'औ':
      return "au";
    case U'ऑ':
      return "O";
    case U'ऋ':
    case U'ॠ':
      return "ri";
    case U'ऍ':
      return "E";
    default:
      return "";
  }
}

absl::string_view MatraVowelId(char32_t cp) {
  switch (cp) {
    case 0x093E:
      return "aa";
    case 0x093F:
      return "i";
    case 0x0940:
      return "ii";
    case 0x0941:
      return "u";
    case 0x0942:
      return "uu";
    case 0x0947:
      return "e";
    case 0x0948:
      return "ai";
    case 0x094B:
      return "o";
    case 0x094C:
      return "au";
    case 0x0949:
      return "O";
    case 0x0943:
      return "ri";
    case 0x0945:
      return "E";
    default:
      return "";
  }
}

bool IsRephTrillTrigger(absl::string_view next_cons) {
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>> kSet({
      "m",
      "n",
      "ɳ",
      "ʋ",
      "j",
      "ʂ",
      "ʃ",
      "s",
      "z",
      "f",
      "h",
      "l",
      "ɾ",
  });
  return kSet->contains(next_cons);
}

bool IsNasalConsonant(absl::string_view c) {
  return c == "n" || c == "m" || c == "ɳ" || c == "ɲ" || c == "ŋ";
}

bool IsHLike(absl::string_view c) { return c == "h" || c == "r.h"; }

absl::string_view NasalPlaceForStop(absl::string_view next_cons) {
  if (next_cons == "k" || next_cons == "kʰ" || next_cons == "ɡ" ||
      next_cons == "ɡʰ") {
    return "ŋ";
  }
  if (next_cons == "c" || next_cons == "cʰ" || next_cons == "ɟ" ||
      next_cons == "ɟʰ" || next_cons == "j") {
    return "ɲ";
  }
  if (next_cons == "ʈ" || next_cons == "ʈʰ" || next_cons == "ɖ" ||
      next_cons == "ɖʰ") {
    return "ɳ";
  }
  if (next_cons == "t" || next_cons == "tʰ" || next_cons == "d" ||
      next_cons == "dʰ") {
    return "n";
  }
  if (next_cons == "p" || next_cons == "pʰ" || next_cons == "b" ||
      next_cons == "bʰ") {
    return "m";
  }
  return "n";
}

bool IsUnstressedWord(absl::string_view word, bool isolated) {
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>>
      kAlwaysUnstressed({
          "का",
          "की",
          "के",
          "को",
          "से",
          "है",
          "हैं",
          "हो",
          "था",
          "थी",
          "थे",
          "थीं",
          "और",
          "वे",
          "ये",
          "मैं",
      });
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>>
      kClauseUnstressed({
          "में",
          "हूँ",
          "हूं",
          "तो",
          "पे",
      });
  if (kAlwaysUnstressed->contains(word)) {
    return true;
  }
  return !isolated && kClauseUnstressed->contains(word);
}

bool IsSecondaryStressWord(absl::string_view word) {
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>> kSet({
      "वह", "यह",   "रहे",   "रही", "रहा",   "मेरी",   "मेरा",
      "मेरे", "उसका", "उसकी", "उसके", "हमारा", "हमारी", "हमारे",
      "तक", "हम",   "यदि",  "मुझे",  "हमे",    "आप",
  });
  return kSet->contains(word);
}

const absl::flat_hash_map<std::string, std::string>& GetHindiWordExceptions() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, std::string>>
      kExceptions({
          {"पसंदीदा", "pˌəsndˈiːdaː"},
          {"क्या", "kːjaː"},
          {"क्रिकेट", "kɾˈɪkɪt"},
          {"मीटिंग", "miʈˈĩɡ"},
          {"वो", "ʋoː"},
          {"न", "nˈə"},
          {"अगर", "ˌʌɡəɾ"},
          {"उनके", "ˌʊnkˌeː"},
          {"व", "ʋˈə"},
          {"आपको", "ˌaːpkˌoː"},
          {"आपके", "ˌaːpkˌeː"},
          {"तुम", "tˌʊm"},
          {"उनकी", "ˌʊnkˌi"},
          {"इसका", "ˌɪskˌaː"},
          {"आपकी", "ˌaːpkˌi"},
          {"उनका", "ˌʊnkˌaː"},
          {"बिना", "bˌɪnaː"},
          {"आपका", "ˌaːpkˌaː"},
          {"तू", "tuː"},
          {"जन्म", "ɟˈʌnm"},
          {"द", "dˈə"},
          {"कार्यक्रम", "kˈaːrjəkɾˌəm"},
          {"तेरा", "tˌeːɾaː"},
          {"शुभकामनाएं", "ʃˌʊbʰkˌaːmnˈaːẽː"},
          {"राष्ट्रपति", "ɾˌaːʃtɾapˈati"},
          {"तुम्हारे", "tʊmhˌaːɾeː"},
          {"तुझे", "tˌʊɟʰeː"},
          {"तुम्हारी", "tʊmhˌaːɾi"},
          {"लखनऊ", "lˈʌkʰnˈəu"},
          {"क", "kˈə"},
          {"इन्हें", "ˌɪnhẽː"},
          {"नेतृत्व", "nˈeːtɾɪtʋ"},
          {"तुम्हें", "tˌʊmhẽː"},
          {"महादेव", "mˌahaːdˈeʋ"},
          {"इनका", "ˌɪnkˌaː"},
          {"छत्तीसगढ़", "cʰˌətːisəɡˈʌr.h"},
          {"पहुंच", "pəhˈũc"},
          {"तुम्हारा", "tʊmhˌaːɾaː"},
          {"महाराज", "mˌahaːɾˈaːɟ"},
          {"व्यवहार", "ʋjˌəʋhˈaːɾ"},
          {"ह", "hˈə"},
          {"कार्यकर्ताओं", "kˈaːrjəkrtaːw"},
          {"कार्यकर्ता", "kˈaːrjəkrtaː"},
          {"मतदान", "matdˈaːn"},
          {"इंग्लैंड", "ˈĩɡlɛ̃ɖ"},
          {"पहुंचे", "pəhˈũceː"},
          {"र", "ɾˈə"},
          {"मशहूर", "mˌəʃhˈuːɾ"},
          {"वैज्ञानिक", "ʋˈɛːɡjaːnˌɪk"},
          {"अकाउंट", "ˌʌkaːˈʊɳʈ"},
          {"पहुंचा", "pəhˈũcaː"},
      });
  return *kExceptions;
}

enum class UnitKind { kConsonant, kVowel, kNasal, kIgnored };
enum class StressLevel { kUnstressed, kPrimary, kSecondary };

struct Unit {
  UnitKind kind = UnitKind::kConsonant;
  // For kConsonant: IPA string; for kVowel: vowel ID ("@", "#", "a", "aa",
  // "i", "ii", "u", "uu", "U", "e", "ai", "o", "au", "O", "E", "ri"); for
  // kNasal: "ans" (anusvara) or "cb" (chandrabindu).
  std::string val;
  char32_t base_cp = 0;
  bool independent = false;
  bool conjunct = false;
  StressLevel stress = StressLevel::kUnstressed;
};

std::vector<Unit> ParseDevanagariUnits(const std::u32string& norm_word) {
  std::vector<Unit> units;
  units.reserve(norm_word.size() * 2);
  const size_t n = norm_word.size();
  size_t i = 0;
  while (i < n) {
    const char32_t ch = norm_word[i];
    const absl::string_view base_cons = ConsonantIpa(ch);
    if (!base_cons.empty()) {
      std::string ipa;
      if (i + 1 < n && norm_word[i + 1] == kNukta) {
        ipa = std::string(NuktaConsonantIpa(ch));
        i += 2;
      } else {
        ipa = std::string(base_cons);
        ++i;
      }
      if (!units.empty() && units.back().kind == UnitKind::kConsonant) {
        if (ch == U'ष' && units.back().val == "k") {
          ipa = "ʃ";  // क्ष -> kʃ
        } else if (ch == U'ञ' && units.back().base_cp == U'ज') {
          units.back().val = "ɡ";  // ज्ञ -> ɡj
          ipa = "j";
        }
      }
      units.push_back({UnitKind::kConsonant, std::move(ipa), ch, false, false});
      if (i < n && norm_word[i] == kHalant) {
        ++i;
      } else if (i < n && !MatraVowelId(norm_word[i]).empty()) {
        units.push_back({UnitKind::kVowel,
                         std::string(MatraVowelId(norm_word[i])), 0, false,
                         false});
        ++i;
      } else {
        units.push_back({UnitKind::kVowel, "@", 0, false, false});
      }
      continue;
    }
    const absl::string_view ind_v = IndependentVowelId(ch);
    if (!ind_v.empty()) {
      units.push_back({UnitKind::kVowel, std::string(ind_v), ch, true, false});
      ++i;
      continue;
    }
    if (ch == kAnusvara || ch == kChandrabindu) {
      units.push_back({UnitKind::kNasal, (ch == kAnusvara) ? "ans" : "cb", ch,
                       false, false});
      ++i;
      continue;
    }
    if (ch == kVisarga) {
      units.push_back({UnitKind::kConsonant, "h", ch, false, false});
      ++i;
      continue;
    }
    ++i;
  }

  for (size_t k = 0; k + 1 < units.size(); ++k) {
    if (units[k].kind == UnitKind::kConsonant && units[k].base_cp == U'र' &&
        units[k + 1].kind == UnitKind::kConsonant &&
        IsRephTrillTrigger(units[k + 1].val)) {
      units[k].val = "r";
    }
  }
  for (size_t k = 0; k < units.size(); ++k) {
    if (units[k].kind == UnitKind::kConsonant) {
      units[k].conjunct =
          (k + 1 < units.size() && units[k + 1].kind == UnitKind::kConsonant);
    }
  }
  return units;
}

void AssignUnderlyingStress(std::vector<Unit>& units, absl::string_view word,
                            bool isolated) {
  std::vector<size_t> vidx;
  for (size_t k = 0; k < units.size(); ++k) {
    if (units[k].kind == UnitKind::kVowel) {
      vidx.push_back(k);
    }
  }
  const size_t n = vidx.size();
  if (n == 0 || IsUnstressedWord(word, isolated)) {
    return;
  }

  std::vector<int> heavy(n, 0);
  for (size_t j = 0; j < n; ++j) {
    const size_t k = vidx[j];
    const absl::string_view v = units[k].val;
    const bool long_v = (v != "@" && v != "a" && v != "i" && v != "u" &&
                         v != "O" && v != "E" && v != "U" && v != "ri");
    const size_t nxt = (j + 1 < n) ? vidx[j + 1] : units.size();
    int ncons = 0;
    for (size_t m = k + 1; m < nxt; ++m) {
      if (units[m].kind == UnitKind::kConsonant) {
        ++ncons;
      }
    }
    bool nasal = false;
    for (size_t m = k + 1; m < nxt; ++m) {
      if (units[m].kind == UnitKind::kNasal &&
          (v == "@" || v == "a" || (units[m].val == "cb" && ncons >= 1))) {
        nasal = true;
        break;
      }
    }
    const bool closed =
        (j + 1 == n) ? (ncons >= 1 || nasal) : (ncons >= 2 || nasal);
    heavy[j] = (long_v ? 1 : 0) + (closed ? 1 : 0);
  }

  if (n == 1) {
    units[vidx[0]].stress = StressLevel::kPrimary;
  } else {
    size_t primary = 0;
    int best = 0;
    for (size_t j = 0; j + 1 < n; ++j) {
      best = std::max(best, heavy[j]);
    }
    if (best > 0) {
      for (size_t j = n - 1; j > 0; --j) {
        if (heavy[j - 1] == best) {
          primary = j - 1;
          break;
        }
      }
    }
    units[vidx[primary]].stress = StressLevel::kPrimary;
    if (primary >= 2) {
      units[vidx[0]].stress = StressLevel::kSecondary;
    }
    if (primary + 2 < n && heavy[primary + 1] == 0) {
      units[vidx[primary + 2]].stress = StressLevel::kSecondary;
    }
  }

  if (IsSecondaryStressWord(word)) {
    for (size_t idx : vidx) {
      if (units[idx].stress != StressLevel::kUnstressed) {
        units[idx].stress = StressLevel::kSecondary;
      }
    }
  }
}

bool IsConsAt(const std::vector<Unit>& units, int k) {
  return k >= 0 && static_cast<size_t>(k) < units.size() &&
         units[k].kind == UnitKind::kConsonant;
}

bool IsVowelAt(const std::vector<Unit>& units, int k) {
  if (k < 0 || static_cast<size_t>(k) >= units.size()) {
    return false;
  }
  const Unit& u = units[k];
  if (u.kind == UnitKind::kVowel) {
    return u.val != "#";
  }
  if (u.kind == UnitKind::kNasal) {
    if (k < 1 || units[k - 1].kind != UnitKind::kVowel ||
        units[k - 1].val == "@" || units[k - 1].val == "#") {
      return false;
    }
    if (u.val == "cb" && static_cast<size_t>(k + 1) < units.size() &&
        units[k + 1].kind == UnitKind::kConsonant) {
      return false;
    }
    return true;
  }
  return false;
}

const Unit* NextPronouncedVowel(const std::vector<Unit>& units, size_t start) {
  for (size_t j = start; j < units.size(); ++j) {
    if (units[j].kind == UnitKind::kVowel && units[j].val != "#") {
      return &units[j];
    }
  }
  return nullptr;
}

bool IsGeminatePair(absl::string_view a, absl::string_view b) {
  return a == b || b == absl::StrCat(a, "ʰ");
}

void DeleteSchwas(std::vector<Unit>& units) {
  if (units.empty()) {
    return;
  }
  int last = static_cast<int>(units.size()) - 1;
  bool trailing_n = false;
  while (last >= 0 && units[last].kind == UnitKind::kNasal) {
    trailing_n = true;
    --last;
  }
  int nvow = 0;
  for (const Unit& u : units) {
    if (u.kind == UnitKind::kVowel) {
      ++nvow;
    }
  }
  if (last >= 1 && units[last].kind == UnitKind::kVowel &&
      units[last].val == "@" && nvow > 1 && !trailing_n) {
    const Unit& c1 = units[last - 1];
    const Unit* c0 = (last >= 2) ? &units[last - 2] : nullptr;
    bool keep = false;
    if (c0 != nullptr && c0->kind == UnitKind::kConsonant) {
      if (c1.val == "j" || c1.val == "ɾ") {
        keep = true;
      } else if ((IsNasalConsonant(c1.val) || c1.val == "l") &&
                 !((c0->val == "r" || c0->val == "ɾ") && c1.val == "m")) {
        keep = true;
      }
    }
    if (!keep) {
      units[last].val = "#";
    }
  }

  // Right-to-left medial schwa deletion in V C _ C (C) V.
  int k = static_cast<int>(units.size()) - 1;
  while (k >= 0) {
    Unit& u = units[k];
    if (u.kind == UnitKind::kVowel && u.val == "@" && k > 0 &&
        u.stress != StressLevel::kPrimary) {
      int left = k - 2;
      if (IsConsAt(units, left) && IsConsAt(units, k - 1) &&
          IsGeminatePair(units[left].val, units[k - 1].val)) {
        --left;
      }
      bool ok = IsConsAt(units, k - 1) && IsVowelAt(units, left);
      if (ok && static_cast<size_t>(k + 1) < units.size() &&
          units[k + 1].kind == UnitKind::kVowel && units[k + 1].independent) {
        u.val = "#";
        k -= 2;
        continue;
      }
      ok = ok && IsConsAt(units, k + 1);
      if (ok) {
        if (IsVowelAt(units, k + 2)) {
          // V C _ C V
        } else if (IsConsAt(units, k + 2) && IsVowelAt(units, k + 3) &&
                   !IsHLike(units[k + 1].val)) {
          // V C _ C C V
        } else {
          ok = false;
        }
      }
      if (ok) {
        const Unit* nv = NextPronouncedVowel(units, static_cast<size_t>(k + 1));
        if (nv != nullptr && nv->val == "@") {
          ok = false;
        }
      }
      if (ok && (IsHLike(units[k - 1].val) || IsHLike(units[k + 1].val))) {
        ok = false;
      }
      if (ok) {
        u.val = "#";
        k -= 2;
        continue;
      }
    }
    --k;
  }
}

enum class SegKind { kConsonant, kVowel, kDeleted };

struct Segment {
  SegKind kind = SegKind::kConsonant;
  std::string val;
  bool conj = false;
  bool has_nas = false;
  std::string nas_kind;
  std::string nas_next_cons;
  bool nasalized = false;
  bool independent = false;
  bool schwa = false;
  StressLevel stress = StressLevel::kUnstressed;
};

std::vector<Segment> UnitsToSegments(std::vector<Unit>& units) {
  std::vector<Segment> segs;
  segs.reserve(units.size() + 2);
  for (size_t idx = 0; idx < units.size(); ++idx) {
    Unit& u = units[idx];
    if (u.kind == UnitKind::kConsonant) {
      Segment s;
      s.kind = SegKind::kConsonant;
      s.val = u.val;
      s.conj = u.conjunct;
      segs.push_back(std::move(s));
    } else if (u.kind == UnitKind::kVowel) {
      if (u.val == "#") {
        if (u.stress != StressLevel::kUnstressed) {
          Segment s;
          s.kind = SegKind::kDeleted;
          s.stress = u.stress;
          segs.push_back(std::move(s));
        }
        continue;
      }
      std::string vid = (u.val == "@") ? "a" : u.val;
      if (vid == "ri") {
        Segment r_seg;
        r_seg.kind = SegKind::kConsonant;
        r_seg.val = "ɾ";
        segs.push_back(std::move(r_seg));
        vid = "i";
      }
      const bool prev_v =
          (!segs.empty() && segs.back().kind == SegKind::kVowel);
      const bool nxt_n =
          (idx + 1 < units.size() && units[idx + 1].kind == UnitKind::kNasal);
      if (u.independent && prev_v && segs.back().val == "aa" && nxt_n &&
          idx + 2 == units.size()) {
        if (vid == "o") {
          Segment w_seg;
          w_seg.kind = SegKind::kConsonant;
          w_seg.val = "w";
          segs.push_back(std::move(w_seg));
          units[idx + 1].kind = UnitKind::kIgnored;
          continue;
        }
        if (vid == "e" && units[idx + 1].val == "cb") {
          Segment j_seg;
          j_seg.kind = SegKind::kConsonant;
          j_seg.val = "j";
          segs.push_back(std::move(j_seg));
          units[idx + 1].kind = UnitKind::kIgnored;
          continue;
        }
      }
      Segment v_seg;
      v_seg.kind = SegKind::kVowel;
      v_seg.val = std::move(vid);
      v_seg.independent = u.independent;
      v_seg.schwa = (u.val == "@");
      v_seg.stress = u.stress;
      segs.push_back(std::move(v_seg));
    } else if (u.kind == UnitKind::kNasal) {
      std::string nxt_cons;
      for (size_t m = idx + 1; m < units.size(); ++m) {
        if (units[m].kind == UnitKind::kConsonant) {
          nxt_cons = units[m].val;
          break;
        }
        if (units[m].kind == UnitKind::kVowel && units[m].val != "#") {
          break;
        }
      }
      if (!segs.empty() && segs.back().kind == SegKind::kVowel) {
        segs.back().has_nas = true;
        segs.back().nas_kind = u.val;
        segs.back().nas_next_cons = nxt_cons;
      } else {
        Segment n_seg;
        n_seg.kind = SegKind::kConsonant;
        n_seg.val =
            nxt_cons.empty() ? "n" : std::string(NasalPlaceForStop(nxt_cons));
        segs.push_back(std::move(n_seg));
      }
    }
  }
  return segs;
}

bool IsGeminateLenConsonant(absl::string_view c) {
  return c == "k" || c == "t" || c == "p" || c == "c" || c == "d" || c == "ɟ" ||
         c == "ɡ";
}

absl::string_view AspiratedGeminateReplacement(absl::string_view a,
                                               absl::string_view b) {
  if (a == "t" && b == "tʰ") return "tʰː";
  if (a == "d" && b == "dʰ") return "dʰː";
  if (a == "k" && b == "kʰ") return "kkʰ";
  if (a == "ɖ" && b == "ɖʰ") return "ɖʰɖʰ";
  if (a == "c" && b == "cʰ") return "cʰcʰ";
  if (a == "p" && b == "pʰ") return "pʰpʰ";
  if (a == "ɡ" && b == "ɡʰ") return "ɡʰː";
  if (a == "b" && b == "bʰ") return "bʰː";
  if (a == "ɟ" && b == "ɟʰ") return "ɟʰː";
  return "";
}

std::vector<Segment> RealizeSegments(const std::vector<Segment>& raw_segs) {
  StressLevel del_st = StressLevel::kUnstressed;
  if (!raw_segs.empty() && raw_segs.back().kind == SegKind::kDeleted) {
    del_st = raw_segs.back().stress;
  }
  std::vector<Segment> segs;
  segs.reserve(raw_segs.size());
  for (const Segment& s : raw_segs) {
    if (s.kind != SegKind::kDeleted) {
      segs.push_back(s);
    }
  }

  std::vector<Segment> out;
  out.reserve(segs.size() + 2);
  for (const Segment& s : segs) {
    if (s.kind == SegKind::kConsonant || !s.has_nas) {
      out.push_back(s);
      continue;
    }
    if (s.val == "a" && s.nas_kind == "ans") {
      Segment copy = s;
      copy.has_nas = false;
      out.push_back(std::move(copy));
      Segment n_seg;
      n_seg.kind = SegKind::kConsonant;
      n_seg.val = s.nas_next_cons.empty()
                      ? "n"
                      : std::string(NasalPlaceForStop(s.nas_next_cons));
      out.push_back(std::move(n_seg));
      continue;
    }
    if (s.independent && s.nas_kind == "cb" && !s.nas_next_cons.empty()) {
      Segment copy = s;
      copy.has_nas = false;
      out.push_back(std::move(copy));
      Segment n_seg;
      n_seg.kind = SegKind::kConsonant;
      n_seg.val = "n";
      out.push_back(std::move(n_seg));
      continue;
    }
    Segment copy = s;
    copy.nasalized = true;
    out.push_back(std::move(copy));
    if (s.nas_kind == "cb" && !s.nas_next_cons.empty()) {
      Segment n_seg;
      n_seg.kind = SegKind::kConsonant;
      n_seg.val = "n";
      out.push_back(std::move(n_seg));
    }
  }

  // Word-final `-अय`: schwa + j at word end -> `ɛj` ("Ej").
  if (out.size() >= 3 && out.back().kind == SegKind::kConsonant &&
      out.back().val == "j" && out[out.size() - 2].kind == SegKind::kVowel &&
      out[out.size() - 2].val == "a" && out[out.size() - 2].schwa &&
      out[out.size() - 3].kind == SegKind::kConsonant) {
    const bool cluster =
        (out.size() >= 4 && out[out.size() - 4].kind == SegKind::kConsonant);
    const Segment prev_schwa = out[out.size() - 2];
    out.pop_back();
    out.pop_back();
    if (cluster) {
      out.push_back(prev_schwa);
    }
    Segment ej_seg;
    ej_seg.kind = SegKind::kVowel;
    ej_seg.val = "Ej";
    ej_seg.stress = cluster ? del_st
                            : ((prev_schwa.stress == StressLevel::kPrimary)
                                   ? StressLevel::kUnstressed
                                   : del_st);
    out.push_back(std::move(ej_seg));
    if (!cluster && prev_schwa.stress == StressLevel::kPrimary) {
      out.insert(out.end() - 1, prev_schwa);
    }
    Segment j_seg;
    j_seg.kind = SegKind::kConsonant;
    j_seg.val = "j";
    out.push_back(std::move(j_seg));
  }

  // Gemination conventions.
  std::vector<Segment> res;
  res.reserve(out.size());
  size_t k = 0;
  while (k < out.size()) {
    const Segment& s = out[k];
    if (s.kind == SegKind::kConsonant && k + 1 < out.size() &&
        out[k + 1].kind == SegKind::kConsonant) {
      const absl::string_view a = s.val;
      const absl::string_view b = out[k + 1].val;
      if (a == b && IsGeminateLenConsonant(a)) {
        Segment gem;
        gem.kind = SegKind::kConsonant;
        gem.val = absl::StrCat(a, "ː");
        res.push_back(std::move(gem));
        k += 2;
        continue;
      }
      const absl::string_view asp_gem = AspiratedGeminateReplacement(a, b);
      if (!asp_gem.empty()) {
        Segment gem;
        gem.kind = SegKind::kConsonant;
        gem.val = std::string(asp_gem);
        res.push_back(std::move(gem));
        k += 2;
        continue;
      }
      if (b == "j" && (a == "k" || a == "t" || a == "p") && s.conj) {
        Segment gem;
        gem.kind = SegKind::kConsonant;
        gem.val = absl::StrCat(a, "ː");
        res.push_back(std::move(gem));
        ++k;
        continue;
      }
    }
    res.push_back(s);
    ++k;
  }
  return res;
}

absl::string_view OralVowelIpa(absl::string_view vid) {
  if (vid == "a") return "ə";
  if (vid == "aa") return "aː";
  if (vid == "i") return "ɪ";
  if (vid == "ii") return "iː";
  if (vid == "u") return "ʊ";
  if (vid == "uu") return "uː";
  if (vid == "U") return "u";
  if (vid == "e") return "eː";
  if (vid == "ai") return "ɛː";
  if (vid == "o") return "oː";
  if (vid == "au") return "ɔː";
  if (vid == "O") return "ɔ";
  if (vid == "E") return "ɛ";
  return "";
}

absl::string_view NasalVowelIpa(absl::string_view vid) {
  if (vid == "a") return "ʌ\u0303";
  if (vid == "aa") return "a\u0303";
  if (vid == "i" || vid == "ii") return "i\u0303";
  if (vid == "u" || vid == "uu" || vid == "U") return "u\u0303";
  if (vid == "e") return "e\u0303ː";
  if (vid == "ai" || vid == "E") return "ɛ\u0303";
  if (vid == "o") return "o\u0303";
  if (vid == "au" || vid == "O") return "ɔ\u0303";
  return "";
}

absl::string_view HindiUnder100Ipa(int n) {
  static constexpr absl::string_view kTable[100] = {
      "ʃˈuːnjə",       "ˈeːk",          "dˈoː",          "tˈiːn",
      "cˈaːɾ",         "pˈãc",          "cʰˈɛː",         "sˈaːt",
      "ˈaːʈʰ",         "nˈɔː",          "dˈʌs",          "ɡjˈaːɾəh",
      "bˈaːɾəh",       "tˈeːɾəh",       "cˈɔːdəh",       "pˈʌndɾəh",
      "sˈoːləh",       "sˈʌtɾəh",       "ʌʈʰˈaːɾəh",     "ʊnnˈiːs",
      "bˈiːs",         "ɪkkˈiːs",       "bˈaːiːs",       "tˈeːiːs",
      "cˈɔːbiːs",      "pəccˈiːs",      "cʰəbbˈis",      "səttˈaːiːs",
      "ʌʈʰːˈaːiːs",    "ʊntˈiːs",       "tˈiːs",         "ɪktˈiːs",
      "bəttˈiːs",      "tɛ̃tˈiːs",       "cɔ̃tˈiːs",       "pɛ̃tˈiːs",
      "cʰəttˈiːs",     "sɛ̃tˈiːs",       "ʌr.tˈiːs",      "ʊntˈaːlis",
      "cˈaːliːs",      "ɪktˈaːlis",     "bəjˈaːlis",     "tɛ̃tˈaːlis",
      "cəʋˈaːlis",     "pɛ̃tˈaːlis",     "cʰɪjˈaːlis",    "sɛ̃tˈaːlis",
      "ʌr.tˈaːlis",    "ʊncˈaːs",       "pəcˈaːs",       "ɪkjˈaːʋən",
      "bˈaːʋən",       "tɪɾpˈʌn",       "cˈɔːʋən",       "pəcpˈʌn",
      "cʰəppˈʌn",      "səttˈaːʋən",    "ʌʈʰːˈaːʋən",    "ʊnsˈʌʈʰ",
      "sˈaːʈʰ",        "ɪksˈʌʈʰ",       "bˈaːsəʈʰ",      "tɪɾsˈʌʈʰ",
      "cˈɔ̃səʈʰ",       "pˈɛ̃səʈʰ",       "cʰɪjˈaːsəʈʰ",   "sərsˈʌʈʰ",
      "ʌr.sˈʌʈʰ",      "ʊnhˈʌttəɾ",     "sˈʌttəɾ",       "ɪkhˈʌttəɾ",
      "bəhˈʌttəɾ",     "tɪhˈʌttəɾ",     "cɔːhˈʌttəɾ",    "pəchˈʌttəɾ",
      "cʰɪhˈʌttəɾ",    "səthˈʌttəɾ",    "ʌʈʰhˈʌttəɾ",    "ʊnˈaːsi",
      "ˈʌssi",         "ɪkjˈaːsi",      "bəjˈaːsi",      "tɪɾˈaːsi",
      "cɔːɾˈaːsi",     "pəcˈaːsi",      "cʰɪjˈaːsi",     "sətˈaːsi",
      "ʌʈʰˈaːsi",      "nəʋˈaːsi",      "nˈʌbbeː",       "ɪkːjˈaːnəbˌeː",
      "bˈaːnəbˌeː",    "tɪɾˈaːnəbˌeː",  "cɔːɾˈaːnəbˌeː", "pəncˈaːnəbˌeː",
      "cʰɪjˈaːnəbˌeː", "səttˈaːnəbˌeː", "ʌtʰːˈaːnəbˌeː", "nɪɲjˈaːnəbˌeː",
  };
  if (n >= 0 && n < 100) {
    return kTable[n];
  }
  return "";
}

absl::string_view HindiHundredsIpa(int h) {
  static constexpr absl::string_view kHundreds[10] = {
      "",         "ˈeːksˈɔː",  "dˈoːsˈɔː",  "tˈiːnsˈɔː", "cˈaːrsˈɔː",
      "pˈãcsˈɔː", "cʰˈɛːsˈɔː", "sˈaːtsˈɔː", "ˈaːʈʰsˈɔː", "nˈɔːsˈɔː",
  };
  if (h >= 1 && h <= 9) {
    return kHundreds[h];
  }
  return "";
}

std::string NormalizeDigitsToAscii(const std::u32string& u32) {
  std::string out;
  out.reserve(u32.size());
  for (char32_t cp : u32) {
    if (cp >= U'0' && cp <= U'9') {
      out.push_back(static_cast<char>(cp));
    } else if (cp >= 0x0966 && cp <= 0x096F) {
      out.push_back(static_cast<char>('0' + (cp - 0x0966)));
    } else {
      return "";
    }
  }
  return out;
}

std::string HindiNumberToIpa(absl::string_view ascii_digits) {
  if (ascii_digits == "00") {
    return "ʃˈuːnjə ʃˈuːnjə";
  }
  if (ascii_digits == "000") {
    return "ʃˈuːnjəʃˈuːnjə ʃˈuːnjə";
  }
  if (ascii_digits.empty()) {
    return "";
  }
  if (ascii_digits.size() > 1 && ascii_digits[0] == '0') {
    std::vector<absl::string_view> parts;
    parts.reserve(ascii_digits.size());
    for (char ch : ascii_digits) {
      parts.push_back(HindiUnder100Ipa(ch - '0'));
    }
    return absl::StrJoin(parts, " ");
  }
  if (ascii_digits.size() > 5) {
    std::vector<absl::string_view> parts;
    parts.reserve(ascii_digits.size());
    for (char ch : ascii_digits) {
      parts.push_back(HindiUnder100Ipa(ch - '0'));
    }
    return absl::StrJoin(parts, " ");
  }
  int64_t val = 0;
  for (char ch : ascii_digits) {
    val = val * 10 + (ch - '0');
  }
  if (val < 100) {
    return std::string(HindiUnder100Ipa(static_cast<int>(val)));
  }
  if (val < 1000) {
    const int h = static_cast<int>(val / 100);
    const int rem = static_cast<int>(val % 100);
    const absl::string_view prefix = HindiHundredsIpa(h);
    if (rem == 0) {
      return std::string(prefix);
    }
    return absl::StrCat(prefix, " ", HindiNumberToIpa(absl::StrCat(rem)));
  }
  const int th = static_cast<int>(val / 1000);
  const int rem = static_cast<int>(val % 1000);
  const absl::string_view th_ipa =
      (th == 98) ? "ʌtʰːˈaːnəbeː" : HindiUnder100Ipa(th);
  const std::string prefix = absl::StrCat(th_ipa, " hˈʌɟaːɾ");
  if (rem == 0) {
    return prefix;
  }
  return absl::StrCat(prefix, " ", HindiNumberToIpa(absl::StrCat(rem)));
}

std::string PhonemizeHindiWord(const std::u32string& raw_word, bool isolated) {
  const std::string digits = NormalizeDigitsToAscii(raw_word);
  if (!digits.empty()) {
    return HindiNumberToIpa(digits);
  }
  const std::u32string norm_u32 = NormalizeDevanagariU32(raw_word);
  const std::string norm_utf8 = CodePointsToUtf8(norm_u32);
  const auto& exceptions = GetHindiWordExceptions();
  if (const auto it = exceptions.find(norm_utf8); it != exceptions.end()) {
    return it->second;
  }

  std::vector<Unit> units = ParseDevanagariUnits(norm_u32);
  if (units.empty()) {
    return "";
  }
  AssignUnderlyingStress(units, norm_utf8, isolated);
  DeleteSchwas(units);
  const std::vector<Segment> segs = RealizeSegments(UnitsToSegments(units));

  std::string res;
  res.reserve(segs.size() * 3);
  for (size_t k = 0; k < segs.size(); ++k) {
    const Segment& s = segs[k];
    if (s.kind == SegKind::kConsonant) {
      res.append(s.val);
      continue;
    }
    if (s.stress == StressLevel::kPrimary) {
      res.append("ˈ");
    } else if (s.stress == StressLevel::kSecondary) {
      res.append("ˌ");
    }
    if (s.val == "Ej") {
      res.append("ɛ");
      continue;
    }
    if (s.nasalized) {
      if (s.val == "a") {
        res.append((s.stress == StressLevel::kPrimary) ? "ʌ\u0303" : "ə\u0303");
      } else {
        res.append(NasalVowelIpa(s.val));
      }
    } else if (s.val == "a") {
      res.append((s.stress == StressLevel::kPrimary || s.independent) ? "ʌ"
                                                                      : "ə");
    } else if (s.val == "ii") {
      res.append((s.stress == StressLevel::kPrimary && k + 1 != segs.size())
                     ? "iː"
                     : "i");
    } else {
      res.append(OralVowelIpa(s.val));
    }
  }
  return res;
}

bool IsDigitCp(char32_t cp) {
  return (cp >= U'0' && cp <= U'9') || (cp >= 0x0966 && cp <= 0x096F);
}

bool IsDevanagariWordCp(char32_t cp) {
  return cp >= 0x0900 && cp <= 0x097F && cp != 0x0964 && cp != 0x0965 &&
         !(cp >= 0x0966 && cp <= 0x096F);
}

bool IsWhitespaceCp(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r';
}

enum class RawTokenKind { kSpace, kDigits, kWord, kPunct };

struct RawToken {
  RawTokenKind kind;
  std::u32string u32;
  std::string utf8;
};

std::vector<RawToken> TokenizeHindi(absl::string_view text) {
  const std::u32string u32 = Utf8ToU32(text);
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
    if (IsDigitCp(cp)) {
      std::u32string span;
      while (i < u32.size() && IsDigitCp(u32[i])) {
        span.push_back(u32[i]);
        ++i;
      }
      out.push_back({RawTokenKind::kDigits, span, CodePointsToUtf8(span)});
      continue;
    }
    if (IsDevanagariWordCp(cp) ||
        (cp <= 0x7F && absl::ascii_isalpha(static_cast<unsigned char>(cp)))) {
      std::u32string span;
      while (i < u32.size() &&
             (IsDevanagariWordCp(u32[i]) ||
              (u32[i] <= 0x7F &&
               absl::ascii_isalpha(static_cast<unsigned char>(u32[i]))))) {
        span.push_back(u32[i]);
        ++i;
      }
      out.push_back({RawTokenKind::kWord, span, CodePointsToUtf8(span)});
      continue;
    }
    std::u32string single(1, cp);
    out.push_back({RawTokenKind::kPunct, single, CodePointsToUtf8(single)});
    ++i;
  }
  return out;
}

struct SentenceItem {
  bool is_word = true;
  std::string ipa;
  bool tight_before = false;
};

std::string PhonemizeHindiSentence(
    absl::string_view text, bool isolated,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) {
  const std::vector<RawToken> raw_tokens = TokenizeHindi(text);
  std::vector<SentenceItem> items;
  items.reserve(raw_tokens.size());

  const size_t n = raw_tokens.size();
  size_t i = 0;
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
                              !items.empty() && !items.back().is_word;

    if (tok.kind == RawTokenKind::kDigits) {
      items.push_back(
          {true, PhonemizeHindiWord(tok.u32, isolated), tight_before});
      ++i;
      continue;
    }

    if (tok.kind == RawTokenKind::kPunct) {
      if (tok.utf8 == "₹") {
        ++i;
        continue;
      }
      if (tok.utf8 == "+" && next_tok.kind == RawTokenKind::kDigits) {
        items.push_back({true, "plˈʌs", tight_before});
        ++i;
        continue;
      }
      if (tok.utf8 == "%" && prev_tok.kind == RawTokenKind::kDigits) {
        items.push_back({true, "pətˈɪʃət", false});
        ++i;
        continue;
      }
      if (tok.utf8 == "/" && prev_tok.kind == RawTokenKind::kDigits &&
          next_tok.kind == RawTokenKind::kDigits) {
        items.push_back({true, "slˈaʃ", false});
        ++i;
        continue;
      }
      if (tok.utf8 == "-" && prev_tok.kind == RawTokenKind::kDigits &&
          next_tok.kind == RawTokenKind::kDigits) {
        ++i;
        continue;
      }
      if (tok.utf8 == "।" || tok.utf8 == "॥") {
        items.push_back({false, ".", false});
        ++i;
        continue;
      }
      items.push_back({false, tok.utf8, false});
      ++i;
      continue;
    }

    const std::string lower_w = absl::AsciiStrToLower(tok.utf8);
    if (!custom_lexicon.empty()) {
      if (const auto it = custom_lexicon.find(lower_w);
          it != custom_lexicon.end()) {
        items.push_back({true, it->second, tight_before});
        ++i;
        continue;
      }
    }
    items.push_back(
        {true, PhonemizeHindiWord(tok.u32, isolated), tight_before});
    ++i;
  }

  std::string out;
  out.reserve(text.size() * 2);
  for (const SentenceItem& it : items) {
    if (!it.is_word) {
      if (!out.empty() && out.back() == ' ') {
        out.pop_back();
      }
      out.append(it.ipa);
      out.push_back(' ');
    } else {
      if (it.tight_before && !out.empty() && out.back() == ' ') {
        out.pop_back();
      }
      out.append(it.ipa);
      out.push_back(' ');
    }
  }
  while (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

}  // namespace

HindiG2p::HindiG2p(int /*unused*/) {}

absl::StatusOr<HindiG2p> HindiG2p::Create() { return HindiG2p(0); }

absl::StatusOr<std::string> HindiG2p::DoTextToIpa(
    absl::string_view text,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  return PhonemizeHindiSentence(text, /*isolated=*/false, custom_lexicon);
}

absl::StatusOr<std::string> HindiG2p::DoWordToIpa(
    absl::string_view word,
    const absl::flat_hash_map<std::string, std::string>& custom_lexicon) const {
  return PhonemizeHindiSentence(word, /*isolated=*/true, custom_lexicon);
}

}  // namespace litert::omni::tts::kokoro
