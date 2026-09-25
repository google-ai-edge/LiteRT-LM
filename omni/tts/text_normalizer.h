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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_NORMALIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_NORMALIZER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::omni::tts {

// Rewrites digits and symbols into words that a grapheme-to-phoneme front end
// can pronounce, driven entirely by a rule table supplied at construction.
//
// This class holds no language-specific data. Which patterns to match, what to
// replace them with, and how to spell out a digit all live in the rule table,
// which ships as data alongside the model. A language that has no rule table
// is simply left unnormalized.
//
// Rule table format
// -----------------
// A UTF-8 text blob of tab-separated records, one per line. Blank lines and
// lines whose first non-space character is '#' are ignored.
//
//   digit <TAB> <0-9> <TAB> <glyph>
//       How to spell out one digit, for the `#N` and `@N` expansions below.
//
//   unit <TAB> <ten> <TAB> <hundred> <TAB> <thousand> <TAB> <myriad> <TAB>
//   <hundred_million>
//       Positional place-value units (10, 100, 1000, 10^4, 10^8) used by the
//       `@N` cardinal number expansion below.
//
//   rule <TAB> <pattern> <TAB> <replacement>
//       An RE2 pattern and what to replace every match of it with. Rules are
//       applied in the order they appear, each to the whole output of the
//       previous one, so a table can layer rewrites: recognize a percentage
//       first, then let a later rule reach the decimal point inside it.
//
// Within a replacement:
//   \1 through \9   the capture group, verbatim.
//   #1 through #9   the capture group with every ASCII digit replaced by its
//                   glyph. This reads a run of digits out one at a time -- a
//                   year, or the fractional part of a decimal -- instead of as
//                   a cardinal.
//   @1 through @9   the capture group expanded as a cardinal integer using the
//                   `digit` and `unit` tables (e.g. "24" -> "二十四",
//                   "10000" -> "一万").
//   \\  \#  \@      a literal backslash, '#', or '@'.
//   \t  \n          a literal tab or newline.
//
// Patterns are RE2, which has no lookahead or lookbehind, so a rule has to
// consume whatever context it needs and put it back in the replacement.
//
// Example (Mandarin):
//   digit <TAB> 0 <TAB> 零
//   unit  <TAB> 十 <TAB> 百 <TAB> 千 <TAB> 万 <TAB> 亿
//   rule  <TAB> \b(\d{4})\s*年  <TAB> #1年      2026年 -> 二零二六年
//   rule  <TAB> (\d+)\.(\d+)    <TAB> @1点#2    3.14   -> 三点一四
//   rule  <TAB> (\d+)           <TAB> @1        24號   -> 二十四號

class TextNormalizer {
 public:
  // Parses a rule table.
  //
  // args
  // - rule_table: Contents of the rule table, in the format described above.
  //
  // returns
  // - A normalizer applying those rules, or an InvalidArgumentError naming the
  //   offending line if the table is malformed.
  static absl::StatusOr<std::unique_ptr<TextNormalizer>> Create(
      absl::string_view rule_table);

  // Applies every rule, in table order, to the whole text.
  //
  // args
  // - text: Input text.
  //
  // returns
  // - The rewritten text. Text that no rule matches is returned unchanged.
  std::string Normalize(absl::string_view text) const;

  // Returns the number of compiled rewrite rules in the table. Exposed for
  // diagnostics and testing.
  //
  // returns
  // - Number of `rule` entries parsed from the rule table.
  size_t RuleCount() const;

 private:
  // One `rule` record: a compiled pattern and its replacement template.
  struct Rule {
    std::unique_ptr<RE2> pattern;
    std::string replacement;
  };

  // Rewrites every match of a single rule. Factored out of Normalize so that
  // the per-rule scan, which has to cope with empty matches, stays readable.
  std::string ApplyRule(const Rule& rule, absl::string_view text) const;

  // Appends `replacement` to `out`, resolving \N, #N, and @N against `groups`.
  void ExpandReplacement(absl::string_view replacement,
                         absl::Span<const absl::string_view> groups,
                         std::string& out) const;

  // Expands a run of ASCII digits (`digits`) as a cardinal integer using
  // `digit_glyphs_` and `units_`, appending the result to `out`.
  void ExpandCardinal(absl::string_view digits, std::string& out) const;

  std::vector<Rule> rules_;
  // Glyph for each ASCII digit, indexed by value. Empty when the table defines
  // no digits, which is only valid if no rule uses a `#N` or `@N` expansion.
  std::array<std::string, 10> digit_glyphs_;
  // Place-value units (10, 100, 1000, 10^4, 10^8) for `@N` cardinal expansion.
  std::array<std::string, 5> units_;
  bool has_units_ = false;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_NORMALIZER_H_
