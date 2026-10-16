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

#include "omni/text2speech/kokoro/mecab_lattice.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/cjk_lattice.h"

namespace litert::omni::tts::kokoro {
namespace {

// The sections of `ja-lexicon` that segmentation reads. The pruned feature
// blobs are not among them: the front end reads those off the winning path
// afterwards, and the packer is free to rewrite them without moving a word
// boundary.
constexpr absl::string_view kDictionaryIndexSection = "sys-da";
constexpr absl::string_view kDictionarySection = "sys-tok";
constexpr absl::string_view kDictionaryFeatureSection = "sys-feat";
constexpr absl::string_view kConnectionSection = "matrix";
constexpr absl::string_view kCharacterSection = "char";
constexpr absl::string_view kUnknownIndexSection = "unk-da";
constexpr absl::string_view kUnknownSection = "unk-tok";
constexpr absl::string_view kUnknownFeatureSection = "unk-feat";

// How many characters MeCab will group into one guessed word. Beyond this it
// declines to guess at all rather than proposing an implausibly long word.
constexpr int kMaxGroupingSize = 24;

}  // namespace

MecabCostModel::MecabCostModel(MecabDoubleArray dictionary_index,
                               MecabTokenTable dictionary,
                               MecabTokenTable unknown,
                               MecabConnectionMatrix connection,
                               MecabCharProperty characters,
                               std::vector<UnknownEntries> unknown_by_category)
    : dictionary_index_(dictionary_index),
      dictionary_(dictionary),
      unknown_(unknown),
      connection_(connection),
      characters_(characters),
      unknown_by_category_(std::move(unknown_by_category)),
      space_(characters_.Info(' ')) {}

absl::StatusOr<MecabCostModel> MecabCostModel::Create(const CjkBlob& blob) {
  ABSL_ASSIGN_OR_RETURN(const absl::string_view dictionary_index_section,
                        blob.Section(kDictionaryIndexSection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view dictionary_section,
                        blob.Section(kDictionarySection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view dictionary_features,
                        blob.Section(kDictionaryFeatureSection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view connection_section,
                        blob.Section(kConnectionSection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view character_section,
                        blob.Section(kCharacterSection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view unknown_index_section,
                        blob.Section(kUnknownIndexSection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view unknown_section,
                        blob.Section(kUnknownSection));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view unknown_features,
                        blob.Section(kUnknownFeatureSection));

  ABSL_ASSIGN_OR_RETURN(const MecabDoubleArray dictionary_index,
                        MecabDoubleArray::Create(dictionary_index_section));
  ABSL_ASSIGN_OR_RETURN(
      const MecabTokenTable dictionary,
      MecabTokenTable::Create(dictionary_section, dictionary_features));
  ABSL_ASSIGN_OR_RETURN(const MecabDoubleArray unknown_index,
                        MecabDoubleArray::Create(unknown_index_section));
  ABSL_ASSIGN_OR_RETURN(
      const MecabTokenTable unknown,
      MecabTokenTable::Create(unknown_section, unknown_features));
  ABSL_ASSIGN_OR_RETURN(const MecabConnectionMatrix connection,
                        MecabConnectionMatrix::Create(connection_section));
  ABSL_ASSIGN_OR_RETURN(const MecabCharProperty characters,
                        MecabCharProperty::Create(character_section));

  // The unknown-word dictionary is keyed by category name. Resolving all of
  // them now turns every later guess into an array index, and turns a
  // dictionary missing a category into a load failure rather than a word that
  // silently cannot be guessed.
  std::vector<MecabCostModel::UnknownEntries> unknown_by_category(
      characters.CategoryCount());
  for (int category = 0; category < characters.CategoryCount(); ++category) {
    const absl::string_view name = characters.CategoryName(category);
    bool found = false;
    for (const MecabDoubleArray::Match& match :
         unknown_index.CommonPrefixSearch(name)) {
      if (match.length != static_cast<int>(name.size())) continue;
      unknown_by_category[category] = {match.token_index, match.token_count};
      found = true;
    }
    if (!found) {
      return absl::InvalidArgumentError(absl::StrCat(
          "the unknown-word dictionary has no entry for character category '",
          name, "'"));
    }
  }

  return MecabCostModel(dictionary_index, dictionary, unknown, connection,
                        characters, std::move(unknown_by_category));
}

double MecabCostModel::ConnectionCost(int right_id, int left_id) const {
  return connection_.Cost(right_id, left_id);
}

MecabToken MecabCostModel::Token(const LatticeNode& node) const {
  if (node.token_index < 0) return MecabToken();
  return node.is_unknown ? unknown_.Token(node.token_index)
                         : dictionary_.Token(node.token_index);
}

const MecabCharProperty& MecabCostModel::Characters() const {
  return characters_;
}

MecabCostModel::CategoryRun MecabCostModel::SeekPastCategory(
    absl::string_view text, int offset,
    MecabCharProperty::CharInfo info) const {
  const int length = static_cast<int>(text.size());
  CategoryRun run{.end = offset, .character_count = 0};
  while (run.end < length) {
    const MecabCharProperty::DecodedChar next =
        characters_.InfoAt(text, run.end);
    // A run continues while each character shares a category with the one
    // before it, not with the one the run started on.
    if ((info.types & next.info.types) == 0) break;
    run.end += next.length;
    ++run.character_count;
    info = next.info;
  }
  return run;
}

void MecabCostModel::AppendDictionaryMatches(
    absl::string_view text, int start, int surface_start,
    std::vector<LatticeNode>& candidates) const {
  for (const MecabDoubleArray::Match& match :
       dictionary_index_.CommonPrefixSearch(text.substr(surface_start))) {
    // One surface form can stand for several words -- different readings, or
    // the same reading in different parts of speech -- and the search has to
    // see all of them, because which one is cheapest depends on the
    // neighbours.
    for (int i = 0; i < match.token_count; ++i) {
      const int index = match.token_index + i;
      if (index >= dictionary_.Size()) break;
      const MecabToken token = dictionary_.Token(index);
      candidates.push_back({
          .start = start,
          .end = surface_start + match.length,
          .surface = text.substr(surface_start, match.length),
          .left_id = token.left_id,
          .right_id = token.right_id,
          .word_cost = static_cast<double>(token.word_cost),
          .token_index = index,
          .is_unknown = false,
      });
    }
  }
}

void MecabCostModel::AppendUnknown(absl::string_view text, int start,
                                   int surface_start, int end, int category,
                                   std::vector<LatticeNode>& candidates) const {
  if (category < 0 ||
      category >= static_cast<int>(unknown_by_category_.size())) {
    return;
  }
  const UnknownEntries entries = unknown_by_category_[category];
  for (int i = 0; i < entries.count; ++i) {
    const int index = entries.first + i;
    if (index >= unknown_.Size()) break;
    const MecabToken token = unknown_.Token(index);
    candidates.push_back({
        .start = start,
        .end = end,
        .surface = text.substr(surface_start, end - surface_start),
        .left_id = token.left_id,
        .right_id = token.right_id,
        .word_cost = static_cast<double>(token.word_cost),
        .token_index = index,
        .is_unknown = true,
    });
  }
}

void MecabCostModel::CandidatesAt(absl::string_view text, int start,
                                  std::vector<LatticeNode>& candidates) const {
  const int length = static_cast<int>(text.size());
  const size_t before = candidates.size();

  // Whitespace is not a word. Skip it and hand the bytes to whatever follows:
  // `start` stays the offset these nodes consume from, and `surface_start` is
  // where the word itself begins.
  const int surface_start = SeekPastCategory(text, start, space_).end;
  if (surface_start >= length) {
    // Whitespace all the way to the end. MeCab yields nothing here, but a
    // lattice that cannot reach the end of its input has no answer at all, so
    // the run becomes a node of its own. The front end drops it along with the
    // other whitespace tokens MeCab does emit, such as the ideographic space.
    AppendUnknown(text, start, start, length, space_.default_type, candidates);
    return;
  }

  const MecabCharProperty::DecodedChar first =
      characters_.InfoAt(text, surface_start);
  const MecabCharProperty::CharInfo& info = first.info;

  AppendDictionaryMatches(text, start, surface_start, candidates);
  // Categories that set `invoke` are guessed at even where the dictionary has
  // an answer. That is what stops a long run of digits or katakana being cut
  // into whichever short entries happen to be in the dictionary.
  if (candidates.size() > before && !info.invoke) return;

  // First the whole run of the category, if it is not implausibly long.
  int group_end = -1;
  if (info.group) {
    const CategoryRun run =
        SeekPastCategory(text, surface_start + first.length, info);
    group_end = run.end;
    if (run.character_count <= kMaxGroupingSize) {
      AppendUnknown(text, start, surface_start, group_end, info.default_type,
                    candidates);
    }
  }

  // Then the same run cut to each length the category allows, skipping the cut
  // the group above already covers.
  int end = surface_start + first.length;
  for (int i = 1; i <= info.length; ++i) {
    if (end > length) break;
    if (end == group_end) continue;
    AppendUnknown(text, start, surface_start, end, info.default_type,
                  candidates);
    if (end >= length) break;
    const MecabCharProperty::DecodedChar next = characters_.InfoAt(text, end);
    if ((info.types & next.info.types) == 0) break;
    end += next.length;
  }

  // A category that neither groups nor guesses by length still has to produce
  // something, or the sentence has no path through it.
  if (candidates.size() == before) {
    AppendUnknown(text, start, surface_start, end, info.default_type,
                  candidates);
  }
}

}  // namespace litert::omni::tts::kokoro
