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

#include "omni/tts/kokoro/common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/tts/kokoro/kokoro_io_types.h"

namespace litert::omni::tts::kokoro {

namespace {

template <typename Predicate>
size_t FindLastBreak(absl::Span<const int> ids, size_t earliest, size_t end,
                     Predicate is_break) {
  for (size_t pos = end; pos > earliest; --pos) {
    if (is_break(ids[pos - 1])) return pos;
  }
  return 0;
}

bool IsPunctuationToken(int token_id) {
  return token_id >= kMinPunctuationTokenId &&
         token_id <= kMaxPunctuationTokenId;
}

bool IsSpaceToken(int token_id) {
  return token_id == kSpaceTokenId;
}

}  // namespace

std::vector<TokenSlice> SliceTokenIds(absl::Span<const int> token_ids,
                                      int max_capacity) {
  if (token_ids.size() <= static_cast<size_t>(max_capacity)) {
    return {TokenSlice{std::vector<int>(token_ids.begin(), token_ids.end()),
                       SliceJoin::kChunkBoundary, SliceJoin::kChunkBoundary}};
  }

  std::vector<TokenSlice> slices;
  size_t start = 0;
  SliceJoin join_before = SliceJoin::kChunkBoundary;
  while (start < token_ids.size()) {
    size_t end = std::min(start + max_capacity, token_ids.size());
    SliceJoin join_after = SliceJoin::kChunkBoundary;
    if (end < token_ids.size()) {
      // Search backwards for a break, but no further than half the capacity so
      // a slice never becomes uselessly short.
      const size_t earliest = start + (max_capacity / kSplitCapacityDivisor);

      // Prefer punctuation (';', ':', ',', '.', '!', '?'). Each slice is
      // synthesized as its own utterance, so the join carries a pause no
      // matter where it lands; putting it on punctuation makes that pause one
      // the text asked for rather than a stutter mid-phrase.
      size_t split =
          FindLastBreak(token_ids, earliest, end, IsPunctuationToken);
      if (split != 0) {
        join_after = SliceJoin::kPunctuation;
      } else {
        // Otherwise fall back to a word boundary (space).
        split = FindLastBreak(token_ids, earliest, end, IsSpaceToken);
        join_after = SliceJoin::kWordBoundary;
      }

      if (split > earliest) {
        end = split;
      } else {
        // No usable boundary at all; the cut lands mid-word.
        join_after = SliceJoin::kWordBoundary;
      }
    }

    std::vector<int> slice;
    slice.reserve(std::max<size_t>(2, end - start));
    slice.push_back(kBosTokenId);
    if (token_ids[start] == kBosTokenId) {
      ++start;
    }
    slice.insert(slice.end(), token_ids.begin() + start,
                 token_ids.begin() + end);
    if (slice.back() != kEosTokenId) {
      slice.push_back(kEosTokenId);
    }
    slices.push_back(TokenSlice{std::move(slice), join_before, join_after});
    join_before = join_after;
    start = end;
  }
  return slices;
}

int HeadKeepSamples(SliceJoin join) {
  return join == SliceJoin::kChunkBoundary ? kKeepEdgeIntact
                                           : kSliceJoinMarginSamples;
}

int TailKeepSamples(SliceJoin join) {
  switch (join) {
    case SliceJoin::kChunkBoundary:
      return kKeepEdgeIntact;
    case SliceJoin::kWordBoundary:
      // Nothing in the text asks for a pause here, so keep only a decaying
      // tail.
      return kSliceJoinMarginSamples;
    case SliceJoin::kPunctuation:
      // The text does ask for a pause, but only a clause-length one, not the
      // full-stop silence the model puts at the end of an utterance.
      return kSlicePunctuationPauseSamples;
  }
  return kKeepEdgeIntact;
}

void TrimSliceJoinSilence(std::vector<float>& pcm, int head_keep_samples,
                          int tail_keep_samples) {
  const bool trim_head = head_keep_samples != kKeepEdgeIntact;
  const bool trim_tail = tail_keep_samples != kKeepEdgeIntact;
  if (pcm.empty() || (!trim_head && !trim_tail)) return;

  float peak = 0.0f;
  for (const float sample : pcm) peak = std::max(peak, std::abs(sample));
  if (peak <= 0.0f) {
    // Nothing but digital silence; at an inner join it contributes only delay.
    if (trim_head && trim_tail) {
      pcm.resize(std::min<size_t>(pcm.size(),
                                  static_cast<size_t>(head_keep_samples) +
                                      static_cast<size_t>(tail_keep_samples)));
    }
    return;
  }
  // The peak sample itself always clears this threshold, so the two scans below
  // are guaranteed to leave `begin < end`.
  const float threshold = peak * kSliceJoinSilenceRatio;

  size_t begin = 0;
  size_t end = pcm.size();
  if (trim_head) {
    while (begin < end && std::abs(pcm[begin]) < threshold) ++begin;
    begin -= std::min<size_t>(begin, head_keep_samples);
  }
  if (trim_tail) {
    while (end > begin && std::abs(pcm[end - 1]) < threshold) --end;
    end = std::min(pcm.size(), end + tail_keep_samples);
  }
  if (begin == 0 && end == pcm.size()) return;

  // Erase the tail first so the head offset stays valid.
  pcm.erase(pcm.begin() + end, pcm.end());
  pcm.erase(pcm.begin(), pcm.begin() + begin);
}

void TrimSliceJoinSilence(std::vector<float>& pcm, SliceJoin join_before,
                          SliceJoin join_after) {
  TrimSliceJoinSilence(pcm, HeadKeepSamples(join_before),
                       TailKeepSamples(join_after));
}

absl::StatusOr<std::vector<float>> LoadVoiceEmbedding(
    absl::string_view model_dir, absl::string_view voice_identifier) {
  std::vector<float> embed(kVoicePackSize, 0.0f);

  std::vector<std::filesystem::path> candidate_paths;
  std::filesystem::path base_path = std::string(model_dir);
  if (!voice_identifier.empty()) {
    std::string voice_str(voice_identifier);
    candidate_paths.push_back(voice_str);
    candidate_paths.push_back((base_path / voice_str));
    candidate_paths.push_back((base_path / "voices" / voice_str));
    candidate_paths.push_back(
        (base_path / "voices" / absl::StrCat(voice_identifier, ".bin")));
  }
  candidate_paths.push_back((base_path / "voices" / "af_heart"));
  candidate_paths.push_back((base_path / "voices" / "af_heart.bin"));

  for (const auto& path : candidate_paths) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(embed.data()),
                kVoicePackSize * sizeof(float));
      if (file.gcount() ==
          static_cast<std::streamsize>(kVoicePackSize * sizeof(float))) {
        ABSL_LOG(INFO) << "Loaded voice pack embedding (" << kVoicePackSize
                       << " floats) from: " << path;
        return embed;
      }
    }
  }

  return absl::NotFoundError(
      absl::StrCat("Voice embedding '", voice_identifier,
                   "' not found or incomplete in model_dir: ", model_dir));
}

}  // namespace litert::omni::tts::kokoro
