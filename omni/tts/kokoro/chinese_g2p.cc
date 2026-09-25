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

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required by std::filesystem.
#include <utility>
#include <vector>

#include "cppjieba/DictTrie.hpp"
#include "cppjieba/HMMModel.hpp"
#include "cppjieba/MixSegment.hpp"
#include "cppjieba/Trie.hpp"
#include "cppjieba/Unicode.hpp"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/tts/kokoro/cjk_blob.h"

namespace litert::omni::tts::kokoro {
namespace {

constexpr int kHmmStates = 4;

uint32_t ReadLe32(const char* ptr) {
  const auto* u = reinterpret_cast<const uint8_t*>(ptr);
  return static_cast<uint32_t>(u[0]) | (static_cast<uint32_t>(u[1]) << 8) |
         (static_cast<uint32_t>(u[2]) << 16) |
         (static_cast<uint32_t>(u[3]) << 24);
}

double ReadLeDouble(const char* ptr) {
  const auto* u = reinterpret_cast<const uint8_t*>(ptr);
  uint64_t bits = 0;
  for (int i = 0; i < 8; ++i) {
    bits |= static_cast<uint64_t>(u[i]) << (8 * i);
  }
  double val = 0.0;
  std::memcpy(&val, &bits, sizeof(double));
  return val;
}

int Utf8CharLength(absl::string_view text, int offset) {
  const auto lead = static_cast<uint8_t>(text[offset]);
  const int remaining = static_cast<int>(text.size()) - offset;
  if (lead < 0x80) return 1;
  if ((lead & 0xE0) == 0xC0 && remaining >= 2) return 2;
  if ((lead & 0xF0) == 0xE0 && remaining >= 3) return 3;
  if ((lead & 0xF8) == 0xF0 && remaining >= 4) return 4;
  return 1;
}

uint32_t DecodeUtf8CodePoint(absl::string_view text, int offset, int& length) {
  const int len = Utf8CharLength(text, offset);
  length = len;
  const auto* u = reinterpret_cast<const uint8_t*>(text.data() + offset);
  if (len == 1) return u[0];
  if (len == 2) {
    return ((static_cast<uint32_t>(u[0]) & 0x1F) << 6) |
           (static_cast<uint32_t>(u[1]) & 0x3F);
  }
  if (len == 3) {
    return ((static_cast<uint32_t>(u[0]) & 0x0F) << 12) |
           ((static_cast<uint32_t>(u[1]) & 0x3F) << 6) |
           (static_cast<uint32_t>(u[2]) & 0x3F);
  }
  return ((static_cast<uint32_t>(u[0]) & 0x07) << 18) |
         ((static_cast<uint32_t>(u[1]) & 0x3F) << 12) |
         ((static_cast<uint32_t>(u[2]) & 0x3F) << 6) |
         (static_cast<uint32_t>(u[3]) & 0x3F);
}

// Matches `jieba.re_han_default`:
// `re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&._%-]+)")`.
bool IsJiebaHanCodePoint(uint32_t cp) {
  if (cp >= 0x4E00 && cp <= 0x9FD5) return true;
  if (cp >= 'a' && cp <= 'z') return true;
  if (cp >= 'A' && cp <= 'Z') return true;
  if (cp >= '0' && cp <= '9') return true;
  return cp == '+' || cp == '#' || cp == '&' || cp == '.' || cp == '_' ||
         cp == '%' || cp == '-';
}

std::vector<int> Utf8CharEnds(absl::string_view text) {
  std::vector<int> ends;
  int offset = 0;
  while (offset < static_cast<int>(text.size())) {
    offset += Utf8CharLength(text, offset);
    ends.push_back(offset);
  }
  return ends;
}

absl::StatusOr<std::unique_ptr<cppjieba::DictTrie>> CreateDictTrie(
    const CjkBlob& blob) {
  ABSL_ASSIGN_OR_RETURN(const absl::string_view w_idx, blob.Section("w-idx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view w_key, blob.Section("w-key"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view w_frq, blob.Section("w-frq"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view w_tot, blob.Section("w-tot"));
  ABSL_ASSIGN_OR_RETURN(const SortedStringIndex words,
                        SortedStringIndex::Create(w_idx, w_key));
  if (w_frq.size() != static_cast<size_t>(words.Size()) * 4) {
    return absl::InvalidArgumentError(
        "zh-lexicon w-frq size does not match w-idx entry count");
  }
  if (w_tot.size() != sizeof(double)) {
    return absl::InvalidArgumentError("zh-lexicon w-tot must be 8 bytes");
  }
  const double log_total = ReadLeDouble(w_tot.data());

  cppjieba::DictTrie::PrecomputedDict precomputed;
  precomputed.freq_sum = std::exp(log_total);
  double max_weight = -std::numeric_limits<double>::infinity();
  precomputed.node_infos.reserve(words.Size());
  for (int i = 0; i < words.Size(); ++i) {
    const uint32_t freq = ReadLe32(w_frq.data() + static_cast<size_t>(i) * 4);
    if (freq == 0) continue;
    cppjieba::DictUnit unit;
    if (!cppjieba::DecodeUTF8RunesInString(std::string(words.Key(i)),
                                           unit.word) ||
        unit.word.empty()) {
      continue;
    }
    unit.weight = std::log(static_cast<double>(freq)) - log_total;
    if (unit.weight > max_weight) max_weight = unit.weight;
    precomputed.node_infos.push_back(std::move(unit));
  }
  if (precomputed.node_infos.empty()) {
    return absl::InvalidArgumentError(
        "zh-lexicon has no positive-frequency words");
  }
  precomputed.min_weight = -log_total;
  precomputed.max_weight = max_weight;
  precomputed.median_weight = 0.5 * (precomputed.min_weight + max_weight);
  return std::make_unique<cppjieba::DictTrie>(precomputed);
}

void EncodeUtf8Rune(uint32_t cp, std::string& out) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Unpacks the HMM parameters from the `zh-lexicon` blob into `cache_dir` in the
// `hmm_model.utf8` text format that `cppjieba::HMMModel` reads from disk,
// mirroring `UnpackEspeakDataFromLitertLm` in `espeak_assets.cc`.
absl::StatusOr<std::string> UnpackHmmModelToCacheDir(
    const CjkBlob& blob, absl::string_view cache_dir) {
  // zh-lexicon stores HMM states in (B=0, M=1, E=2, S=3) order, while
  // cppjieba::HMMModel reads states in (B=0, E=1, M=2, S=3) order.
  static constexpr int kCppJiebaToBlob[kHmmStates] = {0, 2, 1, 3};

  ABSL_ASSIGN_OR_RETURN(const absl::string_view hmm_st, blob.Section("hmm-st"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view hmm_tr, blob.Section("hmm-tr"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view hmm_en, blob.Section("hmm-en"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view hmm_ec, blob.Section("hmm-ec"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view hmm_ep, blob.Section("hmm-ep"));

  if (hmm_st.size() != kHmmStates * sizeof(double)) {
    return absl::InvalidArgumentError("zh-lexicon hmm-st must be 32 bytes");
  }
  if (hmm_tr.size() != kHmmStates * kHmmStates * sizeof(double)) {
    return absl::InvalidArgumentError("zh-lexicon hmm-tr must be 128 bytes");
  }
  if (hmm_en.size() != kHmmStates * sizeof(uint32_t)) {
    return absl::InvalidArgumentError("zh-lexicon hmm-en must be 16 bytes");
  }

  size_t total_entries = 0;
  uint32_t counts[kHmmStates] = {};
  size_t state_offsets[kHmmStates] = {};
  for (int s = 0; s < kHmmStates; ++s) {
    state_offsets[s] = total_entries;
    counts[s] = ReadLe32(hmm_en.data() + s * sizeof(uint32_t));
    total_entries += counts[s];
  }
  if (hmm_ec.size() != total_entries * sizeof(uint32_t) ||
      hmm_ep.size() != total_entries * sizeof(double)) {
    return absl::InvalidArgumentError(
        "zh-lexicon hmm-ec / hmm-ep size does not match hmm-en");
  }

  std::string content;
  for (int dst_s = 0; dst_s < kHmmStates; ++dst_s) {
    const int src_s = kCppJiebaToBlob[dst_s];
    const double p = ReadLeDouble(hmm_st.data() + src_s * sizeof(double));
    absl::StrAppend(&content, dst_s == 0 ? "" : " ", p);
  }
  content.push_back('\n');

  for (int dst_s = 0; dst_s < kHmmStates; ++dst_s) {
    const int src_s = kCppJiebaToBlob[dst_s];
    for (int dst_next = 0; dst_next < kHmmStates; ++dst_next) {
      const int src_next = kCppJiebaToBlob[dst_next];
      const double p = ReadLeDouble(
          hmm_tr.data() + (src_s * kHmmStates + src_next) * sizeof(double));
      absl::StrAppend(&content, dst_next == 0 ? "" : " ", p);
    }
    content.push_back('\n');
  }

  for (int dst_s = 0; dst_s < kHmmStates; ++dst_s) {
    const int src_s = kCppJiebaToBlob[dst_s];
    const uint32_t count = counts[src_s];
    const size_t base = state_offsets[src_s];
    if (count == 0) {
      absl::StrAppend(&content, "一:", cppjieba::MIN_DOUBLE, "\n");
      continue;
    }
    for (uint32_t i = 0; i < count; ++i) {
      if (i > 0) content.push_back(',');
      const uint32_t rune =
          ReadLe32(hmm_ec.data() + (base + i) * sizeof(uint32_t));
      const double prob =
          ReadLeDouble(hmm_ep.data() + (base + i) * sizeof(double));
      EncodeUtf8Rune(rune, content);
      absl::StrAppend(&content, ":", prob);
    }
    content.push_back('\n');
  }

  std::error_code ec;
  std::filesystem::path base_dir =
      cache_dir.empty() ? std::filesystem::temp_directory_path(ec)
                        : std::filesystem::path(std::string(cache_dir));
  if (ec) {
    return absl::InternalError(absl::StrCat(
        "Failed to resolve cache_dir for cppjieba HMM model: ", ec.message()));
  }
  std::filesystem::create_directories(base_dir, ec);
  std::filesystem::path dest_path = base_dir / "kokoro_zh_hmm_model.utf8";
  if (std::filesystem::exists(dest_path, ec) &&
      std::filesystem::file_size(dest_path, ec) == content.size()) {
    return dest_path.string();
  }

  static std::atomic<uint64_t> staging_counter{0};
  const uint64_t unique_id =
      staging_counter.fetch_add(1, std::memory_order_relaxed);
  const std::string staging_filename =
      absl::StrCat("kokoro_zh_hmm_model.utf8.unpacking.",
                   reinterpret_cast<uintptr_t>(&unique_id), ".", unique_id);
  std::filesystem::path staging_path = base_dir / staging_filename;
  {
    std::ofstream file(staging_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
      // `cache_dir` may point into a read-only runfiles tree in unit tests;
      // fall back to the writable system temporary directory.
      base_dir = std::filesystem::temp_directory_path(ec);
      dest_path = base_dir / "kokoro_zh_hmm_model.utf8";
      if (std::filesystem::exists(dest_path, ec) &&
          std::filesystem::file_size(dest_path, ec) == content.size()) {
        return dest_path.string();
      }
      staging_path = base_dir / staging_filename;
      file.open(staging_path, std::ios::binary | std::ios::trunc);
    }
    if (!file.is_open()) {
      return absl::InternalError(absl::StrCat(
          "Failed to write cppjieba HMM model to: ", staging_path.string()));
    }
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!file.good()) {
      return absl::InternalError(absl::StrCat(
          "Failed to flush cppjieba HMM model to: ", staging_path.string()));
    }
  }
  std::filesystem::rename(staging_path, dest_path, ec);
  if (ec) {
    std::filesystem::remove(staging_path, ec);
    if (!std::filesystem::exists(dest_path, ec)) {
      return absl::InternalError(absl::StrCat(
          "Failed to rename cppjieba HMM model to: ", dest_path.string()));
    }
  }
  return dest_path.string();
}

}  // namespace

ChineseG2p::ChineseG2p(std::unique_ptr<cppjieba::DictTrie> dict_trie,
                       std::unique_ptr<cppjieba::HMMModel> hmm_model,
                       std::unique_ptr<cppjieba::MixSegment> segmenter,
                       SortedStringIndex phrases,
                       StringValueTable phrase_values,
                       SortedStringIndex characters,
                       StringValueTable character_values)
    : dict_trie_(std::move(dict_trie)),
      hmm_model_(std::move(hmm_model)),
      segmenter_(std::move(segmenter)),
      phrases_(phrases),
      phrase_values_(phrase_values),
      characters_(characters),
      character_values_(character_values) {}

absl::StatusOr<ChineseG2p> ChineseG2p::Create(const CjkBlob& blob,
                                              absl::string_view cache_dir) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<cppjieba::DictTrie> dict_trie,
                        CreateDictTrie(blob));
  ABSL_ASSIGN_OR_RETURN(const std::string hmm_model_path,
                        UnpackHmmModelToCacheDir(blob, cache_dir));
  auto hmm_model = std::make_unique<cppjieba::HMMModel>(hmm_model_path);
  auto segmenter =
      std::make_unique<cppjieba::MixSegment>(dict_trie.get(), hmm_model.get());
  segmenter->ResetSeparators("");

  ABSL_ASSIGN_OR_RETURN(const absl::string_view p_idx, blob.Section("p-idx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view p_key, blob.Section("p-key"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view p_vidx, blob.Section("p-vidx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view p_val, blob.Section("p-val"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view c_idx, blob.Section("c-idx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view c_key, blob.Section("c-key"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view c_vidx, blob.Section("c-vidx"));
  ABSL_ASSIGN_OR_RETURN(const absl::string_view c_val, blob.Section("c-val"));

  ABSL_ASSIGN_OR_RETURN(const SortedStringIndex phrases,
                        SortedStringIndex::Create(p_idx, p_key));
  ABSL_ASSIGN_OR_RETURN(const StringValueTable phrase_values,
                        StringValueTable::Create(p_vidx, p_val));
  if (phrases.Size() != phrase_values.Size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "zh-lexicon phrase key count (", phrases.Size(),
        ") does not match value count (", phrase_values.Size(), ")"));
  }

  ABSL_ASSIGN_OR_RETURN(const SortedStringIndex characters,
                        SortedStringIndex::Create(c_idx, c_key));
  ABSL_ASSIGN_OR_RETURN(const StringValueTable character_values,
                        StringValueTable::Create(c_vidx, c_val));
  if (characters.Size() != character_values.Size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "zh-lexicon character key count (", characters.Size(),
        ") does not match value count (", character_values.Size(), ")"));
  }

  return ChineseG2p(std::move(dict_trie), std::move(hmm_model),
                    std::move(segmenter), phrases, phrase_values, characters,
                    character_values);
}

absl::StatusOr<std::vector<absl::string_view>> ChineseG2p::Segment(
    absl::string_view text) const {
  std::vector<absl::string_view> out;
  const int n = static_cast<int>(text.size());
  int offset = 0;
  while (offset < n) {
    int char_len = 0;
    const uint32_t cp = DecodeUtf8CodePoint(text, offset, char_len);
    const bool is_han = IsJiebaHanCodePoint(cp);
    int run_end = offset + char_len;
    while (run_end < n) {
      int next_len = 0;
      const uint32_t next_cp = DecodeUtf8CodePoint(text, run_end, next_len);
      if (IsJiebaHanCodePoint(next_cp) != is_han) break;
      run_end += next_len;
    }
    const absl::string_view block = text.substr(offset, run_end - offset);
    if (is_han) {
      std::vector<cppjieba::Word> words;
      segmenter_->Cut(std::string(block), words, /*hmm=*/true);
      for (const cppjieba::Word& w : words) {
        out.push_back(block.substr(w.offset, w.word.size()));
      }
    } else if (!block.empty()) {
      out.push_back(block);
    }
    offset = run_end;
  }
  return out;
}

void ChineseG2p::MmsegCut(absl::string_view word,
                          std::vector<absl::string_view>& pieces) const {
  absl::string_view remain = word;
  while (!remain.empty()) {
    const std::vector<int> char_ends = Utf8CharEnds(remain);
    int last_valid_bytes = 0;
    bool stopped_prefix = false;
    for (int end_bytes : char_ends) {
      const absl::string_view candidate = remain.substr(0, end_bytes);
      if (phrases_.HasPrefix(candidate)) {
        if (phrases_.Find(candidate) >= 0) {
          last_valid_bytes = end_bytes;
        }
        continue;
      }
      stopped_prefix = true;
      if (last_valid_bytes > 0) {
        pieces.push_back(remain.substr(0, last_valid_bytes));
        remain.remove_prefix(last_valid_bytes);
      } else {
        pieces.push_back(remain.substr(0, char_ends[0]));
        remain.remove_prefix(char_ends[0]);
      }
      break;
    }
    if (stopped_prefix) continue;

    if (last_valid_bytes > 0) {
      pieces.push_back(remain.substr(0, last_valid_bytes));
      remain.remove_prefix(last_valid_bytes);
    } else {
      pieces.push_back(remain.substr(0, char_ends[0]));
      remain.remove_prefix(char_ends[0]);
    }
  }
}

bool ChineseG2p::WordToIpa(absl::string_view word, std::string& out) const {
  std::vector<absl::string_view> pieces;
  MmsegCut(word, pieces);
  std::string buffer;
  for (absl::string_view piece : pieces) {
    const int phrase_pos = phrases_.Find(piece);
    if (phrase_pos >= 0) {
      buffer.append(phrase_values_.Value(phrase_pos));
      continue;
    }
    int offset = 0;
    while (offset < static_cast<int>(piece.size())) {
      const int char_len = Utf8CharLength(piece, offset);
      const absl::string_view ch = piece.substr(offset, char_len);
      const int char_pos = characters_.Find(ch);
      if (char_pos < 0) return false;
      buffer.append(character_values_.Value(char_pos));
      offset += char_len;
    }
  }
  out.append(buffer);
  return true;
}

absl::StatusOr<std::string> ChineseG2p::TextToIpa(
    absl::string_view text) const {
  ABSL_ASSIGN_OR_RETURN(const std::vector<absl::string_view> words,
                        Segment(text));
  std::vector<std::string> word_ipas;
  word_ipas.reserve(words.size());
  for (absl::string_view word : words) {
    std::string ipa;
    if (!WordToIpa(word, ipa)) {
      return absl::NotFoundError(
          absl::StrCat("zh-lexicon has no IPA entry for token '", word, "'"));
    }
    word_ipas.push_back(std::move(ipa));
  }
  return absl::StrJoin(word_ipas, " ");
}

}  // namespace litert::omni::tts::kokoro
