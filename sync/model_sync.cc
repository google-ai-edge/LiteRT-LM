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

#include "sync/model_sync.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define LITERT_LM_HAS_POSIX_CURL 0
#define LITERT_LM_HAS_WIN32_CURL 1
#elif !defined(__ANDROID__) && !defined(__EMSCRIPTEN__) && \
    (!defined(__APPLE__) || TARGET_OS_OSX)
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define LITERT_LM_HAS_POSIX_CURL 1
#define LITERT_LM_HAS_WIN32_CURL 0
#else
#define LITERT_LM_HAS_POSIX_CURL 0
#define LITERT_LM_HAS_WIN32_CURL 0
#endif

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "sync/sha256.h"

namespace litert::lm::sync {
namespace {

constexpr size_t kChunkSize = 1024 * 1024;  // 1 MB streaming chunks.
constexpr uint64_t kMaxHeaderBytes = 64 * 1024 * 1024;  // 64 MB upper bound

// A streambuf wrapper that forwards writes to an underlying stream while
// updating a Sha256Hasher incrementally.
class HashingStreambuf : public std::streambuf {
 public:
  HashingStreambuf(std::streambuf* dest, Sha256Hasher* hasher)
      : dest_(dest), hasher_(hasher) {}

 protected:
  int overflow(int ch) override {
    if (ch != traits_type::eof()) {
      char c = static_cast<char>(ch);
      if (hasher_) hasher_->Update(&c, 1);
      return dest_->sputc(c);
    }
    return ch;
  }

  std::streamsize xsputn(const char* s, std::streamsize count) override {
    if (count > 0) {
      if (hasher_) hasher_->Update(s, count);
      return dest_->sputn(s, count);
    }
    return 0;
  }

  int sync() override { return dest_->pubsync(); }

 private:
  std::streambuf* dest_;
  Sha256Hasher* hasher_;
};

struct SectionInfo {
  size_t index = 0;
  schema::AnySectionDataType data_type = schema::AnySectionDataType_NONE;
  std::string model_type;
  std::string sha256;
  uint64_t begin_offset = 0;
  uint64_t end_offset = 0;

  uint64_t size() const {
    return (end_offset >= begin_offset) ? (end_offset - begin_offset) : 0;
  }
};

absl::StatusOr<std::vector<SectionInfo>> ExtractSections(
    const schema::LiteRTLMMetaData* metadata) {
  if (!metadata || !metadata->section_metadata() ||
      !metadata->section_metadata()->objects()) {
    return std::vector<SectionInfo>{};
  }

  std::vector<SectionInfo> sections;
  const auto* objects = metadata->section_metadata()->objects();
  sections.reserve(objects->size());

  for (size_t i = 0; i < objects->size(); ++i) {
    const auto* obj = objects->Get(i);
    if (!obj) continue;

    SectionInfo info;
    info.index = i;
    info.data_type = obj->data_type();
    info.begin_offset = obj->begin_offset();
    info.end_offset = obj->end_offset();

    if (obj->items()) {
      for (size_t j = 0; j < obj->items()->size(); ++j) {
        const auto* item = obj->items()->Get(j);
        if (!item || !item->key()) continue;
        absl::string_view key = item->key()->string_view();

        if (key == "sha256" &&
            item->value_type() == schema::VData::VData_StringValue &&
            item->value()) {
          const auto* sv =
              static_cast<const schema::StringValue*>(item->value());
          if (sv && sv->value()) {
            info.sha256 = sv->value()->str();
          }
        } else if (key == "model_type" &&
                   item->value_type() == schema::VData::VData_StringValue &&
                   item->value()) {
          const auto* sv =
              static_cast<const schema::StringValue*>(item->value());
          if (sv && sv->value()) {
            info.model_type = sv->value()->str();
          }
        }
      }
    }
    sections.push_back(std::move(info));
  }
  return sections;
}

absl::StatusOr<std::string> FetchRemoteHeaderBytes(
    const RangeFetcher& fetcher, uint64_t initial_fetch_size) {
  if (initial_fetch_size < 32) {
    initial_fetch_size = 32;
  }

  std::ostringstream initial_stream(std::ios::binary);
  absl::Status status = fetcher(0, initial_fetch_size - 1, initial_stream);
  if (!status.ok()) {
    return absl::Status(status.code(),
                        absl::StrCat("Failed to fetch initial remote header: ",
                                     status.message()));
  }

  std::string buffer = initial_stream.str();
  if (buffer.size() < 32) {
    return absl::InvalidArgumentError(
        absl::StrCat("Remote header response is too short: ", buffer.size(),
                     " bytes (expected at least 32 bytes)."));
  }

  if (buffer.substr(0, 8) != "LITERTLM") {
    return absl::InvalidArgumentError(
        "Remote file does not start with LITERTLM magic bytes.");
  }

  uint64_t header_end_offset = 0;
  std::memcpy(&header_end_offset, buffer.data() + 24, sizeof(uint64_t));

  if (header_end_offset < 36 || header_end_offset > kMaxHeaderBytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid header_end_offset %d (must be between 36 and %d bytes).",
        header_end_offset, kMaxHeaderBytes));
  }

  if (buffer.size() < header_end_offset) {
    std::ostringstream remainder_stream(std::ios::binary);
    status = fetcher(buffer.size(), header_end_offset - 1, remainder_stream);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat("Failed to fetch remaining remote header bytes: ",
                       status.message()));
    }
    buffer.append(remainder_stream.str());
  }

  if (buffer.size() < header_end_offset) {
    return absl::InternalError(
        absl::StrFormat("Fetched %d bytes, but header requires %d bytes.",
                        buffer.size(), header_end_offset));
  }

  if (buffer.size() > header_end_offset) {
    buffer.resize(header_end_offset);
  }

  return buffer;
}

struct ParsedModelHeaders {
  std::string remote_header_bytes;
  std::string local_header_bytes;
  std::vector<SectionInfo> remote_sections;
  std::vector<SectionInfo> local_sections;
  bool local_exists = false;
  bool local_header_valid = false;
  uint64_t local_file_size = 0;
  std::string local_model_path;
};

bool IsValidSha256Hex(absl::string_view hex) {
  if (hex.size() != 64) return false;
  bool has_nonzero = false;
  for (char c : hex) {
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    if (c != '0') has_nonzero = true;
  }
  return has_nonzero;
}

const SectionInfo* FindReusableLocalSection(
    const SectionInfo& r_sec, const std::vector<SectionInfo>& local_sections) {
  if (!IsValidSha256Hex(r_sec.sha256)) {
    return nullptr;
  }
  for (const auto& l_sec : local_sections) {
    if (l_sec.size() == r_sec.size() &&
        absl::EqualsIgnoreCase(l_sec.sha256, r_sec.sha256)) {
      return &l_sec;
    }
  }
  return nullptr;
}

absl::Status VerifyFileSectionChecksums(
    absl::string_view file_path, const std::vector<SectionInfo>& sections) {
  std::ifstream in(std::string(file_path), std::ios::binary);
  if (!in.is_open()) {
    return absl::InternalError(absl::StrCat(
        "Failed to open file for checksum verification: ", file_path));
  }
  std::vector<char> buffer(kChunkSize);
  for (const auto& sec : sections) {
    if (sec.size() == 0 || !IsValidSha256Hex(sec.sha256)) continue;
    in.seekg(sec.begin_offset);
    if (!in) {
      return absl::DataLossError(absl::StrCat(
          "Failed to seek to section at offset ", sec.begin_offset));
    }
    uint64_t remaining = sec.size();
    Sha256Hasher hasher;
    while (remaining > 0) {
      size_t to_read = std::min<uint64_t>(remaining, buffer.size());
      in.read(buffer.data(), to_read);
      std::streamsize actual = in.gcount();
      if (actual <= 0) {
        return absl::DataLossError(absl::StrFormat(
            "Section payload truncated: expected %d bytes.", sec.size()));
      }
      hasher.Update(buffer.data(), actual);
      remaining -= actual;
    }
    std::string computed_sha = hasher.FinalHex();
    if (!absl::EqualsIgnoreCase(computed_sha, sec.sha256)) {
      return absl::DataLossError(absl::StrFormat(
          "Section SHA-256 verification failed: expected %s, got %s",
          sec.sha256, computed_sha));
    }
  }
  return absl::OkStatus();
}

bool IsModelUpToDate(const ParsedModelHeaders& parsed,
                     const SyncOptions& options) {
  if (!parsed.local_exists || !parsed.local_header_valid) {
    return false;
  }
  if (parsed.local_sections.size() != parsed.remote_sections.size()) {
    return false;
  }
  if (parsed.remote_sections.empty()) {
    return false;
  }

  // 1. Full header bytes comparison: ensures system metadata, prompt templates,
  // and section metadata (including names and backend constraints) match.
  if (parsed.local_header_bytes != parsed.remote_header_bytes) {
    return false;
  }

  // 2. Section comparison:
  for (size_t i = 0; i < parsed.remote_sections.size(); ++i) {
    const auto& l_sec = parsed.local_sections[i];
    const auto& r_sec = parsed.remote_sections[i];

    if (l_sec.data_type != r_sec.data_type ||
        l_sec.model_type != r_sec.model_type ||
        l_sec.begin_offset != r_sec.begin_offset ||
        l_sec.end_offset != r_sec.end_offset || l_sec.size() != r_sec.size() ||
        !IsValidSha256Hex(r_sec.sha256) ||
        !absl::EqualsIgnoreCase(l_sec.sha256, r_sec.sha256)) {
      return false;
    }
  }

  // 3. Check expected local file size against max end_offset if file size is
  // known:
  uint64_t max_remote_offset = parsed.remote_header_bytes.size();
  for (const auto& r_sec : parsed.remote_sections) {
    max_remote_offset = std::max(max_remote_offset, r_sec.end_offset);
  }
  if (parsed.local_file_size > 0 &&
      parsed.local_file_size < max_remote_offset) {
    return false;
  }

  // 4. If verify_checksums is enabled and local file path is available on disk,
  // verify each section payload matches expected sha256:
  if (options.verify_checksums && !parsed.local_model_path.empty()) {
    if (!VerifyFileSectionChecksums(parsed.local_model_path,
                                    parsed.local_sections)
             .ok()) {
      return false;
    }
  }

  return true;
}

absl::StatusOr<std::vector<SectionInfo>> ParseAndVerifyHeader(
    absl::string_view header_bytes) {
  if (header_bytes.size() < 36) {
    return absl::InvalidArgumentError(
        "Header bytes shorter than 36 bytes (minimum required for metadata).");
  }
  if (header_bytes.substr(0, 8) != "LITERTLM") {
    return absl::InvalidArgumentError(
        "Header does not start with LITERTLM magic bytes.");
  }
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(header_bytes.data() + 32),
      header_bytes.size() - 32);
  if (!schema::VerifyLiteRTLMMetaDataBuffer(verifier)) {
    return absl::InvalidArgumentError("Corrupt LiteRT-LM FlatBuffer metadata.");
  }

  schema::LitertlmHeader header;
  absl::Status status = schema::ReadHeaderFromLiteRTLM(
      const_cast<char*>(header_bytes.data()), header_bytes.size(), &header);
  if (!status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse LiteRT-LM header: ", status.message()));
  }
  if (header.metadata == nullptr) {
    return absl::InvalidArgumentError("Null metadata in LiteRT-LM header.");
  }

  auto sections_res = ExtractSections(header.metadata);
  if (!sections_res.ok()) {
    return sections_res.status();
  }
  std::vector<SectionInfo> sections = std::move(*sections_res);

  // Validate that section offsets do not overlap and do not precede header:
  std::vector<SectionInfo> sorted = sections;
  std::sort(sorted.begin(), sorted.end(),
            [](const SectionInfo& a, const SectionInfo& b) {
              return a.begin_offset < b.begin_offset;
            });
  uint64_t prev_end = header_bytes.size();
  for (const auto& sec : sorted) {
    if (sec.begin_offset < header_bytes.size() ||
        sec.end_offset < sec.begin_offset || sec.begin_offset < prev_end) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Malformed or overlapping section offset range: "
          "header_size=%d, begin=%d, end=%d, prev_end=%d",
          header_bytes.size(), sec.begin_offset, sec.end_offset, prev_end));
    }
    prev_end = sec.end_offset;
  }

  return sections;
}

absl::StatusOr<ParsedModelHeaders> ParseModelHeaders(
    absl::string_view local_model_path, const RangeFetcher& fetcher,
    uint64_t initial_header_fetch_bytes) {
  ParsedModelHeaders parsed;
  parsed.local_model_path = std::string(local_model_path);
  auto remote_header_res =
      FetchRemoteHeaderBytes(fetcher, initial_header_fetch_bytes);
  if (!remote_header_res.ok()) {
    return remote_header_res.status();
  }
  parsed.remote_header_bytes = std::move(*remote_header_res);

  auto remote_sec_res = ParseAndVerifyHeader(parsed.remote_header_bytes);
  if (!remote_sec_res.ok()) {
    return remote_sec_res.status();
  }
  parsed.remote_sections = std::move(*remote_sec_res);

  std::error_code exists_ec;
  parsed.local_exists =
      !local_model_path.empty() &&
      std::filesystem::exists(std::string(local_model_path), exists_ec) &&
      !exists_ec;
  if (parsed.local_exists) {
    std::error_code size_ec;
    parsed.local_file_size =
        std::filesystem::file_size(std::string(local_model_path), size_ec);
    if (size_ec) {
      parsed.local_file_size = 0;
    }

    std::ifstream local_file(std::string(local_model_path), std::ios::binary);
    if (local_file.is_open()) {
      std::string magic(8, '\0');
      local_file.read(magic.data(), 8);
      if (magic == "LITERTLM") {
        local_file.seekg(24);
        uint64_t local_header_end_offset = 0;
        local_file.read(reinterpret_cast<char*>(&local_header_end_offset),
                        sizeof(uint64_t));
        if (local_file && local_header_end_offset >= 36 &&
            local_header_end_offset <= kMaxHeaderBytes &&
            local_header_end_offset <= parsed.local_file_size) {
          parsed.local_header_bytes.resize(local_header_end_offset);
          local_file.seekg(0);
          local_file.read(parsed.local_header_bytes.data(),
                          local_header_end_offset);
          if (local_file) {
            auto local_sec_res =
                ParseAndVerifyHeader(parsed.local_header_bytes);
            if (local_sec_res.ok()) {
              parsed.local_sections = std::move(*local_sec_res);
              parsed.local_header_valid = true;
            }
          }
        }
      }
    }
  }

  return parsed;
}

absl::StatusOr<ParsedModelHeaders> ParseModelHeadersFromFetchers(
    const RangeFetcher& local_fetcher, const RangeFetcher& remote_fetcher,
    uint64_t initial_header_fetch_bytes) {
  ParsedModelHeaders parsed;
  auto remote_header_res =
      FetchRemoteHeaderBytes(remote_fetcher, initial_header_fetch_bytes);
  if (!remote_header_res.ok()) {
    return remote_header_res.status();
  }
  parsed.remote_header_bytes = std::move(*remote_header_res);

  auto remote_sec_res = ParseAndVerifyHeader(parsed.remote_header_bytes);
  if (!remote_sec_res.ok()) {
    return remote_sec_res.status();
  }
  parsed.remote_sections = std::move(*remote_sec_res);

  if (local_fetcher) {
    auto local_header_res =
        FetchRemoteHeaderBytes(local_fetcher, initial_header_fetch_bytes);
    if (local_header_res.ok()) {
      parsed.local_header_bytes = std::move(*local_header_res);
      auto local_sec_res = ParseAndVerifyHeader(parsed.local_header_bytes);
      if (local_sec_res.ok()) {
        parsed.local_sections = std::move(*local_sec_res);
        parsed.local_exists = true;
        parsed.local_header_valid = true;
      }
    }
  }

  return parsed;
}

SyncPlan BuildSyncPlan(const ParsedModelHeaders& parsed,
                       const SyncOptions& options) {
  SyncPlan plan;
  uint64_t max_remote_offset = parsed.remote_header_bytes.size();

  bool is_up_to_date = IsModelUpToDate(parsed, options);

  if (is_up_to_date) {
    plan.is_up_to_date = true;
    plan.use_full_download = false;
    plan.bytes_to_download = 0;
    plan.required_ranges.clear();
    plan.section_diffs.reserve(parsed.remote_sections.size());
    for (const auto& r_sec : parsed.remote_sections) {
      max_remote_offset = std::max(max_remote_offset, r_sec.end_offset);
      plan.bytes_to_reuse += r_sec.size();

      SectionDiff diff;
      diff.section_index = r_sec.index;
      diff.data_type = static_cast<int>(r_sec.data_type);
      diff.model_type = r_sec.model_type;
      diff.local_sha256 = r_sec.sha256;
      diff.remote_sha256 = r_sec.sha256;
      diff.size = r_sec.size();
      diff.begin_offset = r_sec.begin_offset;
      diff.end_offset = r_sec.end_offset;
      diff.action = SectionAction::kReuseLocal;
      plan.section_diffs.push_back(std::move(diff));
    }
    plan.total_remote_bytes = max_remote_offset;
    return plan;
  }

  plan.is_up_to_date = false;
  plan.bytes_to_download = parsed.remote_header_bytes.size();
  plan.section_diffs.reserve(parsed.remote_sections.size());

  for (const auto& r_sec : parsed.remote_sections) {
    max_remote_offset = std::max(max_remote_offset, r_sec.end_offset);

    const SectionInfo* local_match = nullptr;
    if (parsed.local_exists && parsed.local_header_valid) {
      local_match = FindReusableLocalSection(r_sec, parsed.local_sections);
    }

    SectionDiff diff;
    diff.section_index = r_sec.index;
    diff.data_type = static_cast<int>(r_sec.data_type);
    diff.model_type = r_sec.model_type;
    diff.remote_sha256 = r_sec.sha256;
    diff.size = r_sec.size();
    diff.begin_offset = r_sec.begin_offset;
    diff.end_offset = r_sec.end_offset;

    if (local_match != nullptr) {
      diff.local_sha256 = local_match->sha256;
      diff.action = SectionAction::kReuseLocal;
      plan.bytes_to_reuse += r_sec.size();
    } else {
      diff.action = SectionAction::kDownloadDiff;
      plan.bytes_to_download += r_sec.size();
      if (r_sec.size() > 0) {
        plan.required_ranges.push_back(
            ByteRange{r_sec.begin_offset, r_sec.end_offset - 1});
      }
      if (r_sec.index < parsed.local_sections.size()) {
        diff.local_sha256 = parsed.local_sections[r_sec.index].sha256;
      }
    }
    plan.section_diffs.push_back(std::move(diff));
  }

  plan.total_remote_bytes = max_remote_offset;

  uint64_t total_section_payload_bytes = 0;
  for (const auto& r_sec : parsed.remote_sections) {
    total_section_payload_bytes += r_sec.size();
  }

  // Check 80% threshold rule (or forced full download):
  // Measures the fraction of section payload data that needs to be downloaded.
  double change_ratio = 0.0;
  if (total_section_payload_bytes > 0) {
    uint64_t changed_payload_bytes =
        (total_section_payload_bytes > plan.bytes_to_reuse)
            ? (total_section_payload_bytes - plan.bytes_to_reuse)
            : 0;
    change_ratio = static_cast<double>(changed_payload_bytes) /
                   total_section_payload_bytes;
  } else if (plan.total_remote_bytes > 0) {
    change_ratio =
        static_cast<double>(plan.bytes_to_download) / plan.total_remote_bytes;
  }
  if (options.full_download_threshold > 0.0 &&
      change_ratio >= options.full_download_threshold) {
    plan.use_full_download = true;
    plan.bytes_to_download = plan.total_remote_bytes;
    plan.bytes_to_reuse = 0;
    plan.required_ranges.clear();
    if (plan.total_remote_bytes > 0) {
      plan.required_ranges.push_back(ByteRange{0, plan.total_remote_bytes - 1});
    }
    for (auto& diff : plan.section_diffs) {
      diff.action = SectionAction::kDownloadDiff;
    }
  }

  return plan;
}

bool IsRangeUnsupportedStatus(const absl::Status& status) {
  return status.code() == absl::StatusCode::kFailedPrecondition ||
         status.code() == absl::StatusCode::kUnimplemented ||
         status.code() == absl::StatusCode::kOutOfRange;
}

absl::Status DownloadFullModel(
    const RangeFetcher& fetcher, absl::string_view remote_header_bytes,
    uint64_t total_remote_bytes,
    const std::vector<SectionInfo>& sorted_remote_sections,
    absl::string_view temp_output_path, bool verify_checksums,
    bool fallback_on_range_error, std::ofstream& out) {
  out.seekp(0);
  out.write(remote_header_bytes.data(), remote_header_bytes.size());
  if (!out) {
    return absl::InternalError("Failed to write header to output file.");
  }
  if (total_remote_bytes > remote_header_bytes.size()) {
    absl::Status fetch_status =
        fetcher(remote_header_bytes.size(), total_remote_bytes - 1, out);
    if (!fetch_status.ok() && fallback_on_range_error &&
        IsRangeUnsupportedStatus(fetch_status)) {
      out.clear();
      out.seekp(0);
      if (!out) {
        return absl::InternalError(
            "Failed to rewind output stream for full-file download fallback.");
      }
      fetch_status = fetcher(0, total_remote_bytes - 1, out);
    }
    if (!fetch_status.ok()) {
      return absl::Status(fetch_status.code(),
                          absl::StrCat("Failed to fetch full model payload: ",
                                       fetch_status.message()));
    }
    if (!out) {
      return absl::DataLossError("Failed to write full model payload.");
    }
  }

  if (verify_checksums) {
    out.flush();
    absl::Status status =
        VerifyFileSectionChecksums(temp_output_path, sorted_remote_sections);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status AssembleDifferentialModel(
    const RangeFetcher& fetcher, absl::string_view remote_header_bytes,
    const std::vector<SectionInfo>& sorted_remote_sections,
    absl::string_view local_model_path, bool local_file_available,
    const std::vector<SectionInfo>& local_sections, const SyncOptions& options,
    uint64_t total_remote_bytes, absl::string_view temp_output_path,
    std::ofstream& out) {
  out.write(remote_header_bytes.data(), remote_header_bytes.size());
  if (!out) {
    return absl::InternalError("Failed to write header to output file.");
  }

  std::ifstream local_in;
  if (local_file_available) {
    local_in.open(std::string(local_model_path), std::ios::binary);
  }

  std::vector<char> buffer(kChunkSize);
  bool fallback_to_full = false;

  for (const auto& r_sec : sorted_remote_sections) {
    std::streampos pos = out.tellp();
    if (pos == std::streampos(-1) || !out) {
      return absl::InternalError(
          "Failed to determine output stream position before section write.");
    }
    uint64_t current_pos = static_cast<uint64_t>(pos);
    if (current_pos < r_sec.begin_offset) {
      uint64_t pad_bytes = r_sec.begin_offset - current_pos;
      std::vector<char> zeroes(std::min<uint64_t>(pad_bytes, 4096), 0);
      while (pad_bytes > 0) {
        size_t write_now = std::min<uint64_t>(pad_bytes, zeroes.size());
        out.write(zeroes.data(), write_now);
        if (!out) {
          return absl::DataLossError(
              "Failed to write padding bytes to output file.");
        }
        pad_bytes -= write_now;
      }
    }

    out.seekp(r_sec.begin_offset);
    if (!out) {
      return absl::InternalError(
          "Failed to seek to section offset in output file.");
    }

    const SectionInfo* local_match = nullptr;
    if (local_in.is_open()) {
      local_match = FindReusableLocalSection(r_sec, local_sections);
    }

    Sha256Hasher section_hasher;
    bool local_copy_succeeded = false;

    if (local_match != nullptr) {
      local_in.seekg(local_match->begin_offset);
      uint64_t bytes_left = r_sec.size();
      bool read_error = false;
      while (bytes_left > 0) {
        size_t read_size = std::min<uint64_t>(bytes_left, buffer.size());
        local_in.read(buffer.data(), read_size);
        std::streamsize read_actual = local_in.gcount();
        if (read_actual <= 0) {
          read_error = true;
          break;
        }
        out.write(buffer.data(), read_actual);
        if (!out) {
          return absl::DataLossError(
              "Failed to write local section payload to output file.");
        }
        if (options.verify_checksums) {
          section_hasher.Update(buffer.data(), read_actual);
        }
        bytes_left -= read_actual;
      }

      if (!read_error) {
        if (!options.verify_checksums || !IsValidSha256Hex(r_sec.sha256) ||
            absl::EqualsIgnoreCase(section_hasher.FinalHex(), r_sec.sha256)) {
          local_copy_succeeded = true;
        }
      }

      if (!local_copy_succeeded) {
        out.seekp(r_sec.begin_offset);
        if (!out) {
          return absl::InternalError(
              "Failed to seek back to section offset for remote fetch.");
        }
      }
    }

    if (!local_copy_succeeded && r_sec.size() > 0) {
      Sha256Hasher remote_hasher;
      HashingStreambuf hashing_buf(
          out.rdbuf(), options.verify_checksums ? &remote_hasher : nullptr);
      std::ostream wrapped_out(&hashing_buf);

      absl::Status fetch_status =
          fetcher(r_sec.begin_offset, r_sec.end_offset - 1, wrapped_out);
      if (!fetch_status.ok()) {
        if (options.fallback_on_range_error &&
            IsRangeUnsupportedStatus(fetch_status)) {
          fallback_to_full = true;
          break;
        }
        return absl::Status(fetch_status.code(),
                            absl::StrCat("Failed to fetch section payload: ",
                                         fetch_status.message()));
      }

      std::streampos fetch_pos = out.tellp();
      if (fetch_pos == std::streampos(-1) || !out) {
        return absl::InternalError(
            "Failed to determine output stream position after fetch.");
      }
      uint64_t written_size =
          static_cast<uint64_t>(fetch_pos) - r_sec.begin_offset;
      if (written_size != r_sec.size()) {
        return absl::DataLossError(absl::StrFormat(
            "Section payload size mismatch: expected %d bytes, got %d bytes.",
            r_sec.size(), written_size));
      }

      if (options.verify_checksums && IsValidSha256Hex(r_sec.sha256)) {
        std::string computed_sha = remote_hasher.FinalHex();
        if (!absl::EqualsIgnoreCase(computed_sha, r_sec.sha256)) {
          return absl::DataLossError(absl::StrFormat(
              "Section SHA-256 verification failed: expected %s, got %s",
              r_sec.sha256, computed_sha));
        }
      }
    }
  }

  if (local_in.is_open()) {
    local_in.close();
  }

  if (fallback_to_full) {
    return DownloadFullModel(fetcher, remote_header_bytes, total_remote_bytes,
                             sorted_remote_sections, temp_output_path,
                             options.verify_checksums,
                             options.fallback_on_range_error, out);
  }
  return absl::OkStatus();
}

}  // namespace

std::string SyncPlan::Summary() const {
  std::string out;
  absl::StrAppend(&out, "=== LiteRT-LM Sync Plan ===\n");
  absl::StrAppend(
      &out, "Status: ", is_up_to_date ? "UP TO DATE" : "UPDATE REQUIRED", "\n");
  absl::StrAppend(&out, "Mode:   ",
                  is_up_to_date ? "NO_OP"
                  : use_full_download
                      ? "FULL DOWNLOAD (Threshold exceeded or clean mode)"
                      : "DIFFERENTIAL SLICE UPDATE",
                  "\n");
  absl::StrAppendFormat(&out, "Total Remote Size: %d bytes (%.2f MB)\n",
                        total_remote_bytes,
                        total_remote_bytes / (1024.0 * 1024.0));
  absl::StrAppendFormat(&out, "Bytes to Reuse:    %d bytes (%.2f MB)\n",
                        bytes_to_reuse, bytes_to_reuse / (1024.0 * 1024.0));
  absl::StrAppendFormat(&out, "Bytes to Download: %d bytes (%.2f MB)\n",
                        bytes_to_download,
                        bytes_to_download / (1024.0 * 1024.0));
  if (!section_diffs.empty()) {
    absl::StrAppend(&out, "\nSection Comparison:\n");
    for (const auto& diff : section_diffs) {
      std::string data_type_str;
      if (diff.data_type >= static_cast<int>(schema::AnySectionDataType_MIN) &&
          diff.data_type <= static_cast<int>(schema::AnySectionDataType_MAX)) {
        const char* name = schema::EnumNameAnySectionDataType(
            static_cast<schema::AnySectionDataType>(diff.data_type));
        if (name != nullptr && name[0] != '\0') {
          data_type_str = name;
        }
      }
      if (data_type_str.empty()) {
        data_type_str = absl::StrCat(diff.data_type);
      }
      absl::StrAppendFormat(&out,
                            "  [%d] data_type=%s, model_type=\"%s\", size=%d "
                            "bytes\n",
                            diff.section_index, data_type_str, diff.model_type,
                            diff.size);
      if (diff.action == SectionAction::kReuseLocal) {
        absl::StrAppendFormat(
            &out, "      Action: REUSE LOCAL (SHA-256 matches: %.16s...)\n",
            diff.local_sha256);
      } else {
        absl::StrAppendFormat(
            &out,
            "      Action: DOWNLOAD DIFF (Local: %.16s..., Remote: %.16s...)\n",
            diff.local_sha256.empty() ? "(none)" : diff.local_sha256,
            diff.remote_sha256.empty() ? "(none)" : diff.remote_sha256);
        if (diff.size > 0 && !use_full_download) {
          absl::StrAppendFormat(&out, "      Range:  [%d, %d]\n",
                                diff.begin_offset,
                                diff.end_offset > 0 ? diff.end_offset - 1 : 0);
        }
      }
    }
  }
  return out;
}

absl::StatusOr<SyncPlan> ModelSync::PlanSync(RangeFetcher local_fetcher,
                                             RangeFetcher remote_fetcher,
                                             const SyncOptions& options) {
  auto parsed_res = ParseModelHeadersFromFetchers(
      local_fetcher, remote_fetcher, options.initial_header_fetch_bytes);
  if (!parsed_res.ok()) {
    return parsed_res.status();
  }
  return BuildSyncPlan(*parsed_res, options);
}

absl::StatusOr<SyncPlan> ModelSync::PlanSync(absl::string_view local_model_path,
                                             RangeFetcher fetcher,
                                             const SyncOptions& options) {
  auto parsed_res = ParseModelHeaders(local_model_path, fetcher,
                                      options.initial_header_fetch_bytes);
  if (!parsed_res.ok()) {
    return parsed_res.status();
  }
  return BuildSyncPlan(*parsed_res, options);
}

absl::Status ModelSync::Update(absl::string_view local_model_path,
                               absl::string_view output_model_path,
                               RangeFetcher fetcher,
                               const SyncOptions& options) {
  if (output_model_path.empty()) {
    return absl::InvalidArgumentError("output_model_path must not be empty.");
  }

  auto parsed_res = ParseModelHeaders(local_model_path, fetcher,
                                      options.initial_header_fetch_bytes);
  if (!parsed_res.ok()) {
    return parsed_res.status();
  }
  const auto& parsed = *parsed_res;
  const auto& remote_header_bytes = parsed.remote_header_bytes;
  const auto& remote_sections = parsed.remote_sections;
  const auto& local_sections = parsed.local_sections;
  bool local_exists = parsed.local_exists;

  SyncPlan plan = BuildSyncPlan(parsed, options);

  if (plan.is_up_to_date) {
    if (local_model_path == output_model_path) {
      return absl::OkStatus();
    }
    std::error_code ec;
    std::filesystem::copy_file(
        std::string(local_model_path), std::string(output_model_path),
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      return absl::InternalError(absl::StrCat(
          "Failed to copy up-to-date model to output path: ", ec.message()));
    }
    return absl::OkStatus();
  }

  uint64_t max_remote_offset = plan.total_remote_bytes;
  bool use_full_download = plan.use_full_download;

  std::filesystem::path out_path(std::string{output_model_path});
  std::filesystem::path out_dir = out_path.parent_path();
  if (out_dir.empty()) {
    std::error_code cur_ec;
    out_dir = std::filesystem::current_path(cur_ec);
    if (cur_ec) {
      out_dir = ".";
    }
  }

  bool space_constrained = false;
  // Pre-flight disk space check:
  if (options.check_disk_space) {
    uint64_t available_space = 0;
    bool has_space_info = false;
    if (options.available_disk_space_override.has_value()) {
      available_space = *options.available_disk_space_override;
      has_space_info = true;
    } else {
      std::error_code ec;
      auto space_info = std::filesystem::space(out_dir, ec);
      if (!ec) {
        available_space = space_info.available;
        has_space_info = true;
      }
    }
    if (has_space_info) {
      if (available_space < max_remote_offset) {
        space_constrained = true;
      }
      if (options.clean && local_exists &&
          local_model_path == output_model_path) {
        std::error_code size_ec;
        uint64_t old_size =
            std::filesystem::file_size(std::string(local_model_path), size_ec);
        if (!size_ec) {
          available_space += old_size;
        }
      }
      if (available_space < max_remote_offset) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Insufficient disk space in %s: %d bytes available, %d bytes "
            "required for staging. Pass --clean to replace old model directly.",
            out_dir.string(), available_space, max_remote_offset));
      }
    }
  }

  if (space_constrained && options.clean) {
    use_full_download = true;
  }

  // Validate and sort remote section ranges before assembly:
  std::vector<SectionInfo> sorted_remote_sections = remote_sections;
  std::sort(sorted_remote_sections.begin(), sorted_remote_sections.end(),
            [](const SectionInfo& a, const SectionInfo& b) {
              return a.begin_offset < b.begin_offset;
            });

  uint64_t prev_end_offset = remote_header_bytes.size();
  for (const auto& sec : sorted_remote_sections) {
    if (sec.begin_offset < remote_header_bytes.size() ||
        sec.end_offset < sec.begin_offset ||
        sec.begin_offset < prev_end_offset) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Malformed or overlapping section offset range: "
                          "header_size=%d, begin=%d, end=%d, prev_end=%d",
                          remote_header_bytes.size(), sec.begin_offset,
                          sec.end_offset, prev_end_offset));
    }
    prev_end_offset = sec.end_offset;
  }

  std::string temp_output_path =
      absl::StrCat(output_model_path, ".tmp_updating");

  bool update_success = false;
  absl::Cleanup remove_tmp_guard = [&] {
    if (!update_success) {
      std::error_code ec;
      std::filesystem::remove(temp_output_path, ec);
    }
  };

  std::ofstream out(temp_output_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return absl::PermissionDeniedError(absl::StrCat(
        "Permission denied: Cannot open output file for writing: ",
        temp_output_path, ". Check destination directory permissions."));
  }

  // If clean is enabled with in-place overwrite and space is constrained,
  // reclaim space upfront AFTER temp output file has been successfully opened:
  bool local_file_available = local_exists;
  if (options.clean && local_exists && local_model_path == output_model_path &&
      space_constrained && use_full_download) {
    std::error_code rm_ec;
    std::filesystem::remove(std::string(local_model_path), rm_ec);
    local_file_available = false;
  }

  if (use_full_download) {
    absl::Status full_status = DownloadFullModel(
        fetcher, remote_header_bytes, max_remote_offset, sorted_remote_sections,
        temp_output_path, options.verify_checksums,
        options.fallback_on_range_error, out);
    if (!full_status.ok()) {
      return full_status;
    }
  } else {
    absl::Status diff_status = AssembleDifferentialModel(
        fetcher, remote_header_bytes, sorted_remote_sections, local_model_path,
        local_file_available, local_sections, options, max_remote_offset,
        temp_output_path, out);
    if (!diff_status.ok()) {
      return diff_status;
    }
  }

  out.flush();
  out.close();
  if (!out) {
    return absl::DataLossError("Failed to flush and close output model file.");
  }

  std::error_code ec;
  std::filesystem::rename(temp_output_path, std::string(output_model_path), ec);
  if (ec) {
    return absl::InternalError(absl::StrCat(
        "Failed to rename temporary model to destination: ", ec.message()));
  }

  update_success = true;
  return absl::OkStatus();
}

namespace {

std::string SanitizeModelIdToFilename(absl::string_view model_id) {
  std::string filename;
  filename.reserve(model_id.size() + 9);
  for (char c : model_id) {
    if (absl::ascii_isalnum(static_cast<unsigned char>(c)) || c == '-' ||
        c == '_' || c == '.') {
      filename.push_back(c);
    } else {
      filename.push_back('_');
    }
  }
  if (!absl::EndsWith(filename, ".litertlm")) {
    absl::StrAppend(&filename, ".litertlm");
  }
  return filename;
}

}  // namespace

absl::StatusOr<std::string> ModelSync::Sync(absl::string_view cache_dir,
                                            absl::string_view model_id,
                                            RangeFetcher fetcher,
                                            const SyncOptions& options) {
  if (cache_dir.empty()) {
    return absl::InvalidArgumentError("cache_dir cannot be empty.");
  }
  if (model_id.empty()) {
    return absl::InvalidArgumentError("model_id cannot be empty.");
  }

  std::error_code ec;
  std::filesystem::path dir_path(std::string{cache_dir});
  std::filesystem::create_directories(dir_path, ec);
  if (ec) {
    return absl::InternalError(absl::StrCat("Failed to create cache directory ",
                                            cache_dir, ": ", ec.message()));
  }

  std::string filename = SanitizeModelIdToFilename(model_id);
  std::filesystem::path cached_model_path = dir_path / filename;
  std::string local_path_str = cached_model_path.string();

  absl::Status update_status =
      Update(local_path_str, local_path_str, fetcher, options);

  if (update_status.ok()) {
    return local_path_str;
  }

  std::error_code exists_ec;
  if (options.allow_offline_fallback &&
      (update_status.code() == absl::StatusCode::kUnavailable ||
       update_status.code() == absl::StatusCode::kDeadlineExceeded) &&
      std::filesystem::exists(cached_model_path, exists_ec) && !exists_ec) {
    schema::LitertlmHeader cached_header;
    if (schema::ReadHeaderFromLiteRTLM(local_path_str, &cached_header).ok() &&
        cached_header.metadata != nullptr) {
      return local_path_str;
    }
  }

  return update_status;
}

namespace {

bool IsValidHttpUrl(absl::string_view url) {
  if (!IsHttpUrl(url)) return false;
  for (unsigned char c : url) {
    if (c <= 0x20 || c == 0x7f || c == '"') return false;
  }
  return true;
}

int ParseLastHttpStatusCode(absl::string_view headers_text) {
  int last_code = 0;
  size_t pos = 0;
  while (pos < headers_text.size()) {
    size_t line_end = headers_text.find('\n', pos);
    if (line_end == absl::string_view::npos) {
      line_end = headers_text.size();
    }
    absl::string_view line = headers_text.substr(pos, line_end - pos);
    pos = line_end + 1;
    if (!absl::StartsWithIgnoreCase(line, "HTTP/")) continue;
    size_t sp = line.find(' ');
    if (sp == absl::string_view::npos) continue;
    while (sp < line.size() && line[sp] == ' ') ++sp;
    if (sp + 3 <= line.size() &&
        std::isdigit(static_cast<unsigned char>(line[sp])) &&
        std::isdigit(static_cast<unsigned char>(line[sp + 1])) &&
        std::isdigit(static_cast<unsigned char>(line[sp + 2])) &&
        (sp + 3 == line.size() ||
         !std::isdigit(static_cast<unsigned char>(line[sp + 3])))) {
      last_code = (line[sp] - '0') * 100 + (line[sp + 1] - '0') * 10 +
                  (line[sp + 2] - '0');
    }
  }
  return last_code;
}

int ReadHttpStatusFromFile(absl::string_view hdr_path) {
  if (hdr_path.empty()) return 0;
  std::ifstream in(std::string(hdr_path), std::ios::binary);
  if (!in.is_open()) return 0;
  std::string contents;
  char buf[4096];
  constexpr size_t kMaxHeaderFileBytes = 256 * 1024;
  while (contents.size() < kMaxHeaderFileBytes) {
    in.read(buf, sizeof(buf));
    std::streamsize n = in.gcount();
    if (n <= 0) break;
    contents.append(buf, static_cast<size_t>(n));
  }
  return ParseLastHttpStatusCode(contents);
}

absl::Status MapHttpErrorCode(int http_status, absl::string_view url_str) {
  switch (http_status) {
    case 400:
      return absl::InvalidArgumentError(
          absl::StrCat("HTTP 400 Bad Request for ", url_str));
    case 401:
      return absl::UnauthenticatedError(
          absl::StrCat("HTTP 401 Unauthorized for ", url_str));
    case 403:
      return absl::PermissionDeniedError(
          absl::StrCat("HTTP 403 Forbidden for ", url_str));
    case 404:
      return absl::NotFoundError(
          absl::StrCat("HTTP 404 Not Found for ", url_str));
    case 408:
      return absl::DeadlineExceededError(
          absl::StrCat("HTTP 408 Request Timeout for ", url_str));
    case 416:
      return absl::OutOfRangeError(
          absl::StrCat("HTTP 416 Range Not Satisfiable for ", url_str));
    case 429:
      return absl::ResourceExhaustedError(
          absl::StrCat("HTTP 429 Too Many Requests for ", url_str));
    default:
      if (http_status >= 400 && http_status < 500) {
        return absl::FailedPreconditionError(
            absl::StrCat("HTTP ", http_status, " client error for ", url_str));
      }
      return absl::UnavailableError(
          absl::StrCat("HTTP ", http_status, " server error for ", url_str));
  }
}

absl::Status MapCurlOutcome(bool write_failed, bool exceeded_expected,
                            uint64_t start_byte, int exit_code, int http_status,
                            absl::string_view url_str) {
  if (write_failed) {
    return absl::DataLossError("Failed to write HTTP range bytes to stream.");
  }
  if (exceeded_expected) {
    return absl::UnimplementedError(
        absl::StrCat("Server ignored Range header for ", url_str));
  }
  if (exit_code == 127) {
    return absl::FailedPreconditionError(
        absl::StrCat("curl executable not found while fetching ", url_str));
  }
  if (exit_code == 28) {
    return absl::DeadlineExceededError(
        absl::StrCat("HTTP request timed out: ", url_str));
  }
  if (exit_code == 63) {
    return absl::UnimplementedError(
        absl::StrCat("Server ignored Range header for ", url_str));
  }
  if (exit_code != 0 || http_status >= 400) {
    if (http_status >= 400) {
      return MapHttpErrorCode(http_status, url_str);
    }
    return absl::UnavailableError(absl::StrCat("curl failed with exit code ",
                                               exit_code, " for ", url_str));
  }
  if (start_byte > 0 && http_status == 200) {
    return absl::UnimplementedError(absl::StrCat(
        "Server ignored Range header (returned HTTP 200 OK) for ", url_str));
  }
  return absl::OkStatus();
}

#if LITERT_LM_HAS_POSIX_CURL
std::string CreateTempHeaderPath() {
  std::error_code ec;
  std::filesystem::path tmp_dir = std::filesystem::temp_directory_path(ec);
  if (ec || tmp_dir.empty()) {
    return "";
  }
  std::string tmpl = (tmp_dir / "litert_lm_http_hdr_XXXXXX").string();
  int fd = mkstemp(tmpl.data());
  if (fd < 0) {
    return "";
  }
  close(fd);
  return tmpl;
}
#elif LITERT_LM_HAS_WIN32_CURL
std::string CreateTempHeaderPath() {
  char tmp_dir[MAX_PATH] = {};
  DWORD dir_len = GetTempPathA(MAX_PATH, tmp_dir);
  if (dir_len == 0 || dir_len >= MAX_PATH) {
    return "";
  }
  char tmp_file[MAX_PATH] = {};
  if (GetTempFileNameA(tmp_dir, "llm", 0, tmp_file) == 0) {
    return "";
  }
  std::string path(tmp_file);
  if (path.find('"') != std::string::npos) {
    DeleteFileA(path.c_str());
    return "";
  }
  return path;
}

std::string ResolveWindowsCurlPath(absl::string_view explicit_path) {
  if (!explicit_path.empty()) {
    std::string path(explicit_path);
    return path.find('"') == std::string::npos ? path : "";
  }
  char sys_dir[MAX_PATH] = {};
  UINT len = GetSystemDirectoryA(sys_dir, MAX_PATH);
  if (len > 0 && len < MAX_PATH) {
    std::string candidate = absl::StrCat(sys_dir, "\\curl.exe");
    DWORD attrs = GetFileAttributesA(candidate.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES &&
        (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0) {
      return candidate;
    }
  }
  // Scan absolute PATH entries only so CWD / relative directories cannot
  // hijack.
  DWORD path_len = GetEnvironmentVariableA("PATH", nullptr, 0);
  if (path_len > 1) {
    std::string path_env(path_len, '\0');
    DWORD actual = GetEnvironmentVariableA("PATH", path_env.data(), path_len);
    if (actual > 0 && actual < path_len) {
      path_env.resize(actual);
      size_t start = 0;
      while (start <= path_env.size()) {
        size_t semi = path_env.find(';', start);
        if (semi == std::string::npos) semi = path_env.size();
        absl::string_view dir(path_env.data() + start, semi - start);
        start = semi + 1;
        bool is_abs_drive = dir.size() >= 3 &&
                            std::isalpha(static_cast<unsigned char>(dir[0])) &&
                            dir[1] == ':' && (dir[2] == '\\' || dir[2] == '/');
        bool is_unc = dir.size() >= 2 && dir[0] == '\\' && dir[1] == '\\';
        if (!is_abs_drive && !is_unc) continue;
        std::string candidate = absl::StrCat(dir, "\\curl.exe");
        if (candidate.find('"') != std::string::npos) continue;
        DWORD attrs = GetFileAttributesA(candidate.c_str());
        if (attrs != INVALID_FILE_ATTRIBUTES &&
            (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0) {
          return candidate;
        }
      }
    }
  }
  return "";
}
#endif

RangeFetcher CreateHttpRangeFetcherImpl(absl::string_view url,
                                        absl::string_view curl_path) {
  return [url_str = std::string(url), curl_exe = std::string(curl_path)](
             uint64_t start_byte, uint64_t end_byte,
             std::ostream& out) -> absl::Status {
    if (!IsValidHttpUrl(url_str)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid HTTP URL: ", url_str));
    }
    if (end_byte < start_byte ||
        end_byte - start_byte == std::numeric_limits<uint64_t>::max()) {
      return absl::InvalidArgumentError("Invalid byte range.");
    }
    const uint64_t expected_bytes = (end_byte - start_byte) + 1;
#if LITERT_LM_HAS_POSIX_CURL
    std::string range_arg = absl::StrCat(start_byte, "-", end_byte);
    std::string max_size_arg = absl::StrCat(expected_bytes);
    std::string hdr_path = CreateTempHeaderPath();
    absl::Cleanup remove_hdr = [&] {
      if (!hdr_path.empty()) {
        std::error_code ec;
        std::filesystem::remove(hdr_path, ec);
      }
    };

    const char* exe = curl_exe.empty() ? "curl" : curl_exe.c_str();
    std::vector<const char*> argv = {
        exe,
        "-f",
        "-s",
        "-S",
        "-L",
        "-g",
        "--proto",
        "=http,https",
        "--proto-redir",
        "=http,https",
        "--connect-timeout",
        "15",
        "--speed-limit",
        "1",
        "--speed-time",
        "30",
        "--max-filesize",
        max_size_arg.c_str(),
        "--range",
        range_arg.c_str(),
    };
    if (!hdr_path.empty()) {
      argv.push_back("-D");
      argv.push_back(hdr_path.c_str());
    }
    argv.push_back("--");
    argv.push_back(url_str.c_str());
    argv.push_back(nullptr);

    int pipefd[2] = {-1, -1};
    if (pipe(pipefd) != 0) {
      return absl::InternalError(
          absl::StrCat("pipe() failed: ", std::strerror(errno)));
    }
    if (fcntl(pipefd[0], F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(pipefd[1], F_SETFD, FD_CLOEXEC) != 0) {
      int err = errno;
      close(pipefd[0]);
      close(pipefd[1]);
      return absl::InternalError(
          absl::StrCat("fcntl(FD_CLOEXEC) failed: ", std::strerror(err)));
    }
    pid_t pid = fork();
    if (pid < 0) {
      int err = errno;
      close(pipefd[0]);
      close(pipefd[1]);
      return absl::InternalError(
          absl::StrCat("fork() failed: ", std::strerror(err)));
    }
    if (pid == 0) {
      dup2(pipefd[1], STDOUT_FILENO);
      close(pipefd[0]);
      close(pipefd[1]);
      execvp(exe, const_cast<char* const*>(argv.data()));
      _exit(127);
    }

    close(pipefd[1]);
    uint64_t total_written = 0;
    bool exceeded_expected = false;
    bool write_failed = false;
    std::vector<char> buffer(64 * 1024);
    while (true) {
      ssize_t n = read(pipefd[0], buffer.data(), buffer.size());
      if (n > 0) {
        if (static_cast<uint64_t>(n) > expected_bytes - total_written) {
          exceeded_expected = true;
          break;
        }
        out.write(buffer.data(), static_cast<std::streamsize>(n));
        if (!out) {
          write_failed = true;
          break;
        }
        total_written += static_cast<uint64_t>(n);
      } else if (n == 0) {
        break;
      } else if (errno != EINTR) {
        break;
      }
    }
    close(pipefd[0]);

    if (exceeded_expected || write_failed) {
      kill(pid, SIGKILL);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    int http_status = ReadHttpStatusFromFile(hdr_path);
    return MapCurlOutcome(write_failed, exceeded_expected, start_byte,
                          exit_code, http_status, url_str);
#elif LITERT_LM_HAS_WIN32_CURL
    std::string resolved_curl = ResolveWindowsCurlPath(curl_exe);
    if (resolved_curl.empty()) {
      return absl::FailedPreconditionError(
          "curl.exe not found in System32 or PATH.");
    }
    std::string hdr_path = CreateTempHeaderPath();
    absl::Cleanup remove_hdr = [&] {
      if (!hdr_path.empty()) {
        DeleteFileA(hdr_path.c_str());
      }
    };
    std::string cmd_line = absl::StrCat(
        "\"", resolved_curl,
        "\" -f -s -S -L -g --proto =http,https --proto-redir =http,https "
        "--connect-timeout 15 --speed-limit 1 --speed-time 30 --max-filesize ",
        expected_bytes, " --range ", start_byte, "-", end_byte);
    if (!hdr_path.empty()) {
      absl::StrAppend(&cmd_line, " -D \"", hdr_path, "\"");
    }
    absl::StrAppend(&cmd_line, " -- ", url_str);

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
      return absl::InternalError(
          absl::StrCat("CreatePipe() failed: error ", GetLastError()));
    }
    auto close_read = absl::MakeCleanup([&] {
      if (read_pipe != nullptr) CloseHandle(read_pipe);
    });
    if (!SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0)) {
      DWORD err = GetLastError();
      CloseHandle(write_pipe);
      return absl::InternalError(
          absl::StrCat("SetHandleInformation() failed: error ", err));
    }

    HANDLE handles_to_inherit[2] = {write_pipe, nullptr};
    DWORD handle_count = 1;
    HANDLE stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
    DWORD stderr_flags = 0;
    bool inherit_stderr = stderr_handle != nullptr &&
                          stderr_handle != INVALID_HANDLE_VALUE &&
                          GetHandleInformation(stderr_handle, &stderr_flags) &&
                          (stderr_flags & HANDLE_FLAG_INHERIT) != 0;
    if (inherit_stderr) {
      handles_to_inherit[handle_count++] = stderr_handle;
    }

    SIZE_T attr_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
    if (attr_size == 0) {
      CloseHandle(write_pipe);
      return absl::InternalError(absl::StrCat(
          "InitializeProcThreadAttributeList() size failed: error ",
          GetLastError()));
    }
    std::vector<char> attr_buf(attr_size);
    auto* attr_list =
        reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
    if (!InitializeProcThreadAttributeList(attr_list, 1, 0, &attr_size)) {
      DWORD err = GetLastError();
      CloseHandle(write_pipe);
      return absl::InternalError(absl::StrCat(
          "InitializeProcThreadAttributeList() failed: error ", err));
    }
    auto delete_attrs =
        absl::MakeCleanup([&] { DeleteProcThreadAttributeList(attr_list); });
    if (!UpdateProcThreadAttribute(
            attr_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles_to_inherit,
            handle_count * sizeof(HANDLE), nullptr, nullptr)) {
      DWORD err = GetLastError();
      CloseHandle(write_pipe);
      return absl::InternalError(
          absl::StrCat("UpdateProcThreadAttribute() failed: error ", err));
    }

    STARTUPINFOEXA si{};
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = nullptr;
    si.StartupInfo.hStdOutput = write_pipe;
    si.StartupInfo.hStdError = inherit_stderr ? stderr_handle : nullptr;
    si.lpAttributeList = attr_list;
    PROCESS_INFORMATION pi{};
    BOOL started =
        CreateProcessA(resolved_curl.c_str(), cmd_line.data(), nullptr, nullptr,
                       /*bInheritHandles=*/TRUE,
                       CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                       nullptr, &si.StartupInfo, &pi);
    DWORD start_err = started ? 0 : GetLastError();
    CloseHandle(write_pipe);
    if (!started) {
      if (start_err == ERROR_FILE_NOT_FOUND ||
          start_err == ERROR_PATH_NOT_FOUND) {
        return absl::FailedPreconditionError(absl::StrCat(
            "curl.exe not found (error ", start_err, ") for ", url_str));
      }
      return absl::UnavailableError(absl::StrCat(
          "Failed to launch curl.exe (error ", start_err, ") for ", url_str));
    }
    auto close_proc = absl::MakeCleanup([&] {
      CloseHandle(pi.hProcess);
      CloseHandle(pi.hThread);
    });

    uint64_t total_written = 0;
    bool exceeded_expected = false;
    bool write_failed = false;
    bool read_failed = false;
    std::vector<char> buffer(64 * 1024);
    while (true) {
      DWORD n = 0;
      if (!ReadFile(read_pipe, buffer.data(), static_cast<DWORD>(buffer.size()),
                    &n, nullptr)) {
        if (GetLastError() != ERROR_BROKEN_PIPE) {
          read_failed = true;
        }
        break;
      }
      if (n == 0) break;
      if (static_cast<uint64_t>(n) > expected_bytes - total_written) {
        exceeded_expected = true;
        break;
      }
      out.write(buffer.data(), static_cast<std::streamsize>(n));
      if (!out) {
        write_failed = true;
        break;
      }
      total_written += static_cast<uint64_t>(n);
    }
    CloseHandle(read_pipe);
    read_pipe = nullptr;

    if (exceeded_expected || write_failed || read_failed) {
      TerminateProcess(pi.hProcess, 1);
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    if (read_failed) {
      return absl::UnavailableError(
          absl::StrCat("Failed reading curl.exe output for ", url_str));
    }
    int http_status = ReadHttpStatusFromFile(hdr_path);
    return MapCurlOutcome(write_failed, exceeded_expected, start_byte,
                          static_cast<int>(exit_code), http_status, url_str);
#else
    (void)expected_bytes;
    (void)out;
    (void)curl_exe;
    return absl::UnimplementedError(
        "CreateHttpRangeFetcher is only supported on desktop hosts (Linux, "
        "macOS, Windows).");
#endif
  };
}

}  // namespace

bool IsHttpUrl(absl::string_view source) {
  return absl::StartsWithIgnoreCase(source, "http://") ||
         absl::StartsWithIgnoreCase(source, "https://");
}

RangeFetcher CreateHttpRangeFetcher(absl::string_view url) {
  return CreateHttpRangeFetcherImpl(url, "");
}

namespace internal {

RangeFetcher CreateHttpRangeFetcherWithExecutable(absl::string_view url,
                                                  absl::string_view curl_path) {
  return CreateHttpRangeFetcherImpl(url, curl_path);
}

}  // namespace internal

}  // namespace litert::lm::sync
