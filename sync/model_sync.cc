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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <iosfwd>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <vector>

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
    return absl::InternalError(
        absl::StrCat("Failed to open file for checksum verification: ",
                     file_path));
  }
  std::vector<char> buffer(kChunkSize);
  for (const auto& sec : sections) {
    if (sec.size() == 0 || !IsValidSha256Hex(sec.sha256)) continue;
    in.seekg(sec.begin_offset);
    if (!in) {
      return absl::DataLossError(
          absl::StrCat("Failed to seek to section at offset ",
                       sec.begin_offset));
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
      return absl::InvalidArgumentError(
          absl::StrFormat("Malformed or overlapping section offset range: "
                          "header_size=%d, begin=%d, end=%d, prev_end=%d",
                          header_bytes.size(), sec.begin_offset,
                          sec.end_offset, prev_end));
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
      auto local_sec_res =
          ParseAndVerifyHeader(parsed.local_header_bytes);
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

absl::Status DownloadFullModel(
    const RangeFetcher& fetcher, absl::string_view remote_header_bytes,
    uint64_t total_remote_bytes,
    const std::vector<SectionInfo>& sorted_remote_sections,
    absl::string_view temp_output_path, bool verify_checksums,
    std::ofstream& out) {
  out.seekp(0);
  out.write(remote_header_bytes.data(), remote_header_bytes.size());
  if (!out) {
    return absl::InternalError("Failed to write header to output file.");
  }
  if (total_remote_bytes > remote_header_bytes.size()) {
    absl::Status fetch_status =
        fetcher(remote_header_bytes.size(), total_remote_bytes - 1, out);
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
        bool is_range_unsupported =
            options.fallback_on_range_error &&
            (fetch_status.code() == absl::StatusCode::kFailedPrecondition ||
             fetch_status.code() == absl::StatusCode::kUnimplemented ||
             fetch_status.code() == absl::StatusCode::kOutOfRange);
        if (is_range_unsupported) {
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
    return DownloadFullModel(
        fetcher, remote_header_bytes, total_remote_bytes,
        sorted_remote_sections, temp_output_path, options.verify_checksums,
        out);
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
      absl::StrAppendFormat(
          &out, "  [%d] data_type=%d, model_type=\"%s\", size=%d bytes\n",
          diff.section_index, diff.data_type, diff.model_type, diff.size);
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
        temp_output_path, options.verify_checksums, out);
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

}  // namespace litert::lm::sync
