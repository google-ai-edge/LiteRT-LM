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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SYNC_MODEL_SYNC_H_
#define THIRD_PARTY_ODML_LITERT_LM_SYNC_MODEL_SYNC_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm::sync {

// Callback provided by the host platform to fetch byte range [start, end]
// inclusive over HTTP Range requests and write payload into `output_stream`.
using RangeFetcher = std::function<absl::Status(
    uint64_t start_byte, uint64_t end_byte, std::ostream& output_stream)>;

// Details about a byte range that needs to be downloaded.
struct ByteRange {
  uint64_t start_byte;
  uint64_t end_byte;  // inclusive
};

// Action planned for a section during synchronization.
enum class SectionAction {
  kReuseLocal,    // Reused directly from local cache (0 network download).
  kDownloadDiff,  // Downloaded from remote (hash changed or new section).
};

// Information on a section comparison between local and remote models.
struct SectionDiff {
  // Index of the section in the remote model's section metadata.
  size_t section_index = 0;
  // `AnySectionDataType` of the section (for example, TFLite model or
  // tokenizer), stored as its integer value.
  int data_type = 0;
  // Value of the section's `model_type` metadata entry; empty if not set.
  std::string model_type;
  // SHA-256 hex digest recorded in the local header for the matching local
  // section (same hash when reused, same index otherwise); empty if there is
  // no such section or it has no recorded digest.
  std::string local_sha256;
  // SHA-256 hex digest recorded in the remote header; empty if not recorded.
  std::string remote_sha256;
  // Size of the section payload in bytes.
  uint64_t size = 0;
  // Offset of the first byte of the section payload in the remote file.
  uint64_t begin_offset = 0;
  // Offset one past the last byte of the section payload in the remote file.
  uint64_t end_offset = 0;
  // Whether the section is reused from the local file or downloaded.
  SectionAction action = SectionAction::kDownloadDiff;
};

// Summary of an update/sync plan.
struct SyncPlan {
  // True if the local model already matches the remote model, so nothing
  // needs to be downloaded.
  bool is_up_to_date = false;
  // If true, a full sequential model download is performed instead of slice
  // assembly (e.g. changes exceed full_download_threshold, or forced).
  bool use_full_download = false;
  // Size of the remote model in bytes as implied by its header: the end offset
  // of the last section, or the header size if larger.
  uint64_t total_remote_bytes = 0;
  // Number of bytes that the update fetches from the remote model.
  uint64_t bytes_to_download = 0;
  // Number of section bytes copied from the local model instead of downloaded.
  uint64_t bytes_to_reuse = 0;
  // Byte ranges of the remote model that must be fetched.
  std::vector<ByteRange> required_ranges;
  // Per-section comparison between the local and remote models.
  std::vector<SectionDiff> section_diffs;

  // Formats a human-readable comparison report of the sync plan.
  std::string Summary() const;
};

// Options configuring the sync process.
struct SyncOptions {
  // If true, computes and validates SHA-256 for all assembled sections against
  // the remote header before finishing.
  bool verify_checksums = true;

  // Initial header fetch size in bytes. Defaults to 16 KB (16384 bytes).
  uint64_t initial_header_fetch_bytes = 16384;

  // If true and update fails (e.g. offline / network drop), falls back to the
  // already cached local model if it exists on disk.
  bool allow_offline_fallback = true;

  // If true, checks available disk space before starting download.
  bool check_disk_space = true;

  // If true, deletes existing local model upfront if disk space is constrained
  // or full download is required, reducing peak disk requirement from 2x to 1x.
  bool clean = false;

  // Ratio threshold (0.0 to 1.0) above which section changes trigger a direct
  // full model download instead of slicing. Defaults to 0.80 (80%).
  double full_download_threshold = 0.80;

  // If true, automatically falls back to full sequential download if an HTTP
  // Range slice fetch fails (e.g. server returns HTTP 200 instead of 206).
  bool fallback_on_range_error = true;

  // Test hook: override available disk space in bytes for testing
  // space-constrained scenarios.
  std::optional<uint64_t> available_disk_space_override;
};

// Differential synchronizer for LiteRT-LM (`.litertlm`) model files.
//
// ModelSync brings a local model up to date with a remote version while
// downloading as little as possible. It fetches the small remote FlatBuffer
// header through a `RangeFetcher`, compares each section's SHA-256 hash with
// the local model, copies unchanged sections from the local file, and
// downloads only the sections that changed. It downloads the whole file
// instead when the changed fraction reaches
// `SyncOptions::full_download_threshold` (0 disables this), when a range fetch
// fails and `fallback_on_range_error` is set, or when `clean` is set and disk
// space is short.
//
// All methods are static. Reads of the remote model go through the
// caller-provided `RangeFetcher`.
//
// Example:
//
//   // Optional: inspect what would change without writing anything.
//   absl::StatusOr<SyncPlan> plan = ModelSync::PlanSync(local_path, fetcher);
//   if (plan.ok()) std::cout << plan->Summary();
//
//   // Leaves the file untouched if it is already up to date.
//   absl::Status status = ModelSync::Update(local_path, local_path, fetcher);
//   if (!status.ok()) { /* Handle the error. */ }
class ModelSync {
 public:
  // Inspects the local model (if present) and queries the remote header via
  // `fetcher` to determine what byte ranges must be downloaded.
  static absl::StatusOr<SyncPlan> PlanSync(absl::string_view local_model_path,
                                           RangeFetcher fetcher,
                                           const SyncOptions& options = {});

  // Inspects models via two RangeFetchers without requiring filesystem access.
  // Ideal for pure metadata inspection, --diff, and in-memory or remote
  // streams.
  static absl::StatusOr<SyncPlan> PlanSync(RangeFetcher local_fetcher,
                                           RangeFetcher remote_fetcher,
                                           const SyncOptions& options = {});

  // Executes the complete differential update / sync:
  // 1. Fetches remote header via `fetcher`.
  // 2. Compares section hashes between local and remote.
  // 3. Downloads modified sections via `fetcher`.
  // 4. Stitches unmodified sections directly from `local_model_path`.
  // 5. Writes the target `.litertlm` file to `output_model_path`.
  static absl::Status Update(absl::string_view local_model_path,
                             absl::string_view output_model_path,
                             RangeFetcher fetcher,
                             const SyncOptions& options = {});

  // Synchronizes a model into the local cache directory:
  // 1. Resolves destination path inside `cache_dir` for `model_id`.
  // 2. Checks local cache for existing model.
  // 3. Peeks remote header via `fetcher` and differentially updates if needed.
  // 4. If network is unreachable and `allow_offline_fallback` is enabled, falls
  //    back to existing cached model.
  // 5. Returns the filesystem path of the ready-to-use .litertlm file.
  static absl::StatusOr<std::string> Sync(absl::string_view cache_dir,
                                          absl::string_view model_id,
                                          RangeFetcher fetcher,
                                          const SyncOptions& options = {});
};

// Returns true if `source` uses an `http://` or `https://` scheme.
bool IsHttpUrl(absl::string_view source);

// Creates a `RangeFetcher` that fetches inclusive byte ranges `[start_byte,
// end_byte]` from an `http://` or `https://` URL (following redirects).
//
// Supported on desktop hosts (Linux, macOS, and Windows with `curl` /
// `curl.exe`). Invoking the returned `RangeFetcher` returns
// `absl::InvalidArgumentError` on malformed URLs and
// `absl::UnimplementedError` on unsupported platforms (e.g. Android, iOS,
// WebAssembly).
RangeFetcher CreateHttpRangeFetcher(absl::string_view url);

namespace internal {

// Creates an HTTP `RangeFetcher` using an explicit `curl` executable path.
// Exposed for hermetic testing without mutating process-wide `PATH`.
RangeFetcher CreateHttpRangeFetcherWithExecutable(absl::string_view url,
                                                  absl::string_view curl_path);

}  // namespace internal

}  // namespace litert::lm::sync

#endif  // THIRD_PARTY_ODML_LITERT_LM_SYNC_MODEL_SYNC_H_
