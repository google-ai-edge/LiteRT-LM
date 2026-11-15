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

// Command-line tool for differential synchronization of LiteRT-LM
// (`.litertlm`) model files.
//
// The tool compares the per-section SHA-256 hashes of an existing local model
// (--local) with an upstream model file or HTTP/HTTPS URL (--remote). It then
// writes an updated model that reuses unchanged sections from the local file
// and fetches only changed sections from the upstream source, or downloads the
// whole file when the changed fraction reaches or exceeds --threshold. By
// default, --local is updated in place; pass --output to write the result to a
// separate file instead. See README.md in this directory for the full list of
// flags.
//
// Build and run with:
//   bazel run //sync:model_sync_cli -- <flags>
//
// Examples:
//   # Print which sections changed against Hugging Face, without writing files:
//   model_sync_cli --diff --local=v1.litertlm \
//     --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm
//
//   # Update v1.litertlm in place from a remote model:
//   model_sync_cli --local=v1.litertlm \
//     --remote=https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/usage.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "sync/model_sync.h"

// Remote upstream model file or URL:
ABSL_FLAG(std::string, remote, "",
          "Path or HTTP/HTTPS URL of the upstream remote .litertlm file.");

// Existing local model file on disk:
ABSL_FLAG(std::string, local, "",
          "Path to the existing local .litertlm file on disk to reuse "
          "unchanged sections from (with --diff, may also be an http(s):// "
          "URL).");

// Output destination:
ABSL_FLAG(std::string, output, "",
          "Path where the synced model will be written. If omitted, updates "
          "--local in-place.");

// Inspection / dry-run mode:
ABSL_FLAG(bool, diff, false,
          "If true, compares local and remote models and prints the section "
          "diff plan without downloading.");

// Clean storage mode (delete local upfront on low-storage devices):
ABSL_FLAG(bool, clean, false,
          "If true, deletes the existing local model upfront to reclaim disk "
          "space before downloading (useful on storage-constrained devices).");

// Fallback & validation thresholds:
ABSL_FLAG(double, threshold, 0.80,
          "Change ratio threshold (0.0 to 1.0) above which full download is "
          "used instead of slice updates (default 0.80).");
ABSL_FLAG(bool, check_disk, true,
          "If true, validates available disk space before starting download.");

namespace {

litert::lm::sync::RangeFetcher CreateLocalFileRangeFetcher(
    absl::string_view source_path) {
  return [path = std::string(source_path)](
             uint64_t start_byte, uint64_t end_byte,
             std::ostream& out) -> absl::Status {
    if (end_byte < start_byte) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid byte range: end_byte (", end_byte,
                       ") < start_byte (", start_byte, ")"));
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return absl::NotFoundError(
          absl::StrCat("Remote source file not found: ", path));
    }
    in.seekg(start_byte);
    uint64_t bytes_to_read = (end_byte - start_byte) + 1;
    std::vector<char> buffer(64 * 1024);
    while (bytes_to_read > 0) {
      size_t chunk = std::min<uint64_t>(bytes_to_read, buffer.size());
      in.read(buffer.data(), chunk);
      std::streamsize actual = in.gcount();
      if (actual <= 0) break;
      out.write(buffer.data(), actual);
      bytes_to_read -= actual;
    }
    return absl::OkStatus();
  };
}

litert::lm::sync::RangeFetcher MakeRangeFetcher(absl::string_view source) {
  return litert::lm::sync::IsHttpUrl(source)
             ? litert::lm::sync::CreateHttpRangeFetcher(source)
             : CreateLocalFileRangeFetcher(source);
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "LiteRT-LM differential model synchronization utility.\n\n"
      "Usage:\n"
      "  model_sync_cli --remote=<upstream_model_or_url> "
      "[--local=<local_model>] [--output=<output_model>] [--diff] [--clean]");
  absl::ParseCommandLine(argc, argv);

  std::string remote_path = absl::GetFlag(FLAGS_remote);
  std::string local_path = absl::GetFlag(FLAGS_local);
  std::string output_path = absl::GetFlag(FLAGS_output);
  bool diff_only = absl::GetFlag(FLAGS_diff);

  if (remote_path.empty()) {
    std::cerr << "Error: --remote is required.\n";
    return 1;
  }

  if (output_path.empty() && !local_path.empty()) {
    output_path = local_path;  // In-place sync by default
  }

  if (!diff_only) {
    if (!local_path.empty() && litert::lm::sync::IsHttpUrl(local_path)) {
      std::cerr << "Error: --local must be a local file path when updating "
                   "without --diff.\n";
      return 1;
    }
    if (output_path.empty()) {
      std::cerr << "Error: --output is required when --local is empty.\n";
      return 1;
    }
    if (litert::lm::sync::IsHttpUrl(output_path)) {
      std::cerr << "Error: --output must be a local file path.\n";
      return 1;
    }
  }

  auto remote_fetcher = MakeRangeFetcher(remote_path);

  litert::lm::sync::SyncOptions options;
  options.clean = absl::GetFlag(FLAGS_clean);
  options.full_download_threshold = absl::GetFlag(FLAGS_threshold);
  options.check_disk_space = absl::GetFlag(FLAGS_check_disk);

  // 1. Inspect and show what changed between local and remote models.
  absl::StatusOr<litert::lm::sync::SyncPlan> plan_or =
      (!local_path.empty() && litert::lm::sync::IsHttpUrl(local_path))
          ? litert::lm::sync::ModelSync::PlanSync(MakeRangeFetcher(local_path),
                                                  remote_fetcher, options)
          : litert::lm::sync::ModelSync::PlanSync(local_path, remote_fetcher,
                                                  options);
  if (!plan_or.ok()) {
    std::cerr << "Failed to plan sync: " << plan_or.status().message() << "\n";
    return 1;
  }

  const auto& plan = *plan_or;
  std::cout << plan.Summary() << "\n";

  if (diff_only) {
    return 0;
  }

  if (plan.is_up_to_date && local_path == output_path) {
    std::cout << "Model is already up to date. No action needed.\n";
    return 0;
  }

  // 2. Perform differential sync.
  std::cout << "Applying differential sync to " << output_path << "...\n";
  absl::Status status = litert::lm::sync::ModelSync::Update(
      local_path, output_path, remote_fetcher, options);
  if (!status.ok()) {
    std::cerr << "Sync failed: " << status.message() << "\n";
    return 1;
  }

  std::cout << "Successfully synced model: " << output_path << "\n";
  return 0;
}
