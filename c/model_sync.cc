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

#include "c/model_sync.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "sync/model_sync.h"

struct LiteRtLmModelSyncOptions {
  litert::lm::sync::SyncOptions options;
};

struct LiteRtLmModelSyncPlan {
  explicit LiteRtLmModelSyncPlan(const litert::lm::sync::SyncPlan& p)
      : formatted_summary(p.Summary()),
        is_up_to_date(p.is_up_to_date),
        total_remote_bytes(p.total_remote_bytes),
        bytes_to_download(p.bytes_to_download),
        bytes_to_reuse(p.bytes_to_reuse) {}

  // Computed once at construction and never modified afterwards so pointers
  // returned by `litert_lm_model_sync_plan_format` stay valid and thread-safe
  // for the lifetime of the plan.
  std::string formatted_summary;
  bool is_up_to_date = false;
  uint64_t total_remote_bytes = 0;
  uint64_t bytes_to_download = 0;
  uint64_t bytes_to_reuse = 0;
};

struct LiteRtLmModelSyncRangeWriter {
  std::ostream& out;
  std::string error_message;
};

struct LiteRtLmModelSyncResult {
  std::string cached_path;
};

namespace {

using ::litert::lm::c::ReturnError;

absl::Status ToAbslErrorStatus(int raw_code, absl::string_view message) {
  absl::StatusCode absl_code = (raw_code > kLiteRtLmStatusOk &&
                                raw_code <= kLiteRtLmStatusUnauthenticated)
                                   ? static_cast<absl::StatusCode>(raw_code)
                                   : absl::StatusCode::kUnknown;
  return absl::Status(absl_code,
                      message.empty() ? "Range fetch failed" : message);
}

litert::lm::sync::RangeFetcher AdaptFetcher(
    LiteRtLmModelSyncRangeFetcherCallback c_fetcher, void* user_data) {
  if (c_fetcher == nullptr) {
    return nullptr;
  }
  return [c_fetcher, user_data](uint64_t start_byte, uint64_t end_byte,
                                std::ostream& out) -> absl::Status {
    uint64_t length = 0;
    if (end_byte >= start_byte) {
      const uint64_t diff = end_byte - start_byte;
      if (diff == std::numeric_limits<uint64_t>::max()) {
        return absl::InvalidArgumentError(
            "Byte range length overflows uint64_t.");
      }
      length = diff + 1;
    }

    LiteRtLmModelSyncRangeWriter writer{out, {}};
    LiteRtLmStatusCode code = c_fetcher(start_byte, length, &writer, user_data);
    static_assert(sizeof(int) == sizeof(LiteRtLmStatusCode));
    int raw_code = 0;
    std::memcpy(&raw_code, &code, sizeof(raw_code));
    if (raw_code != kLiteRtLmStatusOk) {
      return ToAbslErrorStatus(raw_code, writer.error_message);
    }
    if (!out) {
      return absl::DataLossError("Failed to write fetched data to stream.");
    }
    return absl::OkStatus();
  };
}

}  // namespace

extern "C" {

LiteRtLmStatusCode litert_lm_model_sync_range_writer_write(
    LiteRtLmModelSyncRangeWriter* writer, const void* data, size_t size) {
  LITERT_LM_C_RETURN_IF_NULL(writer);
  if (size == 0) {
    return kLiteRtLmStatusOk;
  }
  LITERT_LM_C_RETURN_IF_NULL(data);
  writer->out.write(static_cast<const char*>(data),
                    static_cast<std::streamsize>(size));
  if (!writer->out) {
    return ReturnError(absl::StatusCode::kDataLoss,
                       "Failed to write fetched chunk to stream.");
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_range_writer_set_error(
    LiteRtLmModelSyncRangeWriter* writer, const char* message) {
  LITERT_LM_C_RETURN_IF_NULL(writer);
  writer->error_message = message != nullptr ? message : "";
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_options_create(
    LiteRtLmModelSyncOptions** out_options) {
  LITERT_LM_C_RETURN_IF_NULL(out_options);
  *out_options = new LiteRtLmModelSyncOptions();
  return kLiteRtLmStatusOk;
}

void litert_lm_model_sync_options_delete(LiteRtLmModelSyncOptions* options) {
  delete options;
}

LiteRtLmStatusCode litert_lm_model_sync_options_set_full_download_threshold(
    LiteRtLmModelSyncOptions* options, double threshold) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  if (std::isnan(threshold) || threshold < 0.0 || threshold > 1.0) {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "full_download_threshold must be within [0.0, 1.0].");
  }
  options->options.full_download_threshold = threshold;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_options_set_clean(
    LiteRtLmModelSyncOptions* options, bool clean) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  options->options.clean = clean;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_options_set_check_disk_space(
    LiteRtLmModelSyncOptions* options, bool check_disk_space) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  options->options.check_disk_space = check_disk_space;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_options_set_allow_offline_fallback(
    LiteRtLmModelSyncOptions* options, bool allow_offline_fallback) {
  LITERT_LM_C_RETURN_IF_NULL(options);
  options->options.allow_offline_fallback = allow_offline_fallback;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_plan_create_from_file(
    const char* local_model_path,
    LiteRtLmModelSyncRangeFetcherCallback remote_fetcher, void* user_data,
    const LiteRtLmModelSyncOptions* options, LiteRtLmModelSyncPlan** out_plan) {
  LITERT_LM_C_RETURN_IF_NULL(out_plan);
  *out_plan = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(remote_fetcher);

  litert::lm::sync::RangeFetcher fetcher =
      AdaptFetcher(remote_fetcher, user_data);
  std::string local_path = local_model_path != nullptr ? local_model_path : "";
  const litert::lm::sync::SyncOptions cpp_opts =
      options != nullptr ? options->options : litert::lm::sync::SyncOptions{};

  LITERT_LM_C_ASSIGN_OR_RETURN(litert::lm::sync::SyncPlan plan,
                               litert::lm::sync::ModelSync::PlanSync(
                                   local_path, std::move(fetcher), cpp_opts));

  *out_plan = new LiteRtLmModelSyncPlan(plan);
  return kLiteRtLmStatusOk;
}

void litert_lm_model_sync_plan_delete(LiteRtLmModelSyncPlan* plan) {
  delete plan;
}

LiteRtLmStatusCode litert_lm_model_sync_plan_is_up_to_date(
    const LiteRtLmModelSyncPlan* plan, bool* out_is_up_to_date) {
  LITERT_LM_C_RETURN_IF_NULL(out_is_up_to_date);
  LITERT_LM_C_RETURN_IF_NULL(plan);
  *out_is_up_to_date = plan->is_up_to_date;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_plan_get_bytes_to_download(
    const LiteRtLmModelSyncPlan* plan, uint64_t* out_bytes) {
  LITERT_LM_C_RETURN_IF_NULL(out_bytes);
  LITERT_LM_C_RETURN_IF_NULL(plan);
  *out_bytes = plan->bytes_to_download;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_plan_get_bytes_to_reuse(
    const LiteRtLmModelSyncPlan* plan, uint64_t* out_bytes) {
  LITERT_LM_C_RETURN_IF_NULL(out_bytes);
  LITERT_LM_C_RETURN_IF_NULL(plan);
  *out_bytes = plan->bytes_to_reuse;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_plan_get_total_remote_size(
    const LiteRtLmModelSyncPlan* plan, uint64_t* out_bytes) {
  LITERT_LM_C_RETURN_IF_NULL(out_bytes);
  LITERT_LM_C_RETURN_IF_NULL(plan);
  *out_bytes = plan->total_remote_bytes;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_plan_format(
    const LiteRtLmModelSyncPlan* plan, const char** out_formatted_string) {
  LITERT_LM_C_RETURN_IF_NULL(out_formatted_string);
  *out_formatted_string = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(plan);
  *out_formatted_string = plan->formatted_summary.c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_update(
    const char* local_model_path, const char* output_model_path,
    LiteRtLmModelSyncRangeFetcherCallback remote_fetcher, void* user_data,
    const LiteRtLmModelSyncOptions* options) {
  LITERT_LM_C_RETURN_IF_NULL(output_model_path);
  if (output_model_path[0] == '\0') {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "output_model_path must not be empty.");
  }
  LITERT_LM_C_RETURN_IF_NULL(remote_fetcher);

  std::string local_path = local_model_path != nullptr ? local_model_path : "";
  litert::lm::sync::RangeFetcher fetcher =
      AdaptFetcher(remote_fetcher, user_data);
  const litert::lm::sync::SyncOptions cpp_opts =
      options != nullptr ? options->options : litert::lm::sync::SyncOptions{};

  LITERT_LM_C_RETURN_IF_ERROR(litert::lm::sync::ModelSync::Update(
      local_path, output_model_path, std::move(fetcher), cpp_opts));
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_model_sync_cache_sync(
    const char* cache_dir, const char* model_id,
    LiteRtLmModelSyncRangeFetcherCallback remote_fetcher, void* user_data,
    const LiteRtLmModelSyncOptions* options,
    LiteRtLmModelSyncResult** out_result) {
  LITERT_LM_C_RETURN_IF_NULL(out_result);
  *out_result = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(cache_dir);
  if (cache_dir[0] == '\0') {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "cache_dir must not be empty.");
  }
  LITERT_LM_C_RETURN_IF_NULL(model_id);
  if (model_id[0] == '\0') {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "model_id must not be empty.");
  }
  LITERT_LM_C_RETURN_IF_NULL(remote_fetcher);

  litert::lm::sync::RangeFetcher fetcher =
      AdaptFetcher(remote_fetcher, user_data);
  const litert::lm::sync::SyncOptions cpp_opts =
      options != nullptr ? options->options : litert::lm::sync::SyncOptions{};

  LITERT_LM_C_ASSIGN_OR_RETURN(
      std::string cached_path,
      litert::lm::sync::ModelSync::Sync(cache_dir, model_id, std::move(fetcher),
                                        cpp_opts));

  *out_result = new LiteRtLmModelSyncResult{std::move(cached_path)};
  return kLiteRtLmStatusOk;
}

void litert_lm_model_sync_result_delete(LiteRtLmModelSyncResult* result) {
  delete result;
}

LiteRtLmStatusCode litert_lm_model_sync_result_get_cached_path(
    const LiteRtLmModelSyncResult* result, const char** out_cached_path) {
  LITERT_LM_C_RETURN_IF_NULL(out_cached_path);
  *out_cached_path = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(result);
  *out_cached_path = result->cached_path.c_str();
  return kLiteRtLmStatusOk;
}

}  // extern "C"
