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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "c/error_reporter.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "sync/test_utils.h"

namespace litert::lm::c {
namespace {

using ::litert::lm::sync::CreateTestModel;

struct CFileFetcherContext {
  std::string file_path;
  std::vector<std::pair<uint64_t, uint64_t>> fetched_ranges;
};

LiteRtLmStatusCode CFileRangeFetcher(uint64_t offset, uint64_t length,
                                     LiteRtLmModelSyncRangeWriter* writer,
                                     void* user_data) {
  auto* ctx = static_cast<CFileFetcherContext*>(user_data);
  if (ctx == nullptr) {
    litert_lm_model_sync_range_writer_set_error(writer,
                                                "Fetcher context is NULL.");
    return kLiteRtLmStatusInvalidArgument;
  }
  ctx->fetched_ranges.push_back({offset, length});

  std::ifstream in(ctx->file_path, std::ios::binary);
  if (!in) {
    litert_lm_model_sync_range_writer_set_error(
        writer, "Remote test file does not exist.");
    return kLiteRtLmStatusNotFound;
  }

  in.seekg(static_cast<std::streamoff>(offset));
  uint64_t bytes_to_read = length;
  std::vector<char> buf(4096);
  while (bytes_to_read > 0) {
    size_t n = std::min<uint64_t>(bytes_to_read, buf.size());
    in.read(buf.data(), n);
    std::streamsize actual = in.gcount();
    if (actual <= 0) break;
    LiteRtLmStatusCode write_status =
        litert_lm_model_sync_range_writer_write(writer, buf.data(), actual);
    if (write_status != kLiteRtLmStatusOk) {
      return write_status;
    }
    bytes_to_read -= static_cast<uint64_t>(actual);
  }
  return kLiteRtLmStatusOk;
}

std::string ReadFileBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

class ModelSyncCTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::path(testing::TempDir()) / "model_sync_c_test";
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  std::filesystem::path test_dir_;
};

TEST_F(ModelSyncCTest, OptionsCreateSetAndValidation) {
  LiteRtLmModelSyncOptions* raw_opts = nullptr;
  EXPECT_EQ(litert_lm_model_sync_options_create(nullptr),
            kLiteRtLmStatusInvalidArgument);
  litert_lm_model_sync_options_delete(nullptr);

  ASSERT_EQ(litert_lm_model_sync_options_create(&raw_opts), kLiteRtLmStatusOk);
  ASSERT_NE(raw_opts, nullptr);
  std::unique_ptr<LiteRtLmModelSyncOptions,
                  decltype(&litert_lm_model_sync_options_delete)>
      opts(raw_opts, &litert_lm_model_sync_options_delete);

  EXPECT_EQ(
      litert_lm_model_sync_options_set_full_download_threshold(opts.get(), 0.5),
      kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_model_sync_options_set_clean(opts.get(), true),
            kLiteRtLmStatusOk);
  EXPECT_EQ(
      litert_lm_model_sync_options_set_check_disk_space(opts.get(), false),
      kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_model_sync_options_set_allow_offline_fallback(opts.get(),
                                                                    false),
            kLiteRtLmStatusOk);

  // NULL handles and out-of-range thresholds return
  // kLiteRtLmStatusInvalidArgument.
  EXPECT_EQ(
      litert_lm_model_sync_options_set_full_download_threshold(nullptr, 0.5),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_model_sync_options_set_full_download_threshold(opts.get(),
                                                                     -0.1),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_model_sync_options_set_full_download_threshold(opts.get(), 1.1),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_model_sync_options_set_full_download_threshold(
                opts.get(), std::numeric_limits<double>::quiet_NaN()),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_model_sync_options_set_clean(nullptr, true),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_model_sync_options_set_check_disk_space(nullptr, false),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_model_sync_options_set_allow_offline_fallback(nullptr, false),
      kLiteRtLmStatusInvalidArgument);
}

TEST_F(ModelSyncCTest, InvalidArgumentsReturnError) {
  LiteRtLmModelSyncPlan* plan = reinterpret_cast<LiteRtLmModelSyncPlan*>(0x1);
  LiteRtLmStatusCode status = litert_lm_model_sync_plan_create_from_file(
      "/dummy/path", nullptr, nullptr, nullptr, &plan);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(plan, nullptr);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);

  EXPECT_EQ(litert_lm_model_sync_plan_create_from_file(
                "/dummy/path", CFileRangeFetcher, nullptr, nullptr, nullptr),
            kLiteRtLmStatusInvalidArgument);

  bool bool_sentinel = true;
  EXPECT_EQ(litert_lm_model_sync_plan_is_up_to_date(nullptr, &bool_sentinel),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_TRUE(bool_sentinel);

  uint64_t u64_sentinel = 123;
  EXPECT_EQ(
      litert_lm_model_sync_plan_get_bytes_to_download(nullptr, &u64_sentinel),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_model_sync_plan_get_bytes_to_reuse(nullptr, &u64_sentinel),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_model_sync_plan_get_total_remote_size(nullptr, &u64_sentinel),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(u64_sentinel, uint64_t{123});

  const char* out_str = "initial";
  EXPECT_EQ(litert_lm_model_sync_plan_format(nullptr, &out_str),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(out_str, nullptr);

  EXPECT_EQ(litert_lm_model_sync_update("/local", nullptr, CFileRangeFetcher,
                                        nullptr, nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_model_sync_update("/local", "", CFileRangeFetcher,
                                        nullptr, nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_model_sync_update("/local", "/out", nullptr, nullptr, nullptr),
      kLiteRtLmStatusInvalidArgument);

  LiteRtLmModelSyncResult* result =
      reinterpret_cast<LiteRtLmModelSyncResult*>(0x1);
  EXPECT_EQ(litert_lm_model_sync_cache_sync(nullptr, "id", CFileRangeFetcher,
                                            nullptr, nullptr, &result),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(result, nullptr);

  result = reinterpret_cast<LiteRtLmModelSyncResult*>(0x1);
  EXPECT_EQ(litert_lm_model_sync_cache_sync("", "id", CFileRangeFetcher,
                                            nullptr, nullptr, &result),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(result, nullptr);

  result = reinterpret_cast<LiteRtLmModelSyncResult*>(0x1);
  EXPECT_EQ(
      litert_lm_model_sync_cache_sync("/cache", nullptr, CFileRangeFetcher,
                                      nullptr, nullptr, &result),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(result, nullptr);

  result = reinterpret_cast<LiteRtLmModelSyncResult*>(0x1);
  EXPECT_EQ(litert_lm_model_sync_cache_sync("/cache", "", CFileRangeFetcher,
                                            nullptr, nullptr, &result),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(result, nullptr);

  result = reinterpret_cast<LiteRtLmModelSyncResult*>(0x1);
  EXPECT_EQ(litert_lm_model_sync_cache_sync("/cache", "id", nullptr, nullptr,
                                            nullptr, &result),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(result, nullptr);

  EXPECT_EQ(litert_lm_model_sync_cache_sync("/cache", "id", CFileRangeFetcher,
                                            nullptr, nullptr, nullptr),
            kLiteRtLmStatusInvalidArgument);

  const char* cached_path = "initial";
  EXPECT_EQ(litert_lm_model_sync_result_get_cached_path(nullptr, &cached_path),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(cached_path, nullptr);

  EXPECT_EQ(litert_lm_model_sync_range_writer_write(nullptr, "a", 1),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_model_sync_range_writer_set_error(nullptr, "err"),
            kLiteRtLmStatusInvalidArgument);

  litert_lm_model_sync_plan_delete(nullptr);
  litert_lm_model_sync_result_delete(nullptr);
}

TEST_F(ModelSyncCTest, RangeWriterErrorPropagationAndTlsPreservation) {
  // Record an existing thread-local error before a successful C API call and
  // confirm `AdaptFetcher` does not clear it.
  LiteRtLmModelSyncOptions* null_opts = nullptr;
  ASSERT_EQ(litert_lm_model_sync_options_set_clean(null_opts, true),
            kLiteRtLmStatusInvalidArgument);
  const char* prior_error = litert_lm_get_last_error_message();
  ASSERT_NE(prior_error, nullptr);
  const std::string expected_prior_error = prior_error;

  std::string remote_path = (test_dir_ / "tls_remote.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_tls", 4096}});
  CFileFetcherContext remote_ctx{remote_path, {}};
  LiteRtLmModelSyncPlan* plan = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_create_from_file(
                "", CFileRangeFetcher, &remote_ctx, nullptr, &plan),
            kLiteRtLmStatusOk);
  ASSERT_NE(plan, nullptr);
  litert_lm_model_sync_plan_delete(plan);

  const char* preserved_error = litert_lm_get_last_error_message();
  ASSERT_NE(preserved_error, nullptr);
  EXPECT_EQ(std::string(preserved_error), expected_prior_error);

  // Verify writer error message propagation and out-of-range status mapping.
  auto custom_error_fetcher = [](uint64_t, uint64_t,
                                 LiteRtLmModelSyncRangeWriter* writer,
                                 void*) -> LiteRtLmStatusCode {
    EXPECT_EQ(litert_lm_model_sync_range_writer_write(writer, nullptr, 0),
              kLiteRtLmStatusOk);
    EXPECT_EQ(litert_lm_model_sync_range_writer_write(writer, nullptr, 4),
              kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(litert_lm_model_sync_range_writer_set_error(
                  writer, "Custom upstream HTTP 503"),
              kLiteRtLmStatusOk);
    return kLiteRtLmStatusUnavailable;
  };
  LiteRtLmModelSyncPlan* failed_plan = nullptr;
  EXPECT_EQ(litert_lm_model_sync_plan_create_from_file(
                "", custom_error_fetcher, nullptr, nullptr, &failed_plan),
            kLiteRtLmStatusUnavailable);
  EXPECT_EQ(failed_plan, nullptr);
  ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_NE(std::string(litert_lm_get_last_error_message())
                .find("Custom upstream HTTP 503"),
            std::string::npos);

  // Use a value outside [kLiteRtLmStatusOk, kLiteRtLmStatusUnauthenticated]
  // (0..16) while remaining representable in the C++ enum bit-range (0..31).
  auto bogus_status_fetcher = [](uint64_t, uint64_t,
                                 LiteRtLmModelSyncRangeWriter*,
                                 void*) -> LiteRtLmStatusCode {
    return static_cast<LiteRtLmStatusCode>(20);
  };
  EXPECT_EQ(litert_lm_model_sync_plan_create_from_file(
                "", bogus_status_fetcher, nullptr, nullptr, &failed_plan),
            kLiteRtLmStatusUnknown);
  EXPECT_EQ(failed_plan, nullptr);
}

TEST_F(ModelSyncCTest, PlanSyncFromFileSuccess) {
  std::string local_path = (test_dir_ / "local.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "weights_payload_shared", 8192}});

  std::string remote_path = (test_dir_ / "remote.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_v2_updated", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "weights_payload_shared", 8192}});

  CFileFetcherContext remote_ctx{remote_path, {}};
  LiteRtLmModelSyncOptions* raw_opts = nullptr;
  ASSERT_EQ(litert_lm_model_sync_options_create(&raw_opts), kLiteRtLmStatusOk);
  std::unique_ptr<LiteRtLmModelSyncOptions,
                  decltype(&litert_lm_model_sync_options_delete)>
      opts(raw_opts, &litert_lm_model_sync_options_delete);

  LiteRtLmModelSyncPlan* raw_plan = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_create_from_file(
                local_path.c_str(), CFileRangeFetcher, &remote_ctx, opts.get(),
                &raw_plan),
            kLiteRtLmStatusOk);
  ASSERT_NE(raw_plan, nullptr);
  std::unique_ptr<LiteRtLmModelSyncPlan,
                  decltype(&litert_lm_model_sync_plan_delete)>
      plan(raw_plan, &litert_lm_model_sync_plan_delete);

  bool is_up_to_date = true;
  EXPECT_EQ(litert_lm_model_sync_plan_is_up_to_date(plan.get(), &is_up_to_date),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(is_up_to_date);

  uint64_t total_remote_size = 0;
  EXPECT_EQ(litert_lm_model_sync_plan_get_total_remote_size(plan.get(),
                                                            &total_remote_size),
            kLiteRtLmStatusOk);
  EXPECT_EQ(total_remote_size, std::filesystem::file_size(remote_path));

  uint64_t bytes_to_download = 0;
  EXPECT_EQ(litert_lm_model_sync_plan_get_bytes_to_download(plan.get(),
                                                            &bytes_to_download),
            kLiteRtLmStatusOk);
  EXPECT_GE(bytes_to_download, std::string("tokenizer_v2_updated").size());

  uint64_t bytes_to_reuse = 0;
  EXPECT_EQ(
      litert_lm_model_sync_plan_get_bytes_to_reuse(plan.get(), &bytes_to_reuse),
      kLiteRtLmStatusOk);
  EXPECT_EQ(bytes_to_reuse, std::string("weights_payload_shared").size());

  const char* formatted = nullptr;
  EXPECT_EQ(litert_lm_model_sync_plan_format(plan.get(), &formatted),
            kLiteRtLmStatusOk);
  ASSERT_NE(formatted, nullptr);
  EXPECT_NE(std::string(formatted).find("DIFFERENTIAL SLICE UPDATE"),
            std::string::npos);

  // Lowering full_download_threshold below the changed byte fraction triggers
  // a full download in the computed plan.
  ASSERT_EQ(litert_lm_model_sync_options_set_full_download_threshold(opts.get(),
                                                                     0.01),
            kLiteRtLmStatusOk);
  LiteRtLmModelSyncPlan* full_raw_plan = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_create_from_file(
                local_path.c_str(), CFileRangeFetcher, &remote_ctx, opts.get(),
                &full_raw_plan),
            kLiteRtLmStatusOk);
  std::unique_ptr<LiteRtLmModelSyncPlan,
                  decltype(&litert_lm_model_sync_plan_delete)>
      full_plan(full_raw_plan, &litert_lm_model_sync_plan_delete);
  const char* full_formatted = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_format(full_plan.get(), &full_formatted),
            kLiteRtLmStatusOk);
  EXPECT_NE(std::string(full_formatted).find("FULL"), std::string::npos);
}

TEST_F(ModelSyncCTest, FormatStringStaysValidAcrossCalls) {
  std::string local_path = (test_dir_ / "format_local.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_v1", 4096}});
  std::string remote_path = (test_dir_ / "format_remote.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_v2", 4096}});

  CFileFetcherContext remote_ctx{remote_path, {}};
  LiteRtLmModelSyncPlan* raw_plan = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_create_from_file(
                local_path.c_str(), CFileRangeFetcher, &remote_ctx, nullptr,
                &raw_plan),
            kLiteRtLmStatusOk);
  ASSERT_NE(raw_plan, nullptr);
  std::unique_ptr<LiteRtLmModelSyncPlan,
                  decltype(&litert_lm_model_sync_plan_delete)>
      plan(raw_plan, &litert_lm_model_sync_plan_delete);

  const char* first = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_format(plan.get(), &first),
            kLiteRtLmStatusOk);
  ASSERT_NE(first, nullptr);
  const std::string first_copy = first;

  const char* second = nullptr;
  ASSERT_EQ(litert_lm_model_sync_plan_format(plan.get(), &second),
            kLiteRtLmStatusOk);

  // The string is owned by the plan, so the one returned by the first call
  // must stay valid after further calls until the plan is deleted.
  ASSERT_EQ(second, first);
  EXPECT_EQ(std::string(first), first_copy);
  EXPECT_NE(first_copy.find("Sync Plan"), std::string::npos);
}

TEST_F(ModelSyncCTest, UpdateDifferentialSuccess) {
  std::string local_path = (test_dir_ / "local_update.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_old", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "heavy_weights_shared", 20480}});

  std::string remote_path = (test_dir_ / "remote_update.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_new", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "heavy_weights_shared", 20480}});

  std::string output_path = (test_dir_ / "synced.litertlm").string();
  CFileFetcherContext remote_ctx{remote_path, {}};

  LiteRtLmModelSyncOptions* raw_opts = nullptr;
  ASSERT_EQ(litert_lm_model_sync_options_create(&raw_opts), kLiteRtLmStatusOk);
  std::unique_ptr<LiteRtLmModelSyncOptions,
                  decltype(&litert_lm_model_sync_options_delete)>
      opts(raw_opts, &litert_lm_model_sync_options_delete);

  LiteRtLmStatusCode status =
      litert_lm_model_sync_update(local_path.c_str(), output_path.c_str(),
                                  CFileRangeFetcher, &remote_ctx, opts.get());
  ASSERT_EQ(status, kLiteRtLmStatusOk);
  ASSERT_TRUE(std::filesystem::exists(output_path));

  // Verify the assembled file matches the remote file byte-for-byte, and that
  // the reused weights section at offset 20480 was never fetched remotely.
  EXPECT_EQ(ReadFileBytes(output_path), ReadFileBytes(remote_path));
  for (const auto& range : remote_ctx.fetched_ranges) {
    EXPECT_LT(range.first + range.second, uint64_t{20480});
  }
}

TEST_F(ModelSyncCTest, CacheSyncSuccess) {
  std::string remote_path = (test_dir_ / "remote_cache.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_cache", 4096}});

  std::string cache_dir = (test_dir_ / "cache").string();
  CFileFetcherContext remote_ctx{remote_path, {}};

  LiteRtLmModelSyncOptions* raw_opts = nullptr;
  ASSERT_EQ(litert_lm_model_sync_options_create(&raw_opts), kLiteRtLmStatusOk);
  std::unique_ptr<LiteRtLmModelSyncOptions,
                  decltype(&litert_lm_model_sync_options_delete)>
      opts(raw_opts, &litert_lm_model_sync_options_delete);

  LiteRtLmModelSyncResult* raw_result = nullptr;
  LiteRtLmStatusCode status = litert_lm_model_sync_cache_sync(
      cache_dir.c_str(), "model-cache-test", CFileRangeFetcher, &remote_ctx,
      opts.get(), &raw_result);
  ASSERT_EQ(status, kLiteRtLmStatusOk);
  ASSERT_NE(raw_result, nullptr);
  std::unique_ptr<LiteRtLmModelSyncResult,
                  decltype(&litert_lm_model_sync_result_delete)>
      result(raw_result, &litert_lm_model_sync_result_delete);

  const char* cached_path = nullptr;
  EXPECT_EQ(
      litert_lm_model_sync_result_get_cached_path(result.get(), &cached_path),
      kLiteRtLmStatusOk);
  ASSERT_NE(cached_path, nullptr);
  EXPECT_TRUE(std::filesystem::exists(cached_path));
  EXPECT_EQ(ReadFileBytes(cached_path), ReadFileBytes(remote_path));
}

TEST_F(ModelSyncCTest, CacheSyncOfflineFallbackMatchesNullOptions) {
  std::string remote_path = (test_dir_ / "remote_offline.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_offline", 4096}});

  std::string cache_dir = (test_dir_ / "offline_cache").string();
  CFileFetcherContext remote_ctx{remote_path, {}};

  LiteRtLmModelSyncResult* raw_seeded = nullptr;
  ASSERT_EQ(litert_lm_model_sync_cache_sync(cache_dir.c_str(), "offline-model",
                                            CFileRangeFetcher, &remote_ctx,
                                            nullptr, &raw_seeded),
            kLiteRtLmStatusOk);
  ASSERT_NE(raw_seeded, nullptr);
  std::unique_ptr<LiteRtLmModelSyncResult,
                  decltype(&litert_lm_model_sync_result_delete)>
      seeded_result(raw_seeded, &litert_lm_model_sync_result_delete);
  const char* seeded_path = nullptr;
  ASSERT_EQ(litert_lm_model_sync_result_get_cached_path(seeded_result.get(),
                                                        &seeded_path),
            kLiteRtLmStatusOk);
  ASSERT_NE(seeded_path, nullptr);
  ASSERT_TRUE(std::filesystem::exists(seeded_path));

  auto unavailable_fetcher =
      [](uint64_t, uint64_t, LiteRtLmModelSyncRangeWriter*,
         void*) -> LiteRtLmStatusCode { return kLiteRtLmStatusUnavailable; };

  // Both NULL options and default-created options should fall back to the
  // existing cached model on kLiteRtLmStatusUnavailable.
  LiteRtLmModelSyncResult* raw_null_opts_res = nullptr;
  EXPECT_EQ(litert_lm_model_sync_cache_sync(cache_dir.c_str(), "offline-model",
                                            unavailable_fetcher, nullptr,
                                            nullptr, &raw_null_opts_res),
            kLiteRtLmStatusOk);
  ASSERT_NE(raw_null_opts_res, nullptr);
  std::unique_ptr<LiteRtLmModelSyncResult,
                  decltype(&litert_lm_model_sync_result_delete)>
      null_opts_res(raw_null_opts_res, &litert_lm_model_sync_result_delete);
  const char* null_opts_path = nullptr;
  ASSERT_EQ(litert_lm_model_sync_result_get_cached_path(null_opts_res.get(),
                                                        &null_opts_path),
            kLiteRtLmStatusOk);
  EXPECT_STREQ(null_opts_path, seeded_path);

  LiteRtLmModelSyncOptions* raw_default_opts = nullptr;
  ASSERT_EQ(litert_lm_model_sync_options_create(&raw_default_opts),
            kLiteRtLmStatusOk);
  std::unique_ptr<LiteRtLmModelSyncOptions,
                  decltype(&litert_lm_model_sync_options_delete)>
      default_opts(raw_default_opts, &litert_lm_model_sync_options_delete);
  LiteRtLmModelSyncResult* raw_default_opts_res = nullptr;
  EXPECT_EQ(litert_lm_model_sync_cache_sync(
                cache_dir.c_str(), "offline-model", unavailable_fetcher,
                nullptr, default_opts.get(), &raw_default_opts_res),
            kLiteRtLmStatusOk);
  ASSERT_NE(raw_default_opts_res, nullptr);
  std::unique_ptr<LiteRtLmModelSyncResult,
                  decltype(&litert_lm_model_sync_result_delete)>
      default_opts_res(raw_default_opts_res,
                       &litert_lm_model_sync_result_delete);
  const char* default_opts_path = nullptr;
  ASSERT_EQ(litert_lm_model_sync_result_get_cached_path(default_opts_res.get(),
                                                        &default_opts_path),
            kLiteRtLmStatusOk);
  EXPECT_STREQ(default_opts_path, seeded_path);

  // Explicitly disabling offline fallback propagates the Unavailable error and
  // sets *out_result to NULL.
  ASSERT_EQ(litert_lm_model_sync_options_set_allow_offline_fallback(
                default_opts.get(), false),
            kLiteRtLmStatusOk);
  LiteRtLmModelSyncResult* failed_res =
      reinterpret_cast<LiteRtLmModelSyncResult*>(0x1);
  EXPECT_EQ(litert_lm_model_sync_cache_sync(cache_dir.c_str(), "offline-model",
                                            unavailable_fetcher, nullptr,
                                            default_opts.get(), &failed_res),
            kLiteRtLmStatusUnavailable);
  EXPECT_EQ(failed_res, nullptr);
}

}  // namespace
}  // namespace litert::lm::c
