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
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if !defined(_WIN32) && !defined(__ANDROID__) && !defined(__EMSCRIPTEN__) && \
    (!defined(__APPLE__) || TARGET_OS_OSX)
#include <sys/stat.h>
#define LITERT_LM_HAS_POSIX_CURL_TEST 1
#else
#define LITERT_LM_HAS_POSIX_CURL_TEST 0
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "support/util/test_utils.h"  // NOLINT
#include "sync/sha256.h"
#include "sync/test_utils.h"

namespace litert::lm::sync {
namespace {

using ::testing::IsEmpty;
using ::testing::status::StatusIs;

RangeFetcher CreateFileRangeFetcher(const std::string& remote_file_path,
                                    std::vector<ByteRange>* logged_ranges) {
  return [remote_file_path, logged_ranges](
             uint64_t start_byte, uint64_t end_byte,
             std::ostream& output_stream) -> absl::Status {
    if (logged_ranges) {
      logged_ranges->push_back(ByteRange{start_byte, end_byte});
    }

    std::ifstream in(remote_file_path, std::ios::binary);
    if (!in) {
      return absl::NotFoundError(
          absl::StrCat("Remote file not found: ", remote_file_path));
    }

    in.seekg(start_byte);
    uint64_t bytes_to_read = (end_byte - start_byte) + 1;
    std::vector<char> buf(4096);
    while (bytes_to_read > 0) {
      size_t n = std::min<uint64_t>(bytes_to_read, buf.size());
      in.read(buf.data(), n);
      std::streamsize actual = in.gcount();
      if (actual <= 0) break;
      output_stream.write(buf.data(), actual);
      bytes_to_read -= actual;
    }
    return absl::OkStatus();
  };
}

TEST(Sha256Test, StandardVectors) {
  // Empty string
  EXPECT_EQ(ComputeSha256Hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  // "abc"
  EXPECT_EQ(ComputeSha256Hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // 56-byte message (spans across block boundary during padding)
  EXPECT_EQ(ComputeSha256Hex(
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // 55-byte boundary (largest message fitting in single 64-byte block with pad)
  std::string msg55(55, 'A');
  EXPECT_EQ(ComputeSha256Hex(msg55),
            "8963cc0afd622cc7574ac2011f93a3059b3d65548a77542a1559e3d202e6ab00");
  // 64-byte boundary (exact one block before padding)
  std::string msg64(64, 'B');
  EXPECT_EQ(ComputeSha256Hex(msg64),
            "c422e7070cb1cb455b5de9afee0d975e303d0239c72030cd7414ab5c382d3ae8");
  // 1,000,000 'a' characters (NIST CAVP)
  std::string one_million_a(1000000, 'a');
  EXPECT_EQ(ComputeSha256Hex(one_million_a),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, IncrementalStreamingMatchesSingleCall) {
  std::string data = "The quick brown fox jumps over the lazy dog";
  Sha256Hasher hasher;
  for (char c : data) {
    hasher.Update(&c, 1);
  }
  EXPECT_EQ(hasher.FinalHex(), ComputeSha256Hex(data));
}

TEST(Sha256Test, FinalResetsStateForReuse) {
  Sha256Hasher hasher;
  hasher.Update("abc", 3);
  EXPECT_EQ(hasher.FinalHex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // Hasher was reset by Final/FinalHex, second call hashes empty string.
  EXPECT_EQ(hasher.FinalHex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(Sha256Test, MultiBlockChunksStreaming) {
  std::string data(500, 'x');
  Sha256Hasher hasher;
  hasher.Update(data.data(), 100);
  hasher.Update(data.data() + 100, 250);
  hasher.Update(data.data() + 350, 150);
  EXPECT_EQ(hasher.FinalHex(), ComputeSha256Hex(data));
}

class ModelSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::path(testing::TempDir()) / "model_sync_test";
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  std::filesystem::path test_dir_;
};

TEST_F(ModelSyncTest, PlanSync_ReturnsUpToDateWhenIdentical) {
  std::string model_path = (test_dir_ / "model_v1.litertlm").string();
  CreateTestModel(model_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_bytes", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "prefill_decode", "weight_bytes_12345", 8192}});

  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(model_path, &logged_ranges);

  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(model_path, fetcher));

  EXPECT_TRUE(plan.is_up_to_date);
  EXPECT_GT(plan.bytes_to_reuse, 0);
  EXPECT_EQ(plan.bytes_to_download, 0);
  EXPECT_THAT(plan.required_ranges, IsEmpty());
}

TEST_F(ModelSyncTest, PlanSync_IdentifiesModifiedSection) {
  std::string local_path = (test_dir_ / "model_v1.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_bytes_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "prefill_decode", "weight_bytes_2gb", 8192}});

  std::string remote_path = (test_dir_ / "model_v2.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_bytes_v2_NEW", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "prefill_decode", "weight_bytes_2gb", 8192}});

  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(local_path, fetcher));

  EXPECT_FALSE(plan.is_up_to_date);
  EXPECT_EQ(plan.bytes_to_reuse, std::string("weight_bytes_2gb").size());
  ASSERT_EQ(plan.required_ranges.size(), 1);
  EXPECT_EQ(plan.required_ranges[0].start_byte, 4096);
  ASSERT_EQ(plan.section_diffs.size(), 2);
  EXPECT_EQ(plan.section_diffs[0].action, SectionAction::kDownloadDiff);
  EXPECT_EQ(plan.section_diffs[1].action, SectionAction::kReuseLocal);
}

TEST_F(ModelSyncTest, Update_ReplacesOnlyChangedSectionAndPreservesWeights) {
  std::string local_path = (test_dir_ / "model_v1.litertlm").string();
  CreateTestModel(
      local_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_v1", 4096},
       {schema::AnySectionDataType_TFLiteModel, "prefill_decode",
        "heavy_weights_payload_identical", 8192}});

  std::string remote_path = (test_dir_ / "model_v2.litertlm").string();
  CreateTestModel(remote_path,
                  {{schema::AnySectionDataType_SP_Tokenizer, "",
                    "tokenizer_v2_UPDATED_PROMPT", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "prefill_decode",
                    "heavy_weights_payload_identical", 8192}});

  std::string output_path = (test_dir_ / "model_updated.litertlm").string();
  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  // Verify only header and section 0 were fetched over network.
  for (const auto& range : logged_ranges) {
    EXPECT_LT(range.start_byte, 8192);
  }

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(output_path, &updated_header));
  auto* objects = updated_header.metadata->section_metadata()->objects();
  ASSERT_EQ(objects->size(), 2);

  // Check section 0 has updated payload.
  std::ifstream updated_file(output_path, std::ios::binary);
  std::string sec0_payload(
      objects->Get(0)->end_offset() - objects->Get(0)->begin_offset(), '\0');
  updated_file.seekg(objects->Get(0)->begin_offset());
  updated_file.read(sec0_payload.data(), sec0_payload.size());
  EXPECT_EQ(sec0_payload, "tokenizer_v2_UPDATED_PROMPT");

  // Check section 1 has reused local weights.
  std::string sec1_payload(
      objects->Get(1)->end_offset() - objects->Get(1)->begin_offset(), '\0');
  updated_file.seekg(objects->Get(1)->begin_offset());
  updated_file.read(sec1_payload.data(), sec1_payload.size());
  EXPECT_EQ(sec1_payload, "heavy_weights_payload_identical");
}

TEST_F(ModelSyncTest, Update_MultimodalModel_PatchesOnlyUpdatedVisionEncoder) {
  std::string local_path = (test_dir_ / "multimodal_v1.litertlm").string();
  CreateTestModel(local_path,
                  {{schema::AnySectionDataType_SP_Tokenizer, "",
                    "sp_tokenizer_bytes", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "vision_encoder",
                    "vision_encoder_v1_weights_100mb", 8192},
                   {schema::AnySectionDataType_TFLiteModel, "text_decoder",
                    "text_decoder_weights_2gb_large", 16384},
                   {schema::AnySectionDataType_GenericBinaryData, "metadata_pb",
                    "metadata_proto_payload", 24576}});

  std::string remote_path = (test_dir_ / "multimodal_v2.litertlm").string();
  CreateTestModel(remote_path,
                  {{schema::AnySectionDataType_SP_Tokenizer, "",
                    "sp_tokenizer_bytes", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "vision_encoder",
                    "vision_encoder_v2_NEW_weights_100mb", 8192},
                   {schema::AnySectionDataType_TFLiteModel, "text_decoder",
                    "text_decoder_weights_2gb_large", 16384},
                   {schema::AnySectionDataType_GenericBinaryData, "metadata_pb",
                    "metadata_proto_payload", 24576}});

  std::string output_path =
      (test_dir_ / "multimodal_updated.litertlm").string();
  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  // Only header and vision encoder at 8192 were downloaded.
  for (const auto& r : logged_ranges) {
    if (r.start_byte >= 4096) {
      EXPECT_EQ(r.start_byte, 8192);
    }
  }

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(output_path, &updated_header));
  auto* objs = updated_header.metadata->section_metadata()->objects();
  ASSERT_EQ(objs->size(), 4);
}

TEST_F(ModelSyncTest,
       Update_MultipleSectionsSameDataType_DistinguishesAndReusesProperly) {
  std::string local_path = (test_dir_ / "multi_tflite_v1.litertlm").string();
  CreateTestModel(local_path,
                  {{schema::AnySectionDataType_TFLiteModel, "subgraph_a",
                    "subgraph_a_payload_unmodified", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "subgraph_b",
                    "subgraph_b_v1_payload", 8192}});

  std::string remote_path = (test_dir_ / "multi_tflite_v2.litertlm").string();
  CreateTestModel(remote_path,
                  {{schema::AnySectionDataType_TFLiteModel, "subgraph_a",
                    "subgraph_a_payload_unmodified", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "subgraph_b",
                    "subgraph_b_v2_UPDATED_payload", 8192}});

  std::string output_path = (test_dir_ / "multi_tflite_out.litertlm").string();
  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  for (const auto& r : logged_ranges) {
    if (r.start_byte >= 4096) {
      EXPECT_EQ(r.start_byte, 8192);
    }
  }
}

TEST_F(ModelSyncTest, Update_HandlesAddedAndRemovedSections) {
  std::string local_path = (test_dir_ / "local_2sec.litertlm").string();
  CreateTestModel(
      local_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_bytes", 4096},
       {schema::AnySectionDataType_TFLiteModel, "core", "core_weights", 8192}});

  // Remote adds a 3rd section (lora adapter).
  std::string remote_path = (test_dir_ / "remote_3sec.litertlm").string();
  CreateTestModel(
      remote_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_bytes", 4096},
       {schema::AnySectionDataType_TFLiteModel, "core", "core_weights", 8192},
       {schema::AnySectionDataType_GenericBinaryData, "lora_adapter",
        "lora_adapter_weights", 16384}});

  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(local_path, fetcher));
  EXPECT_FALSE(plan.is_up_to_date);
  ASSERT_EQ(plan.required_ranges.size(), 1);
  EXPECT_EQ(plan.required_ranges[0].start_byte, 16384);

  std::string output_path = (test_dir_ / "output_3sec.litertlm").string();
  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(output_path, &updated_header));
  EXPECT_EQ(updated_header.metadata->section_metadata()->objects()->size(), 3);
}

TEST_F(ModelSyncTest, Update_ReorderedSections_ReusesBothPayloads) {
  std::string local_path = (test_dir_ / "local_ab.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer,
                                "vocab_a", "payload_aaa", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights_b", "payload_bbb", 8192}});

  std::string remote_path = (test_dir_ / "remote_ba.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_TFLiteModel,
                                 "weights_b", "payload_bbb", 4096},
                                {schema::AnySectionDataType_SP_Tokenizer,
                                 "vocab_a", "payload_aaa", 8192}});

  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(local_path, fetcher));
  EXPECT_FALSE(plan.is_up_to_date);
  EXPECT_THAT(plan.required_ranges, IsEmpty());

  std::string output_path = (test_dir_ / "output_ba.litertlm").string();
  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  for (const auto& r : logged_ranges) {
    EXPECT_LT(r.start_byte, 4096);
  }
}

TEST_F(ModelSyncTest, Update_ZeroByteSection_HandledSafely) {
  std::string local_path = (test_dir_ / "local_zero.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_GenericBinaryData,
                                "empty_cfg", "", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "weights_123", 8192}});

  std::string remote_path = (test_dir_ / "remote_zero.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_GenericBinaryData,
                                 "empty_cfg", "", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "weights_123", 8192}});

  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(local_path, fetcher));
  EXPECT_TRUE(plan.is_up_to_date);

  std::string output_path = (test_dir_ / "output_zero.litertlm").string();
  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));
  EXPECT_TRUE(std::filesystem::exists(output_path));
}

TEST_F(ModelSyncTest, Update_VerificationFailsOnCorruptedDownload) {
  std::string local_path = (test_dir_ / "model_v1.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", std::string(1000, 'W'), 8192}});

  std::string remote_path = (test_dir_ / "model_v2.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_v2", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", std::string(1000, 'W'), 8192}});

  std::string output_path = (test_dir_ / "model_corrupted.litertlm").string();

  auto corrupted_fetcher = [&remote_path](uint64_t start_byte,
                                          uint64_t end_byte,
                                          std::ostream& out) -> absl::Status {
    if (start_byte >= 4096) {
      out.write("CORRUPTED_BYTES", 15);
      return absl::OkStatus();
    }
    std::ifstream in(remote_path, std::ios::binary);
    in.seekg(start_byte);
    std::vector<char> buf((end_byte - start_byte) + 1);
    in.read(buf.data(), buf.size());
    out.write(buf.data(), buf.size());
    return absl::OkStatus();
  };

  absl::Status status =
      ModelSync::Update(local_path, output_path, corrupted_fetcher);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_FALSE(std::filesystem::exists(output_path));
}

TEST_F(ModelSyncTest, Update_NetworkDropMidStream_LeavesOriginalIntact) {
  std::string model_path = (test_dir_ / "my_model.litertlm").string();
  CreateTestModel(model_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "old_tokenizer", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "prefill_decode", "weights_12345", 8192}});

  std::string remote_path = (test_dir_ / "remote_model.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "new_tokenizer_updated", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "prefill_decode", "weights_12345", 8192}});

  std::ifstream orig_in(model_path, std::ios::binary);
  std::string original_content((std::istreambuf_iterator<char>(orig_in)),
                               std::istreambuf_iterator<char>());
  orig_in.close();

  auto dropping_fetcher = [&remote_path](uint64_t start_byte, uint64_t end_byte,
                                         std::ostream& out) -> absl::Status {
    if (start_byte >= 4096) {
      out.write("abcde", 5);
      return absl::UnavailableError("Network dropped mid-transfer");
    }
    std::ifstream in(remote_path, std::ios::binary);
    in.seekg(start_byte);
    std::vector<char> buf((end_byte - start_byte) + 1);
    in.read(buf.data(), buf.size());
    out.write(buf.data(), buf.size());
    return absl::OkStatus();
  };

  absl::Status status =
      ModelSync::Update(model_path, model_path, dropping_fetcher);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_FALSE(std::filesystem::exists(model_path + ".tmp_updating"));

  std::ifstream check_in(model_path, std::ios::binary);
  std::string current_content((std::istreambuf_iterator<char>(check_in)),
                              std::istreambuf_iterator<char>());
  check_in.close();
  EXPECT_EQ(current_content, original_content);
}

TEST_F(ModelSyncTest, Sync_FirstTimeAndCached_HandlesOfflineFallback) {
  std::string remote_path = (test_dir_ / "remote.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_bytes", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "prefill_decode", "weights_bytes", 8192}});

  std::string cache_dir = (test_dir_ / "model_cache").string();

  // First sync downloads and caches model.
  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);
  ASSERT_OK_AND_ASSIGN(std::string cached_path,
                       ModelSync::Sync(cache_dir, "gemma-3-1b-it", fetcher));
  EXPECT_TRUE(std::filesystem::exists(cached_path));

  // Second sync when offline falls back to cached model cleanly.
  auto offline_fetcher = [](uint64_t start_byte, uint64_t end_byte,
                            std::ostream& out) -> absl::Status {
    return absl::UnavailableError("Offline mode");
  };

  ASSERT_OK_AND_ASSIGN(
      std::string offline_path,
      ModelSync::Sync(cache_dir, "gemma-3-1b-it", offline_fetcher));
  EXPECT_EQ(offline_path, cached_path);
}

TEST_F(ModelSyncTest,
       Update_OutputNotWritable_FailsCleanlyWithoutModifyingSource) {
  std::string local_path = (test_dir_ / "source.litertlm").string();
  CreateTestModel(
      local_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tok_v1", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "weights_1", 8192}});

  std::string remote_path = (test_dir_ / "remote.litertlm").string();
  CreateTestModel(
      remote_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tok_v2", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "weights_1", 8192}});

  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);

  // Non-writable target directory.
  std::string non_writable_output =
      (test_dir_ / "non_existent_subdir" / "nested" / "out.litertlm").string();

  absl::Status status =
      ModelSync::Update(local_path, non_writable_output, fetcher);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(std::filesystem::exists(local_path));
}

TEST_F(ModelSyncTest, PlanSync_LegacyLocalModel_MissingHashes_DownloadsAll) {
  std::string local_path = (test_dir_ / "legacy_local.litertlm").string();
  CreateTestModel(
      local_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_payload", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "weights_payload",
        8192}},
      /*include_sha256=*/false);

  std::string remote_path = (test_dir_ / "modern_remote.litertlm").string();
  CreateTestModel(
      remote_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_payload", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "weights_payload",
        8192}},
      /*include_sha256=*/true);

  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);
  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(local_path, fetcher));

  // Without local hashes, cannot verify equivalence; plans download of all.
  EXPECT_FALSE(plan.is_up_to_date);
  EXPECT_EQ(plan.bytes_to_reuse, 0);
  EXPECT_GT(plan.bytes_to_download, 0);
}

TEST_F(ModelSyncTest, PlanSync_LegacyRemoteModel_MissingHashes_DownloadsAll) {
  std::string local_path = (test_dir_ / "modern_local.litertlm").string();
  CreateTestModel(
      local_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_payload", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "weights_payload",
        8192}},
      /*include_sha256=*/true);

  std::string remote_path = (test_dir_ / "legacy_remote.litertlm").string();
  CreateTestModel(
      remote_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "tokenizer_payload", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "weights_payload",
        8192}},
      /*include_sha256=*/false);

  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);
  ASSERT_OK_AND_ASSIGN(SyncPlan plan, ModelSync::PlanSync(local_path, fetcher));

  EXPECT_FALSE(plan.is_up_to_date);
  EXPECT_EQ(plan.bytes_to_reuse, 0);
  EXPECT_GT(plan.bytes_to_download, 0);
}

TEST_F(ModelSyncTest, PlanSync_ChangesExceed80Percent_TriggersFullDownload) {
  std::string local_path = (test_dir_ / "local_small.litertlm").string();
  CreateTestModel(local_path,
                  {{schema::AnySectionDataType_SP_Tokenizer, "", "tok", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "weights",
                    std::string(1000, 'A'), 8192}});

  std::string remote_path = (test_dir_ / "remote_big_change.litertlm").string();
  CreateTestModel(remote_path,
                  {{schema::AnySectionDataType_SP_Tokenizer, "", "tok", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "weights",
                    std::string(1000, 'B'), 8192}});

  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);
  SyncOptions options;
  options.full_download_threshold = 0.80;

  ASSERT_OK_AND_ASSIGN(SyncPlan plan,
                       ModelSync::PlanSync(local_path, fetcher, options));

  // Weights changed (>80% of model), triggers full download mode.
  EXPECT_FALSE(plan.is_up_to_date);
  EXPECT_TRUE(plan.use_full_download);
  EXPECT_EQ(plan.bytes_to_reuse, 0);
  EXPECT_EQ(plan.bytes_to_download, plan.total_remote_bytes);
}

TEST_F(ModelSyncTest,
       Update_GrowingEarlierSection_ShiftsOffsetsAndReusesWeights) {
  std::string local_path = (test_dir_ / "local_shift.litertlm").string();
  // 20 KB weights payload so changed ratio is ~45% (< 80% full download limit).
  std::string heavy_weights(20000, 'W');
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tok_short", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", heavy_weights, 8192}});

  std::string remote_path = (test_dir_ / "remote_shift.litertlm").string();
  CreateTestModel(remote_path,
                  {{schema::AnySectionDataType_SP_Tokenizer, "",
                    "tok_much_longer_template_expansion_causes_shift", 4096},
                   {schema::AnySectionDataType_TFLiteModel, "weights",
                    heavy_weights, 16384}});

  std::vector<ByteRange> logged_ranges;
  auto fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);
  std::string output_path = (test_dir_ / "output_shifted.litertlm").string();

  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  // Verify heavy weights were NOT fetched over network despite offset shift.
  for (const auto& r : logged_ranges) {
    EXPECT_LT(r.start_byte, 16384);
  }

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(output_path, &updated_header));
  auto* objs = updated_header.metadata->section_metadata()->objects();
  ASSERT_EQ(objs->size(), 2);
  EXPECT_EQ(objs->Get(1)->begin_offset(), 16384);

  std::ifstream updated_file(output_path, std::ios::binary);
  std::string sec1_payload(
      objs->Get(1)->end_offset() - objs->Get(1)->begin_offset(), '\0');
  updated_file.seekg(objs->Get(1)->begin_offset());
  updated_file.read(sec1_payload.data(), sec1_payload.size());
  EXPECT_EQ(sec1_payload, heavy_weights);
}

TEST_F(ModelSyncTest, Update_LocalHeaderCorrupt_FallsBackToFullDownload) {
  std::string local_path = (test_dir_ / "corrupt_local.litertlm").string();
  std::ofstream out(local_path, std::ios::binary);
  out.write("BAD_HEADER_GARBAGE_BYTES_1234567890", 35);
  out.close();

  std::string remote_path = (test_dir_ / "remote_valid.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_valid", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "weights_valid", 8192}});

  std::string output_path = (test_dir_ / "healed_output.litertlm").string();
  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);

  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(output_path, &updated_header));
  EXPECT_EQ(updated_header.metadata->section_metadata()->objects()->size(), 2);
}

TEST_F(ModelSyncTest, Update_LocalPayloadCorrupt_FallsBackToRemoteFetch) {
  std::string local_path = (test_dir_ / "local_bitrot.litertlm").string();
  // Valid header with "weights_ORIGINAL", but tokenizer is older so it's not
  // up-to-date:
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "weights_ORIGINAL", 8192}});

  std::string remote_path = (test_dir_ / "remote_source.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_v2", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "weights_ORIGINAL", 8192}});

  // Overwrite weights payload on disk in local_path with corrupted bytes.
  // The local header metadata still expects sha256 of "weights_ORIGINAL",
  // so FindReusableLocalSection matches, but the on-disk hash check during
  // reuse fails and self-heals by fetching fresh weights from remote.
  {
    std::fstream file(local_path,
                      std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file.seekp(8192);
    std::string garbage = "weights_CORRUPTED";
    file.write(garbage.data(), garbage.size());
  }

  std::string output_path = (test_dir_ / "output_repaired.litertlm").string();
  auto fetcher = CreateFileRangeFetcher(remote_path, nullptr);

  // Self-healing: detects bit-rot on local weights and fetches fresh remote.
  ASSERT_OK(ModelSync::Update(local_path, output_path, fetcher));

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(output_path, &updated_header));
  auto* objs = updated_header.metadata->section_metadata()->objects();
  ASSERT_EQ(objs->size(), 2);

  std::ifstream updated_file(output_path, std::ios::binary);
  std::string sec1_payload(
      objs->Get(1)->end_offset() - objs->Get(1)->begin_offset(), '\0');
  updated_file.seekg(objs->Get(1)->begin_offset());
  updated_file.read(sec1_payload.data(), sec1_payload.size());
  EXPECT_EQ(sec1_payload, "weights_ORIGINAL");
}

TEST_F(ModelSyncTest, Update_RangeNotSupported_FallsBackToFullDownload) {
  std::string local_path = (test_dir_ / "local_range.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tok_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "weights_payload", 8192}});

  std::string remote_path = (test_dir_ / "remote_range.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tok_v2", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "weights_payload", 8192}});

  std::string output_path = (test_dir_ / "output_fallback.litertlm").string();

  // Simulates server returning error on arbitrary Range slices, but full stream
  // succeeds.
  auto range_failing_fetcher = [&remote_path](
                                   uint64_t start_byte, uint64_t end_byte,
                                   std::ostream& out) -> absl::Status {
    std::ifstream in(remote_path, std::ios::binary);
    in.seekg(0, std::ios::end);
    uint64_t file_len = in.tellg();
    in.seekg(start_byte);

    if (start_byte > 0 && end_byte < file_len - 1) {
      return absl::FailedPreconditionError(
          "HTTP 200 OK: server does not support partial byte ranges");
    }

    uint64_t bytes_to_read = (end_byte - start_byte) + 1;
    std::vector<char> buf(4096);
    while (bytes_to_read > 0) {
      size_t n = std::min<uint64_t>(bytes_to_read, buf.size());
      in.read(buf.data(), n);
      std::streamsize actual = in.gcount();
      if (actual <= 0) break;
      out.write(buf.data(), actual);
      bytes_to_read -= actual;
    }
    return absl::OkStatus();
  };

  SyncOptions options;
  options.fallback_on_range_error = true;
  ASSERT_OK(ModelSync::Update(local_path, output_path, range_failing_fetcher,
                              options));
  EXPECT_TRUE(std::filesystem::exists(output_path));
}

TEST_F(ModelSyncTest, Update_CleanFlag_DeletesOldModelUpfront) {
  std::string model_path = (test_dir_ / "clean_model.litertlm").string();
  CreateTestModel(model_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "old_tok", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "old_weights", 8192}});

  std::string remote_path = (test_dir_ / "remote_clean.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "new_tok", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "new_weights", 8192}});

  auto raw_fetcher = CreateFileRangeFetcher(remote_path, nullptr);
  bool old_model_deleted_during_fetch = false;
  auto fetcher = [&](uint64_t start_byte, uint64_t end_byte,
                     std::ostream& output_stream) -> absl::Status {
    std::error_code ec;
    if (!std::filesystem::exists(model_path, ec) && !ec) {
      old_model_deleted_during_fetch = true;
    }
    return raw_fetcher(start_byte, end_byte, output_stream);
  };

  std::error_code size_ec;
  uint64_t remote_size = std::filesystem::file_size(remote_path, size_ec);
  ASSERT_FALSE(size_ec);

  SyncOptions options;
  options.clean = true;
  // Override available space to less than remote_size to force
  // space_constrained = true, but enough when adding old model's size.
  options.available_disk_space_override = remote_size / 2;

  ASSERT_OK(ModelSync::Update(model_path, model_path, fetcher, options));
  EXPECT_TRUE(old_model_deleted_during_fetch);
  EXPECT_TRUE(std::filesystem::exists(model_path));

  schema::LitertlmHeader updated_header;
  ASSERT_OK(schema::ReadHeaderFromLiteRTLM(model_path, &updated_header));
  EXPECT_EQ(updated_header.metadata->section_metadata()->objects()->size(), 2);
}

TEST_F(ModelSyncTest, Update_NonWritableDestination_ReturnsPermissionDenied) {
  std::string local_path = (test_dir_ / "perm_local.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "old_tok", 4096}});

  std::string remote_path = (test_dir_ / "perm_remote.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "new_tok", 4096}});

  std::filesystem::path readonly_dir = test_dir_ / "readonly_dir";
  std::filesystem::create_directories(readonly_dir);
  std::string readonly_output = (readonly_dir / "output.litertlm").string();

  // Create a directory at the exact temporary staging path so ofstream fails to
  // open it across all platforms (including Windows and when running as root in
  // container environments where POSIX directory permissions are ignored).
  std::filesystem::create_directories(readonly_output + ".tmp_updating");

  std::vector<ByteRange> logged_ranges;
  auto counting_fetcher = CreateFileRangeFetcher(remote_path, &logged_ranges);

  SyncOptions options;
  auto status =
      ModelSync::Update(local_path, readonly_output, counting_fetcher, options);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kPermissionDenied));
  EXPECT_TRUE(std::filesystem::exists(local_path));
}

TEST_F(ModelSyncTest, PlanSync_WithTwoStreamFetchers_DiffsWithoutFilesystem) {
  std::string local_path = (test_dir_ / "stream_local.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "weights_shared", 8192}});

  std::string remote_path = (test_dir_ / "stream_remote.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_v2_NEW", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "weights_shared", 8192}});

  auto local_fetcher = CreateFileRangeFetcher(local_path, nullptr);
  auto remote_fetcher = CreateFileRangeFetcher(remote_path, nullptr);

  // Both local and remote are pure RangeFetchers:
  ASSERT_OK_AND_ASSIGN(SyncPlan plan,
                       ModelSync::PlanSync(local_fetcher, remote_fetcher));

  EXPECT_FALSE(plan.is_up_to_date);
  EXPECT_EQ(plan.bytes_to_reuse, std::string("weights_shared").size());
  ASSERT_EQ(plan.section_diffs.size(), 2);
  EXPECT_EQ(plan.section_diffs[0].action, SectionAction::kDownloadDiff);
  EXPECT_EQ(plan.section_diffs[1].action, SectionAction::kReuseLocal);
}

TEST_F(ModelSyncTest,
       PlanSync_MalformedHeaderEndOffset32_ReturnsInvalidArgument) {
  std::string malformed_path =
      (test_dir_ / "malformed_header_32.litertlm").string();
  std::ofstream out(malformed_path, std::ios::binary | std::ios::trunc);
  out.write("LITERTLM", 8);
  uint32_t major = 1, minor = 9, patch = 0, pad = 0;
  out.write(reinterpret_cast<const char*>(&major), 4);
  out.write(reinterpret_cast<const char*>(&minor), 4);
  out.write(reinterpret_cast<const char*>(&patch), 4);
  out.write(reinterpret_cast<const char*>(&pad), 4);
  uint64_t header_end_offset = 32;
  out.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  out.close();

  auto fetcher = CreateFileRangeFetcher(malformed_path, nullptr);
  auto status_or_plan = ModelSync::PlanSync("", fetcher);
  EXPECT_THAT(status_or_plan.status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::string output_path = (test_dir_ / "out_32.litertlm").string();
  absl::Status update_status = ModelSync::Update("", output_path, fetcher);
  EXPECT_THAT(update_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ModelSyncTest, PlanSync_OverlappingSections_ReturnsInvalidArgument) {
  std::string overlapping_path =
      (test_dir_ / "overlapping_sections.litertlm").string();
  // Section 0: begin=4096, payload size=10 -> end=4106.
  // Section 1: begin=4100 (overlaps with section 0), size=10 -> end=4110.
  CreateTestModel(overlapping_path, {{schema::AnySectionDataType_SP_Tokenizer,
                                      "", "0123456789", 4096},
                                     {schema::AnySectionDataType_TFLiteModel,
                                      "weights", "0123456789", 4100}});

  auto fetcher = CreateFileRangeFetcher(overlapping_path, nullptr);
  auto status_or_plan = ModelSync::PlanSync("", fetcher);
  EXPECT_THAT(status_or_plan.status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::string output_path = (test_dir_ / "out_overlap.litertlm").string();
  absl::Status update_status = ModelSync::Update("", output_path, fetcher);
  EXPECT_THAT(update_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ModelSyncTest, Sync_OfflineFallback_RejectsCorruptCachedModel) {
  std::string cache_dir = (test_dir_ / "corrupt_cache_dir").string();
  std::filesystem::create_directories(cache_dir);
  std::string corrupt_cached_path =
      (std::filesystem::path(cache_dir) / "corrupt_model.litertlm").string();
  std::ofstream out(corrupt_cached_path, std::ios::binary | std::ios::trunc);
  out << "corrupt non-litertlm payload data";
  out.close();

  auto offline_fetcher = [](uint64_t start_byte, uint64_t end_byte,
                            std::ostream& output_stream) -> absl::Status {
    return absl::UnavailableError("Network offline");
  };

  SyncOptions options;
  options.allow_offline_fallback = true;
  auto result =
      ModelSync::Sync(cache_dir, "corrupt_model", offline_fetcher, options);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kUnavailable));
}

TEST(SyncPlanTest, FormatsDataTypeNamesWithUnknownFallback) {
  SyncPlan plan;
  plan.section_diffs = {
      SectionDiff{
          .section_index = 0,
          .data_type =
              static_cast<int>(schema::AnySectionDataType_LlmMetadataProto),
          .size = 128,
          .action = SectionAction::kReuseLocal,
      },
      SectionDiff{
          .section_index = 1,
          .data_type = static_cast<int>(schema::AnySectionDataType_TFLiteModel),
          .model_type = "tf_lite_embedder",
          .size = 256,
          .action = SectionAction::kDownloadDiff,
      },
      SectionDiff{
          .section_index = 2,
          .data_type = 999,
          .size = 64,
          .action = SectionAction::kDownloadDiff,
      },
  };
  std::string summary = plan.Summary();
  EXPECT_NE(summary.find("data_type=LlmMetadataProto"), std::string::npos);
  EXPECT_NE(
      summary.find("data_type=TFLiteModel, model_type=\"tf_lite_embedder\""),
      std::string::npos);
  EXPECT_NE(summary.find("data_type=999"), std::string::npos);
}

TEST(HttpRangeFetcherTest, IsHttpUrl) {
  EXPECT_TRUE(IsHttpUrl("http://example.com/model.litertlm"));
  EXPECT_TRUE(IsHttpUrl("https://huggingface.co/model.litertlm"));
  EXPECT_TRUE(IsHttpUrl("HTTPS://example.com/model.litertlm"));
  EXPECT_FALSE(IsHttpUrl("/tmp/model.litertlm"));
  EXPECT_FALSE(IsHttpUrl(""));
}

TEST(HttpRangeFetcherTest, RejectsInvalidUrlAndRange) {
  auto bad_url = CreateHttpRangeFetcher("/tmp/not_a_url.litertlm");
  std::ostringstream out;
  EXPECT_THAT(bad_url(0, 10, out),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_chars = CreateHttpRangeFetcher("https://example.com/bad url");
  EXPECT_THAT(bad_chars(0, 10, out),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_quote = CreateHttpRangeFetcher("https://example.com/bad\"url");
  EXPECT_THAT(bad_quote(0, 10, out),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto fetcher = CreateHttpRangeFetcher("https://example.com/model.litertlm");
  EXPECT_THAT(fetcher(10, 5, out),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(fetcher(0, std::numeric_limits<uint64_t>::max(), out),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

#if LITERT_LM_HAS_POSIX_CURL_TEST

std::filesystem::path WriteMockCurl(const std::filesystem::path& bin_dir,
                                    const std::filesystem::path& args_log) {
  std::filesystem::create_directories(bin_dir);
  std::filesystem::path script_path = bin_dir / "curl";
  std::ofstream script(script_path);
  script << "#!/bin/sh\n"
         << "printf '%s\\n' \"$*\" > \"" << args_log.string() << "\"\n"
         << "HDR=\"\"\n"
         << "URL=\"\"\n"
         << "PREV=\"\"\n"
         << "for arg in \"$@\"; do\n"
         << "  if [ \"$PREV\" = \"-D\" ]; then HDR=\"$arg\"; fi\n"
         << "  PREV=\"$arg\"\n"
         << "  URL=\"$arg\"\n"
         << "done\n"
         << "write_hdr() {\n"
         << "  if [ -n \"$HDR\" ]; then printf '%b' \"$1\" > \"$HDR\"; fi\n"
         << "}\n"
         << "case \"$URL\" in\n"
         << "  *ok.litertlm*)\n"
         << "    write_hdr 'HTTP/1.1 302 Found\\r\\nLocation: /cdn\\r\\n\\r\\n"
         << "HTTP/2 206 \\r\\n\\r\\n'\n"
         << "    printf 'ABCDEFGHIJ' ;;\n"
         << "  *http200.litertlm*)\n"
         << "    write_hdr 'HTTP/1.1 200 OK\\r\\n\\r\\n'\n"
         << "    printf 'ABCDEFGHIJ' ;;\n"
         << "  *http401.litertlm*)\n"
         << "    write_hdr 'HTTP/1.1 401 Unauthorized\\r\\n\\r\\n'\n"
         << "    exit 22 ;;\n"
         << "  *http403.litertlm*)\n"
         << "    write_hdr 'HTTP/1.1 403 Forbidden\\r\\n\\r\\n'\n"
         << "    exit 22 ;;\n"
         << "  *http404.litertlm*)\n"
         << "    write_hdr 'HTTP/1.1 404 Not Found\\r\\n\\r\\n'\n"
         << "    exit 22 ;;\n"
         << "  *http416.litertlm*)\n"
         << "    write_hdr 'HTTP/1.1 416 Range Not Satisfiable\\r\\n\\r\\n'\n"
         << "    exit 22 ;;\n"
         << "  *oversized.litertlm*) printf '0123456789ABCDEFGH' ;;\n"
         << "  *filesize.litertlm*) exit 63 ;;\n"
         << "  *timeout.litertlm*) exit 28 ;;\n"
         << "  *) exit 22 ;;\n"
         << "esac\n";
  script.close();
  chmod(script_path.c_str(), 0755);
  return script_path;
}

std::filesystem::path WriteModelServingMockCurl(
    const std::filesystem::path& script_path,
    const std::filesystem::path& model_path, bool ignore_nonzero_range,
    bool truncate_section) {
  std::filesystem::create_directories(script_path.parent_path());
  std::ofstream script(script_path);
  script << "#!/bin/sh\n"
         << "HDR=\"\"\n"
         << "RANGE=\"\"\n"
         << "PREV=\"\"\n"
         << "for arg in \"$@\"; do\n"
         << "  if [ \"$PREV\" = \"-D\" ]; then HDR=\"$arg\"; fi\n"
         << "  if [ \"$PREV\" = \"--range\" ]; then RANGE=\"$arg\"; fi\n"
         << "  PREV=\"$arg\"\n"
         << "done\n"
         << "START=${RANGE%%-*}\n"
         << "END=${RANGE##*-}\n"
         << "LEN=$((END - START + 1))\n"
         << "write_hdr() {\n"
         << "  if [ -n \"$HDR\" ]; then printf '%b' \"$1\" > \"$HDR\"; fi\n"
         << "}\n";
  if (ignore_nonzero_range) {
    script << "if [ \"$START\" -gt 0 ]; then\n"
           << "  write_hdr 'HTTP/1.1 200 OK\\r\\n\\r\\n'\n"
           << "  head -c \"$LEN\" \"" << model_path.string() << "\"\n"
           << "  exit 0\n"
           << "fi\n";
  } else if (truncate_section) {
    script << "if [ \"$START\" -gt 0 ]; then\n"
           << "  write_hdr 'HTTP/1.1 206 Partial Content\\r\\n\\r\\n'\n"
           << "  tail -c +$((START + 1)) \"" << model_path.string()
           << "\" | head -c 3\n"
           << "  exit 0\n"
           << "fi\n";
  }
  script << "write_hdr 'HTTP/1.1 206 Partial Content\\r\\n\\r\\n'\n"
         << "tail -c +$((START + 1)) \"" << model_path.string()
         << "\" | head -c \"$LEN\"\n";
  script.close();
  chmod(script_path.c_str(), 0755);
  return script_path;
}

TEST_F(ModelSyncTest, HttpRangeFetcher_FetchesRangeAndMapsErrors) {
  std::filesystem::path bin_dir = test_dir_ / "mock_bin";
  std::filesystem::path args_log = test_dir_ / "curl_args.txt";
  std::filesystem::path mock_curl = WriteMockCurl(bin_dir, args_log);

  auto fetcher = internal::CreateHttpRangeFetcherWithExecutable(
      "https://example.com/ok.litertlm", mock_curl.string());
  std::ostringstream out(std::ios::binary);
  ASSERT_OK(fetcher(10, 19, out));
  EXPECT_EQ(out.str(), "ABCDEFGHIJ");

  std::ifstream log_in(args_log);
  std::string recorded_args((std::istreambuf_iterator<char>(log_in)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(recorded_args.find("-g"), std::string::npos);
  EXPECT_NE(recorded_args.find("--proto =http,https"), std::string::npos);
  EXPECT_NE(recorded_args.find("--proto-redir =http,https"), std::string::npos);
  EXPECT_NE(recorded_args.find("--speed-limit 1"), std::string::npos);
  EXPECT_NE(recorded_args.find("--speed-time 30"), std::string::npos);
  EXPECT_NE(recorded_args.find("--max-filesize 10"), std::string::npos);
  EXPECT_NE(recorded_args.find("--range 10-19"), std::string::npos);

  // HTTP 200 OK is valid when start_byte == 0, but rejected when start_byte >
  // 0 (server ignored Range).
  std::ostringstream ok_zero(std::ios::binary);
  ASSERT_OK(internal::CreateHttpRangeFetcherWithExecutable(
      "https://example.com/http200.litertlm", mock_curl.string())(0, 9,
                                                                  ok_zero));
  EXPECT_EQ(ok_zero.str(), "ABCDEFGHIJ");

  std::ostringstream err_out(std::ios::binary);
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/http200.litertlm", mock_curl.string())(
                  10, 19, err_out),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/oversized.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/filesize.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/timeout.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/http401.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kUnauthenticated));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/http403.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kPermissionDenied));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/http404.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/http416.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/missing.litertlm", mock_curl.string())(
                  0, 9, err_out),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(internal::CreateHttpRangeFetcherWithExecutable(
                  "https://example.com/ok.litertlm",
                  (test_dir_ / "nonexistent_curl").string())(0, 9, err_out),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(ModelSyncTest, HttpSync_Http404AndMissingCurlSkipOfflineFallback) {
  std::filesystem::path cache_dir = test_dir_ / "sync_cache";
  std::filesystem::create_directories(cache_dir);
  std::string cached_path = (cache_dir / "my_model.litertlm").string();
  CreateTestModel(cached_path, {{schema::AnySectionDataType_TFLiteModel,
                                 "weights", "cached_weights", 4096}});

  std::filesystem::path mock_curl =
      WriteMockCurl(test_dir_ / "mock_bin_sync", test_dir_ / "sync_args.txt");

  SyncOptions options;
  options.allow_offline_fallback = true;

  auto not_found_res = ModelSync::Sync(
      cache_dir.string(), "my_model",
      internal::CreateHttpRangeFetcherWithExecutable(
          "https://example.com/http404.litertlm", mock_curl.string()),
      options);
  EXPECT_THAT(not_found_res.status(), StatusIs(absl::StatusCode::kNotFound));

  auto missing_curl_res =
      ModelSync::Sync(cache_dir.string(), "my_model",
                      internal::CreateHttpRangeFetcherWithExecutable(
                          "https://example.com/ok.litertlm",
                          (test_dir_ / "nonexistent_curl").string()),
                      options);
  EXPECT_THAT(missing_curl_res.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(ModelSyncTest, HttpUpdate_RangeIgnoredFallsBackToFullDownload) {
  std::string local_path = (test_dir_ / "http_local_v1.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_LlmMetadataProto, "",
                                "meta_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "shared_weights_payload", 8192}});

  std::string remote_path = (test_dir_ / "http_remote_v2.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_LlmMetadataProto,
                                 "", "meta_v2_updated", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "shared_weights_payload", 8192}});

  std::filesystem::path mock_curl = WriteModelServingMockCurl(
      test_dir_ / "mock_bin_fallback" / "curl", remote_path,
      /*ignore_nonzero_range=*/true, /*truncate_section=*/false);
  std::string output_path = (test_dir_ / "http_fallback_out.litertlm").string();

  SyncOptions options;
  options.fallback_on_range_error = true;
  ASSERT_OK(ModelSync::Update(
      local_path, output_path,
      internal::CreateHttpRangeFetcherWithExecutable(
          "https://example.com/v2.litertlm", mock_curl.string()),
      options));

  std::ifstream expected_in(remote_path, std::ios::binary);
  std::ifstream actual_in(output_path, std::ios::binary);
  std::string expected_bytes((std::istreambuf_iterator<char>(expected_in)),
                             std::istreambuf_iterator<char>());
  std::string actual_bytes((std::istreambuf_iterator<char>(actual_in)),
                           std::istreambuf_iterator<char>());
  EXPECT_EQ(actual_bytes, expected_bytes);
}

TEST_F(ModelSyncTest, HttpUpdate_TruncatedResponseFailsWithDataLoss) {
  std::string local_path = (test_dir_ / "http_trunc_v1.litertlm").string();
  CreateTestModel(local_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                "tokenizer_v1", 4096},
                               {schema::AnySectionDataType_TFLiteModel,
                                "weights", "shared_weights_payload", 8192}});

  std::string remote_path = (test_dir_ / "http_trunc_v2.litertlm").string();
  CreateTestModel(remote_path, {{schema::AnySectionDataType_SP_Tokenizer, "",
                                 "tokenizer_v2_much_longer", 4096},
                                {schema::AnySectionDataType_TFLiteModel,
                                 "weights", "shared_weights_payload", 8192}});

  std::filesystem::path mock_curl = WriteModelServingMockCurl(
      test_dir_ / "mock_bin_trunc" / "curl", remote_path,
      /*ignore_nonzero_range=*/false, /*truncate_section=*/true);
  std::string output_path = (test_dir_ / "http_trunc_out.litertlm").string();

  EXPECT_THAT(ModelSync::Update(
                  local_path, output_path,
                  internal::CreateHttpRangeFetcherWithExecutable(
                      "https://example.com/v2.litertlm", mock_curl.string())),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_FALSE(std::filesystem::exists(output_path));
}

#endif  // LITERT_LM_HAS_POSIX_CURL_TEST

}  // namespace
}  // namespace litert::lm::sync
