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
#include <ostream>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <vector>

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

namespace litert::lm::sync {
namespace {

using ::testing::IsEmpty;
using ::testing::status::StatusIs;

struct TestSectionSpec {
  schema::AnySectionDataType data_type;
  std::string model_type;
  std::string payload;
  size_t begin_offset;
};

std::string CreateTestModel(const std::string& path,
                            const std::vector<TestSectionSpec>& sections,
                            bool include_sha256 = true) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<schema::SectionObject>> section_objs;
  for (const auto& sec : sections) {
    std::vector<flatbuffers::Offset<schema::KeyValuePair>> items;
    if (include_sha256) {
      std::string sha = ComputeSha256Hex(sec.payload);
      items.push_back(schema::CreateKeyValuePair(builder, "sha256", sha));
    }
    if (!sec.model_type.empty()) {
      items.push_back(
          schema::CreateKeyValuePair(builder, "model_type", sec.model_type));
    }

    size_t end_offset = sec.begin_offset + sec.payload.size();
    section_objs.push_back(schema::CreateSectionObject(
        builder, builder.CreateVector(items), sec.begin_offset, end_offset,
        sec.data_type));
  }

  auto section_metadata_offset = schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objs));
  auto root_offset =
      schema::CreateLiteRTLMMetaData(builder, 0, section_metadata_offset);
  builder.Finish(root_offset);

  uint8_t* fb_buf = builder.GetBufferPointer();
  size_t fb_size = builder.GetSize();

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write("LITERTLM", 8);
  uint32_t major = 1, minor = 9, patch = 0, pad = 0;
  out.write(reinterpret_cast<const char*>(&major), 4);
  out.write(reinterpret_cast<const char*>(&minor), 4);
  out.write(reinterpret_cast<const char*>(&patch), 4);
  out.write(reinterpret_cast<const char*>(&pad), 4);

  uint64_t header_end_offset = 32 + fb_size;
  out.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  out.write(reinterpret_cast<const char*>(fb_buf), fb_size);

  for (const auto& sec : sections) {
    if (static_cast<uint64_t>(out.tellp()) < sec.begin_offset) {
      size_t fill = sec.begin_offset - out.tellp();
      std::string zeroes(fill, '\0');
      out.write(zeroes.data(), fill);
    }
    out.seekp(sec.begin_offset);
    out.write(sec.payload.data(), sec.payload.size());
  }
  out.close();

  return path;
}

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
  EXPECT_EQ(
      ComputeSha256Hex(""),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  // "abc"
  EXPECT_EQ(
      ComputeSha256Hex("abc"),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // 56-byte message (spans across block boundary during padding)
  EXPECT_EQ(
      ComputeSha256Hex(
          "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // 55-byte boundary (largest message fitting in single 64-byte block with pad)
  std::string msg55(55, 'A');
  EXPECT_EQ(
      ComputeSha256Hex(msg55),
      "8963cc0afd622cc7574ac2011f93a3059b3d65548a77542a1559e3d202e6ab00");
  // 64-byte boundary (exact one block before padding)
  std::string msg64(64, 'B');
  EXPECT_EQ(
      ComputeSha256Hex(msg64),
      "c422e7070cb1cb455b5de9afee0d975e303d0239c72030cd7414ab5c382d3ae8");
  // 1,000,000 'a' characters (NIST CAVP)
  std::string one_million_a(1000000, 'a');
  EXPECT_EQ(
      ComputeSha256Hex(one_million_a),
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
  EXPECT_EQ(
      hasher.FinalHex(),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // Hasher was reset by Final/FinalHex, second call hashes empty string.
  EXPECT_EQ(
      hasher.FinalHex(),
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
  auto status = ModelSync::Update(local_path, readonly_output, counting_fetcher,
                                  options);

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
  absl::Status update_status =
      ModelSync::Update("", output_path, fetcher);
  EXPECT_THAT(update_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ModelSyncTest, PlanSync_OverlappingSections_ReturnsInvalidArgument) {
  std::string overlapping_path =
      (test_dir_ / "overlapping_sections.litertlm").string();
  // Section 0: begin=4096, payload size=10 -> end=4106.
  // Section 1: begin=4100 (overlaps with section 0), size=10 -> end=4110.
  CreateTestModel(
      overlapping_path,
      {{schema::AnySectionDataType_SP_Tokenizer, "", "0123456789", 4096},
       {schema::AnySectionDataType_TFLiteModel, "weights", "0123456789",
        4100}});

  auto fetcher = CreateFileRangeFetcher(overlapping_path, nullptr);
  auto status_or_plan = ModelSync::PlanSync("", fetcher);
  EXPECT_THAT(status_or_plan.status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::string output_path = (test_dir_ / "out_overlap.litertlm").string();
  absl::Status update_status =
      ModelSync::Update("", output_path, fetcher);
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
  auto result = ModelSync::Sync(cache_dir, "corrupt_model", offline_fetcher,
                                options);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kUnavailable));
}

}  // namespace
}  // namespace litert::lm::sync
