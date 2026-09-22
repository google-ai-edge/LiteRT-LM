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

#include "omni/tts/kokoro/espeak_assets.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/scoped_file.h"
#include "support/util/test_utils.h"  // NOLINT

namespace litert::omni::tts::kokoro {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// A single file to place into a test zip archive.
struct ZipEntry {
  std::string name;
  std::string content;
};

uint32_t Crc32(absl::string_view data) {
  uint32_t crc = 0xffffffffu;
  for (const char c : data) {
    crc ^= static_cast<uint8_t>(c);
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xedb88320u & (~(crc & 1u) + 1u));
    }
  }
  return ~crc;
}

void AppendLe16(std::string& out, uint16_t value) {
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
}

void AppendLe32(std::string& out, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
  }
}

// Builds a zip archive with stored (uncompressed) entries, which is the layout
// the LiteRT-LM zip reader supports.
std::string MakeStoredZip(const std::vector<ZipEntry>& entries) {
  std::string zip;
  std::vector<uint32_t> local_header_offsets;
  std::vector<uint32_t> crcs;

  for (const ZipEntry& entry : entries) {
    local_header_offsets.push_back(static_cast<uint32_t>(zip.size()));
    crcs.push_back(Crc32(entry.content));

    AppendLe32(zip, 0x04034b50);  // Local file header signature.
    AppendLe16(zip, 20);          // Version needed to extract.
    AppendLe16(zip, 0);           // General purpose bit flag.
    AppendLe16(zip, 0);           // Compression method: stored.
    AppendLe16(zip, 0);           // Modification time.
    AppendLe16(zip, 0);           // Modification date.
    AppendLe32(zip, crcs.back());
    AppendLe32(zip, static_cast<uint32_t>(entry.content.size()));
    AppendLe32(zip, static_cast<uint32_t>(entry.content.size()));
    AppendLe16(zip, static_cast<uint16_t>(entry.name.size()));
    AppendLe16(zip, 0);  // Extra field length.
    zip += entry.name;
    zip += entry.content;
  }

  const uint32_t central_directory_offset = static_cast<uint32_t>(zip.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    const ZipEntry& entry = entries[i];
    AppendLe32(zip, 0x02014b50);  // Central directory header signature.
    AppendLe16(zip, 20);          // Version made by.
    AppendLe16(zip, 20);          // Version needed to extract.
    AppendLe16(zip, 0);           // General purpose bit flag.
    AppendLe16(zip, 0);           // Compression method: stored.
    AppendLe16(zip, 0);           // Modification time.
    AppendLe16(zip, 0);           // Modification date.
    AppendLe32(zip, crcs[i]);
    AppendLe32(zip, static_cast<uint32_t>(entry.content.size()));
    AppendLe32(zip, static_cast<uint32_t>(entry.content.size()));
    AppendLe16(zip, static_cast<uint16_t>(entry.name.size()));
    AppendLe16(zip, 0);  // Extra field length.
    AppendLe16(zip, 0);  // File comment length.
    AppendLe16(zip, 0);  // Disk number start.
    AppendLe16(zip, 0);  // Internal file attributes.
    AppendLe32(zip, 0);  // External file attributes.
    AppendLe32(zip, local_header_offsets[i]);
    zip += entry.name;
  }
  const uint32_t central_directory_size =
      static_cast<uint32_t>(zip.size()) - central_directory_offset;

  AppendLe32(zip, 0x06054b50);  // End of central directory signature.
  AppendLe16(zip, 0);           // Disk number.
  AppendLe16(zip, 0);           // Disk with the central directory.
  AppendLe16(zip, static_cast<uint16_t>(entries.size()));
  AppendLe16(zip, static_cast<uint16_t>(entries.size()));
  AppendLe32(zip, central_directory_size);
  AppendLe32(zip, central_directory_offset);
  AppendLe16(zip, 0);  // Zip file comment length.
  return zip;
}

// ModelResources stub serving a single named GenericBinaryData section.
class FakeModelResources : public lm::ModelResources {
 public:
  FakeModelResources(std::string section_name, std::string section_data)
      : section_name_(std::move(section_name)),
        section_data_(std::move(section_data)) {}

  absl::StatusOr<absl::string_view> GetGenericBinaryDataBuffer(
      absl::string_view name) override {
    if (name != section_name_) {
      return absl::NotFoundError(
          absl::StrCat("No GenericBinaryData section named ", name));
    }
    return absl::string_view(section_data_);
  }

  std::vector<std::string> GetGenericBinaryDataNames() const override {
    return {section_name_};
  }

  // Unused parts of the interface.
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::reference_wrapper<lm::ScopedFile>> GetScopedFile()
      override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<lm::FileRegion> GetTFLiteModelSectionFileRegion(
      lm::ModelType model_type) override {
    return absl::UnimplementedError("");
  }
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      lm::ModelType model_type) override {
    return std::nullopt;
  }
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      lm::ModelType model_type) override {
    return std::nullopt;
  }
  absl::StatusOr<std::unique_ptr<lm::Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::LlmMetadata*> GetLlmMetadata() override {
    return absl::UnimplementedError("");
  }
  absl::StatusOr<const lm::proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("");
  }

 private:
  std::string section_name_;
  std::string section_data_;
};

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

std::string MakeCacheDir(absl::string_view name) {
  const std::filesystem::path cache_dir =
      std::filesystem::path(::testing::TempDir()) / std::string(name);
  std::filesystem::remove_all(cache_dir);
  std::filesystem::create_directories(cache_dir);
  return cache_dir.string();
}

TEST(EspeakAssetsTest, UnpacksArchiveIntoCacheDir) {
  FakeModelResources resources(
      std::string(kEspeakNgSectionName),
      MakeStoredZip({{"phontab", "phontab-data"},
                     {"intonations", "intonation-data"},
                     {"voices/!v/Mr serious", "voice-data"}}));
  const std::string cache_dir = MakeCacheDir("espeak_unpack");

  ASSERT_OK_AND_ASSIGN(const std::string data_dir,
                       UnpackEspeakDataFromLitertLm(resources, cache_dir));

  const std::filesystem::path dir(data_dir);
  EXPECT_EQ(dir, std::filesystem::path(cache_dir) / kEspeakDataDirName);
  EXPECT_EQ(ReadFile(dir / "phontab"), "phontab-data");
  EXPECT_EQ(ReadFile(dir / "intonations"), "intonation-data");
  EXPECT_EQ(ReadFile(dir / "voices" / "!v" / "Mr serious"), "voice-data");
}

TEST(EspeakAssetsTest, StripsLeadingDataDirComponent) {
  FakeModelResources resources(
      std::string(kEspeakNgSectionName),
      MakeStoredZip({{"espeak-ng-data/phontab", "phontab-data"},
                     {"espeak-ng-data/phondata", "phon-data"}}));
  const std::string cache_dir = MakeCacheDir("espeak_prefixed");

  ASSERT_OK_AND_ASSIGN(const std::string data_dir,
                       UnpackEspeakDataFromLitertLm(resources, cache_dir));

  const std::filesystem::path dir(data_dir);
  EXPECT_EQ(ReadFile(dir / "phontab"), "phontab-data");
  EXPECT_EQ(ReadFile(dir / "phondata"), "phon-data");
  EXPECT_FALSE(std::filesystem::exists(dir / kEspeakDataDirName));
}

TEST(EspeakAssetsTest, ReusesPreviouslyUnpackedData) {
  FakeModelResources resources(std::string(kEspeakNgSectionName),
                               MakeStoredZip({{"phontab", "phontab-data"}}));
  const std::string cache_dir = MakeCacheDir("espeak_reuse");

  ASSERT_OK_AND_ASSIGN(const std::string data_dir,
                       UnpackEspeakDataFromLitertLm(resources, cache_dir));
  // A marker file survives only if the directory is reused as-is.
  const std::filesystem::path marker =
      std::filesystem::path(data_dir) / "marker";
  std::ofstream(marker) << "marker";

  ASSERT_OK_AND_ASSIGN(const std::string reused_dir,
                       UnpackEspeakDataFromLitertLm(resources, cache_dir));

  EXPECT_EQ(reused_dir, data_dir);
  EXPECT_TRUE(std::filesystem::exists(marker));
}

TEST(EspeakAssetsTest, ReturnsNotFoundWithoutEspeakSection) {
  FakeModelResources resources("af_heart", "voice-pack-bytes");
  const std::string cache_dir = MakeCacheDir("espeak_missing");

  EXPECT_THAT(UnpackEspeakDataFromLitertLm(resources, cache_dir).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(EspeakAssetsTest, RejectsArchiveWithoutPhonemeTable) {
  FakeModelResources resources(std::string(kEspeakNgSectionName),
                               MakeStoredZip({{"intonations", "data"}}));
  const std::string cache_dir = MakeCacheDir("espeak_no_phontab");

  const absl::Status status =
      UnpackEspeakDataFromLitertLm(resources, cache_dir).status();

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("phontab"));
  EXPECT_FALSE(std::filesystem::exists(
      std::filesystem::path(cache_dir) / kEspeakDataDirName));
}

TEST(EspeakAssetsTest, RejectsEntriesEscapingTheDestination) {
  FakeModelResources resources(
      std::string(kEspeakNgSectionName),
      MakeStoredZip({{"../escaped", "data"}, {"phontab", "phontab-data"}}));
  const std::string cache_dir = MakeCacheDir("espeak_zip_slip");

  EXPECT_THAT(UnpackEspeakDataFromLitertLm(resources, cache_dir).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_FALSE(std::filesystem::exists(
      std::filesystem::path(cache_dir).parent_path() / "escaped"));
}

}  // namespace
}  // namespace litert::omni::tts::kokoro
