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

#include "omni/base/model_cache.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/file_util.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
using ::testing::StartsWith;

class ModelCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::path(::testing::TempDir()) / "model_cache";
    std::filesystem::remove_all(root_);
    model_dir_ = root_ / "models";
    std::filesystem::create_directories(model_dir_);
    model_path_ = (model_dir_ / "model.tflite").string();
    std::ofstream(model_path_) << "model";
    cache_dir_ = (root_ / "cache").string();
  }

  void Touch(const std::string& name) {
    std::filesystem::create_directories(cache_dir_);
    std::ofstream((std::filesystem::path(cache_dir_) / name).string()) << "x";
  }

  bool Exists(const std::string& name) {
    return std::filesystem::exists(std::filesystem::path(cache_dir_) / name);
  }

  std::filesystem::path root_;
  std::filesystem::path model_dir_;
  std::string model_path_;
  std::string cache_dir_;
};

TEST_F(ModelCacheTest, ExplicitDirIsCreatedAndKeyEmbedsModelFileId) {
  ASSERT_OK_AND_ASSIGN(ModelCacheConfig cache,
                       ResolveModelCache("m", model_path_, cache_dir_));
  EXPECT_EQ(cache.dir, cache_dir_);
  EXPECT_THAT(cache.key, StartsWith("m_"));
  EXPECT_TRUE(std::filesystem::is_directory(cache_dir_));
}

TEST_F(ModelCacheTest, EmptyDirMeansNextToModelFile) {
  ASSERT_OK_AND_ASSIGN(ModelCacheConfig cache,
                       ResolveModelCache("m", model_path_, ""));
  EXPECT_EQ(cache.dir, model_dir_.string());
  EXPECT_THAT(cache.key, StartsWith("m_"));
}

TEST_F(ModelCacheTest, NoCacheOrMissingModelDisablesCaching) {
  ASSERT_OK_AND_ASSIGN(ModelCacheConfig no_cache,
                       ResolveModelCache("m", model_path_, ":nocache"));
  EXPECT_THAT(no_cache.key, IsEmpty());
  ASSERT_OK_AND_ASSIGN(
      ModelCacheConfig missing_model,
      ResolveModelCache("m", (model_dir_ / "missing.tflite").string(),
                        cache_dir_));
  EXPECT_THAT(missing_model.key, IsEmpty());
}

TEST_F(ModelCacheTest, UnwritableExplicitDirIsPermissionDenied) {
  EXPECT_THAT(ResolveModelCache("m", model_path_, "/non/existent/cache/dir"),
              StatusIs(absl::StatusCode::kPermissionDenied));
}

TEST_F(ModelCacheTest, ReadOnlyModelDirSkipsCachingInsteadOfFailing) {
  namespace fs = std::filesystem;
  const fs::path read_only_dir = root_ / "read_only";
  fs::create_directories(read_only_dir);
  const std::string read_only_model = (read_only_dir / "model.tflite").string();
  std::ofstream(read_only_model) << "model";
  fs::permissions(
      read_only_dir,
      fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write,
      fs::perm_options::remove);
  if (!lm::IsDirectoryWritable(read_only_dir.string())) {  // Not as root.
    ASSERT_OK_AND_ASSIGN(
        ModelCacheConfig implicit_cache,
        ResolveModelCache("m", read_only_model, /*cache_dir=*/""));
    EXPECT_THAT(implicit_cache.key, IsEmpty());
    ASSERT_OK_AND_ASSIGN(
        ModelCacheConfig cache,
        ResolveModelCache("m", read_only_model, read_only_dir.string()));
    EXPECT_THAT(cache.key, IsEmpty());
    EXPECT_THAT(ResolveModelCache("m", model_path_, read_only_dir.string()),
                StatusIs(absl::StatusCode::kPermissionDenied));
  }
  fs::permissions(read_only_dir, fs::perms::owner_write, fs::perm_options::add);
}

TEST_F(ModelCacheTest, CpuCacheDeletesOnlyStaleCachesOfTheSameModel) {
  Touch("m_1_2.xnnpack_cache");      // Older version of model "m": deleted.
  Touch("m_v2_1_2.xnnpack_cache");   // Model "m_v2": kept.
  Touch("m_nosep.xnnpack_cache");    // Not a valid <mtime>_<size> id: kept.
  Touch("other_1_2.xnnpack_cache");  // Other model: kept.
  ASSERT_OK_AND_ASSIGN(ModelCacheConfig cache,
                       ResolveModelCache("m", model_path_, cache_dir_));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());
  auto cpu_options_or = options.GetOptions<::litert::CpuOptions>();
  ASSERT_TRUE(cpu_options_or);
  ::litert::CpuOptions& cpu_options = *cpu_options_or;

  ASSERT_OK(ConfigureCpuModelCache("m", cache, cpu_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto path,
                              cpu_options.GetXNNPackWeightCachePath());
  EXPECT_EQ(std::string(path),
            (std::filesystem::path(cache_dir_) / (cache.key + ".xnnpack_cache"))
                .string());
  EXPECT_FALSE(Exists("m_1_2.xnnpack_cache"));
  EXPECT_TRUE(Exists("m_v2_1_2.xnnpack_cache"));
  EXPECT_TRUE(Exists("m_nosep.xnnpack_cache"));
  EXPECT_TRUE(Exists("other_1_2.xnnpack_cache"));
}

TEST_F(ModelCacheTest, GpuCacheEnablesSerializationAndDeletesStaleCaches) {
  Touch("m_1_2_mldrift_program_cache.bin");
  Touch("m_1_2_mldrift_weight_cache.bin");
  ASSERT_OK_AND_ASSIGN(ModelCacheConfig cache,
                       ResolveModelCache("m", model_path_, cache_dir_));
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());
  auto gpu_options_or = options.GetOptions<::litert::GpuOptions>();
  ASSERT_TRUE(gpu_options_or);
  ::litert::GpuOptions& gpu_options = *gpu_options_or;

  ASSERT_OK(ConfigureGpuModelCache("m", cache, gpu_options));

  const char* key = nullptr;
  ASSERT_EQ(LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
                &key, gpu_options.Get()),
            kLiteRtStatusOk);
  EXPECT_EQ(std::string(key), cache.key);
  const char* dir = nullptr;
  ASSERT_EQ(LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
                &dir, gpu_options.Get()),
            kLiteRtStatusOk);
  EXPECT_EQ(std::string(dir), cache_dir_);
  bool serialize_programs = false;
  ASSERT_EQ(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
                &serialize_programs, gpu_options.Get()),
            kLiteRtStatusOk);
  EXPECT_TRUE(serialize_programs);
  EXPECT_FALSE(Exists("m_1_2_mldrift_program_cache.bin"));
  EXPECT_FALSE(Exists("m_1_2_mldrift_weight_cache.bin"));
}

TEST_F(ModelCacheTest, DisabledCacheIsNoOp) {
  Touch("m_1_2.xnnpack_cache");
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());
  auto cpu_options_or = options.GetOptions<::litert::CpuOptions>();
  ASSERT_TRUE(cpu_options_or);
  auto gpu_options_or = options.GetOptions<::litert::GpuOptions>();
  ASSERT_TRUE(gpu_options_or);

  ASSERT_OK(ConfigureCpuModelCache("m", ModelCacheConfig{}, *cpu_options_or));
  ASSERT_OK(ConfigureGpuModelCache("m", ModelCacheConfig{}, *gpu_options_or));

  LITERT_ASSERT_OK_AND_ASSIGN(auto path,
                              cpu_options_or->GetXNNPackWeightCachePath());
  EXPECT_THAT(std::string(path), IsEmpty());
  EXPECT_TRUE(Exists("m_1_2.xnnpack_cache"));
}

}  // namespace
}  // namespace litert::omni
