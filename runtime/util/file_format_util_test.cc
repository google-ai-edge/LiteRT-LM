// Copyright 2025 The ODML Authors.
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

#include "runtime/util/file_format_util.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(FileFormatUtilTest, FileFormatFromPath) {
  ASSERT_THAT(GetFileFormatFromPath("/path/to/model.tflite"),
              IsOkAndHolds(FileFormat::TFLITE));

  ASSERT_THAT(GetFileFormatFromPath("/path/to/model.task"),
              IsOkAndHolds(FileFormat::TASK));

  ASSERT_THAT(GetFileFormatFromPath("/path/to/model.litertlm"),
              IsOkAndHolds(FileFormat::LITERT_LM));

  // The final extension is used.
  ASSERT_THAT(GetFileFormatFromPath("/path/to/model.litertlm.task"),
              IsOkAndHolds(FileFormat::TASK));
}

TEST(FileFormatUtilTest, FileFormatFromContents) {
  // Read the .tflite file format.
  ASSERT_THAT(GetFileFormatFromFileContents(" TFL3otherstuff"),
              IsOkAndHolds(FileFormat::TFLITE));

  // Read the .task file format.
  ASSERT_THAT(GetFileFormatFromFileContents(" PKblahblah"),
              IsOkAndHolds(FileFormat::TASK));

  // .litertlm file format requires the magic signature to be the very first
  // bytes.
  ASSERT_THAT(GetFileFormatFromFileContents("LITERTLM12345"),
              IsOkAndHolds(FileFormat::LITERT_LM));
  ASSERT_THAT(GetFileFormatFromFileContents(" LITERTLM12345"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Magic signature is case sensitive.
  ASSERT_THAT(GetFileFormatFromFileContents("litertlm12345"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Fail to read empty file.
  ASSERT_THAT(GetFileFormatFromFileContents(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FileFormatUtilTest, FileFormatFromRealFile) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  std::string model_path_str = model_path.string();

  ASSERT_OK_AND_ASSIGN(auto scoped_file,
                              ScopedFile::Open(model_path_str));
  ASSERT_OK_AND_ASSIGN(auto mapped_file,
                       MemoryMappedFile::Create(scoped_file.file()));
  auto shared_scoped_file =
      std::make_shared<ScopedFile>(std::move(scoped_file));
  absl::string_view file_contents(
      reinterpret_cast<const char*>(mapped_file->data()),
      mapped_file->length());

  // From just the path.
  ASSERT_THAT(GetFileFormatFromPath(model_path_str),
              IsOkAndHolds(FileFormat::LITERT_LM));
  ASSERT_THAT(GetFileFormat(model_path_str, nullptr),
              IsOkAndHolds(FileFormat::LITERT_LM));

  // From just the scoped file. Cannot read the file extension in this case.
  ASSERT_THAT(GetFileFormatFromFileContents(file_contents),
              IsOkAndHolds(FileFormat::LITERT_LM));
  ASSERT_THAT(GetFileFormat("", shared_scoped_file),
              IsOkAndHolds(FileFormat::LITERT_LM));

  // From both the path and the scoped file. Should be able to figure it out.
  ASSERT_THAT(GetFileFormat(model_path_str, shared_scoped_file),
              IsOkAndHolds(FileFormat::LITERT_LM));
}

}  // namespace
}  // namespace litert::lm
