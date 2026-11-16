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

#include "runtime/engine/litert_lm_main.h"

#include <cstdlib>
#include <filesystem>  // NOLINT: Required for test path manipulation.
#include <fstream>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "runtime/engine/shared_flags.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;

std::string GetTestModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) /
          "litert_lm/runtime/testdata/test_lm.litertlm")
      .string();
}

TEST(LiteRtLmMainTest, EmptyModelPathPrintsUsageAndReturnsFailure) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_model_path, "");

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_FAILURE);
  EXPECT_THAT(err.str(), HasSubstr("--model_path must be specified."));
  EXPECT_THAT(err.str(),
              HasSubstr("Usage: litert_lm_main --model_path=<path>"));
}

TEST(LiteRtLmMainTest, ConflictingPromptFlagsReturnsFailure) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_model_path, GetTestModelPath());
  absl::SetFlag(&FLAGS_input_prompt, "Hello");
  absl::SetFlag(&FLAGS_input_prompt_file, "prompt.txt");

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_FAILURE);
  EXPECT_THAT(err.str(), HasSubstr("Only one of --input_prompt and "
                                   "--input_prompt_file can be specified."));
}

TEST(LiteRtLmMainTest, MissingInputPromptFileReturnsFailure) {
  absl::FlagSaver flag_saver;
  const std::string missing_file =
      (std::filesystem::path(::testing::TempDir()) / "nonexistent_prompt.txt")
          .string();
  absl::SetFlag(&FLAGS_model_path, GetTestModelPath());
  absl::SetFlag(&FLAGS_input_prompt, "");
  absl::SetFlag(&FLAGS_input_prompt_file, missing_file);

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_FAILURE);
  EXPECT_THAT(err.str(), HasSubstr("Could not open input_prompt_file:"));
}

TEST(LiteRtLmMainTest, InvalidBackendWithDefaultPromptReturnsFailure) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_model_path, GetTestModelPath());
  absl::SetFlag(&FLAGS_backend, "invalid_backend");
  absl::SetFlag(&FLAGS_input_prompt, "");
  absl::SetFlag(&FLAGS_input_prompt_file, "");

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_FAILURE);
  EXPECT_THAT(err.str(), HasSubstr("Error:"));
}

TEST(LiteRtLmMainTest, GpuBackendFailurePrintsCpuFallbackHint) {
  absl::FlagSaver flag_saver;
  const std::string missing_model =
      (std::filesystem::path(::testing::TempDir()) / "missing_model.litertlm")
          .string();
  absl::SetFlag(&FLAGS_model_path, missing_model);
  absl::SetFlag(&FLAGS_backend, "gpu");
  absl::SetFlag(&FLAGS_input_prompt, "Hello");
  absl::SetFlag(&FLAGS_input_prompt_file, "");

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_FAILURE);
  EXPECT_THAT(err.str(), HasSubstr("Error:"));
  EXPECT_THAT(err.str(), HasSubstr("Hint: pass --backend=cpu"));
}

TEST(LiteRtLmMainTest, RunsCpuInferenceWithInputPrompt) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_model_path, GetTestModelPath());
  absl::SetFlag(&FLAGS_backend, "cpu");
  absl::SetFlag(&FLAGS_input_prompt, "Hello");
  absl::SetFlag(&FLAGS_input_prompt_file, "");
  absl::SetFlag(&FLAGS_max_output_tokens, 1);

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_SUCCESS) << err.str();
  EXPECT_EQ(err.str(), "");
  EXPECT_THAT(out.str(), HasSubstr("input_prompt: Hello"));
}

TEST(LiteRtLmMainTest, RunsCpuInferenceWithInputPromptFile) {
  absl::FlagSaver flag_saver;
  const std::string prompt_file =
      (std::filesystem::path(::testing::TempDir()) / "prompt.txt").string();
  {
    std::ofstream file(prompt_file);
    file << "File prompt";
  }

  absl::SetFlag(&FLAGS_model_path, GetTestModelPath());
  absl::SetFlag(&FLAGS_backend, "cpu");
  absl::SetFlag(&FLAGS_input_prompt, "");
  absl::SetFlag(&FLAGS_input_prompt_file, prompt_file);
  absl::SetFlag(&FLAGS_max_output_tokens, 1);

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunLiteRtLmFromFlags(out, err), EXIT_SUCCESS) << err.str();
  EXPECT_EQ(err.str(), "");
  EXPECT_THAT(out.str(), HasSubstr("input_prompt: File prompt"));
}

}  // namespace
}  // namespace litert::lm
