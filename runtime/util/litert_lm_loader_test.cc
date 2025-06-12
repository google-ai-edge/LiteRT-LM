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

#include "runtime/util/litert_lm_loader.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <utility>

#include <gtest/gtest.h>
#include "runtime/components/model_resources.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

namespace {

TEST(LitertLmLoaderTest, InitializeWithSentencePieceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_FALSE(loader.GetHuggingFaceTokenizer());
  ASSERT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  ASSERT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);
  ASSERT_GT(loader.GetLlmMetadata().Size(), 0);
}

TEST(LitertLmLoaderTest, InitializeWithHuggingFaceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_GT(loader.GetHuggingFaceTokenizer()->Size(), 0);
  ASSERT_FALSE(loader.GetSentencePieceTokenizer());
}

TEST(LitertLmLoaderTest, CannotLoadHuggingFaceTokenizerAfterClearing) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_GT(loader.GetHuggingFaceTokenizer()->Size(), 0);
  loader.ClearHuggingFaceTokenizerJson();
  ASSERT_FALSE(loader.GetHuggingFaceTokenizer());
}
}  // namespace
}  // namespace litert::lm
