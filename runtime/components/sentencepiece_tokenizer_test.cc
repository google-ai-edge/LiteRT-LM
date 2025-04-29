#include "runtime/components/sentencepiece_tokenizer.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

std::string GetSentencePieceModelPath() {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          "sentencepiece.model")
      .string();
}

absl::StatusOr<std::string> GetContents(absl::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  if (fd < 0) {
    return absl::NotFoundError(absl::StrCat("File not found: ", path));
  }

  absl::Cleanup fd_closer = [fd]() { close(fd); };  // Called on return.

  int64_t contents_length = lseek(fd, 0, SEEK_END);
  if (contents_length < 0) {
    return absl::InternalError(absl::StrCat("Failed to get length: ", path));
  }

  std::string contents(contents_length, '\0');
  lseek(fd, 0, SEEK_SET);
  char* contents_ptr = contents.data();
  while (contents_length > 0) {
    int read_bytes = read(fd, contents_ptr, contents_length);
    if (read_bytes < 0) {
      return absl::InternalError(absl::StrCat("Failed to read: ", path));
    }
    contents_ptr += read_bytes;
    contents_length -= read_bytes;
  }

  return std::move(contents);
}

TEST(SentencePieceTokenizerTtest, CreateFromFile) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTtest, CreateFromBuffer) {
  auto model_buffer_or = GetContents(GetSentencePieceModelPath());
  EXPECT_TRUE(model_buffer_or.ok());
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromBuffer(*model_buffer_or);
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTtest, Create) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
}

TEST(SentencePieceTokenizerTest, TextToTokenIds) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  absl::string_view text = "How's it going?";
  auto ids_or = tokenizer->TextToTokenIds(text);
  EXPECT_TRUE(ids_or.ok());

  EXPECT_THAT(ids_or.value(),
              ::testing::ElementsAre(224, 24, 8, 66, 246, 18, 2295));
}

TEST(SentencePieceTokenizerTest, TextToTensorBuffer) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  absl::string_view text = "Hello World!";
  auto tensor_or = tokenizer->TextToTensorBuffer(text);
  auto tensor = std::move(tensor_or.value());
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor.TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), ::litert::Dimensions({1, 7}));

  auto copied_data = CopyFromTensorBuffer2D<int>(tensor);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_THAT((*copied_data)[0],
              ::testing::ElementsAre(90, 547, 58, 735, 210, 466, 2294));
}

TEST(SentencePieceTokenizerTest, TextToTensorBufferWithPrependAndPostpend) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  absl::string_view text = "Hello World!";
  auto tensor_or = tokenizer->TextToTensorBuffer(text, {2}, {100});
  auto tensor = std::move(tensor_or.value());
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor.TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), ::litert::Dimensions({1, 9}));

  auto copied_data = CopyFromTensorBuffer2D<int>(tensor);
  EXPECT_TRUE(copied_data.HasValue());
  EXPECT_THAT((*copied_data)[0],
              ::testing::ElementsAre(2, 90, 547, 58, 735, 210, 466, 2294, 100));
}

TEST(SentencePieceTokenizerTest, TokenIdsToText) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  const std::vector<int> ids = {90, 547, 58, 735, 210, 466, 2294};
  auto text_or = tokenizer->TokenIdsToText(ids);
  EXPECT_TRUE(text_or.ok());

  EXPECT_EQ(text_or.value(), "▁Hello▁World!");
}

TEST(SentencePieceTokenizerTest, TensorBufferToText) {
  auto tokenizer_or =
      SentencePieceTokenizer::CreateFromFile(GetSentencePieceModelPath());
  EXPECT_TRUE(tokenizer_or.ok());
  auto tokenizer = std::move(tokenizer_or.value());

  const std::vector<int> ids = {90,  547, 58, 735, 210, 466, 2294,
                                224, 24,  8,  66,  246, 18,  2295};
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer tensor_buffer,
                              CopyToTensorBuffer<int>(ids, {2, 7}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer_type,
                              tensor_buffer.TensorType());
  EXPECT_EQ(tensor_buffer_type.Layout().Dimensions(),
            ::litert::Dimensions({2, 7}));

  auto texts_or = tokenizer->TensorBufferToText(tensor_buffer);
  EXPECT_TRUE(texts_or.ok());
  EXPECT_EQ(texts_or.value().size(), 2);
  EXPECT_EQ(texts_or.value()[0], "▁Hello▁World!");
  EXPECT_EQ(texts_or.value()[1], "▁How's▁it▁going?");
}

}  // namespace
}  // namespace litert::lm
