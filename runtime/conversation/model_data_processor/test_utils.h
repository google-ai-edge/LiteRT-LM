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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_TEST_UTILS_H_

#include <string>
#include <variant>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {

std::string GetTestdataPath(const std::string& file_name);

std::string GetModelPath(absl::string_view relative_path);

std::string GetImageTestdataPath(const std::string& file_name);

std::string ReadFile(absl::string_view path);

absl::StatusOr<std::string> GetContents(const std::string& path);

// Custom matchers for InputData variant items.

MATCHER_P(HasInputText, text_input, "") {
  if (!std::holds_alternative<InputText>(arg)) {
    return false;
  }
  auto text_bytes = std::get<InputText>(arg).GetRawTextString();
  if (!text_bytes.ok()) {
    return false;
  }
  return text_bytes.value() == text_input->GetRawTextString().value();
}

MATCHER_P(HasInputImage, image_input, "") {
  if (!std::holds_alternative<InputImage>(arg)) {
    return false;
  }
  if (std::get<InputImage>(arg).GetRawImageBytes().ok()) {
    auto image_bytes = std::get<InputImage>(arg).GetRawImageBytes();
    return image_bytes.value() == image_input->GetRawImageBytes().value();
  }
  if (std::get<InputImage>(arg).GetPreprocessedImageTensor().ok()) {
    auto buffer_span = ReferTensorBufferAsSpan<float>(
        *std::get<InputImage>(arg).GetPreprocessedImageTensor().value());
    if (!buffer_span.HasValue()) {
      return false;
    }
    auto expected_buffer_span = ReferTensorBufferAsSpan<float>(
        *image_input->GetPreprocessedImageTensor().value());
    if (!expected_buffer_span.HasValue()) {
      return false;
    }
    absl::Span<const float> buffer_span_val = *buffer_span;
    absl::Span<const float> expected_buffer_span_val = *expected_buffer_span;
    return buffer_span_val == expected_buffer_span_val;
  }
  return true;
}

MATCHER_P(HasInputAudio, audio_input, "") {
  if (!std::holds_alternative<InputAudio>(arg)) {
    return false;
  }
  if (std::get<InputAudio>(arg).GetRawAudioBytes().ok()) {
    auto audio_bytes = std::get<InputAudio>(arg).GetRawAudioBytes();
    return audio_bytes.value() == audio_input->GetRawAudioBytes().value();
  }
  if (std::get<InputAudio>(arg).GetPreprocessedAudioTensor().ok()) {
    auto buffer_span = ReferTensorBufferAsSpan<float>(
        *std::get<InputAudio>(arg).GetPreprocessedAudioTensor().value());
    if (!buffer_span.HasValue()) {
      return false;
    }
    auto expected_buffer_span = ReferTensorBufferAsSpan<float>(
        *audio_input->GetPreprocessedAudioTensor().value());
    if (!expected_buffer_span.HasValue()) {
      return false;
    }
    absl::Span<const float> buffer_span_val = *buffer_span;
    absl::Span<const float> expected_buffer_span_val = *expected_buffer_span;
    return buffer_span_val == expected_buffer_span_val;
  }
  return true;
}

MATCHER(HasInputAudioEnd, "") {
  return std::holds_alternative<InputAudioEnd>(arg);
}

MATCHER(HasInputImageEnd, "") {
  return std::holds_alternative<InputImageEnd>(arg);
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_TEST_UTILS_H_
