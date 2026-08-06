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

#include "support/util/io_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::support {

absl::StatusOr<absl::string_view> InputText::GetRawTextString() const {
  if (std::holds_alternative<std::string>(data_)) {
    return absl::string_view(std::get<std::string>(data_));
  }
  return absl::FailedPreconditionError(
      "The text is preprocessed and does not have raw text bytes.");
}

absl::StatusOr<const ::litert::TensorBuffer*> InputText::GetPreprocessedTextTensor()
    const {
  if (std::holds_alternative<::litert::TensorBuffer>(data_)) {
    return &std::get<::litert::TensorBuffer>(data_);
  }
  return absl::FailedPreconditionError(
      "The text is not preprocessed and does not have a tensor.");
}

absl::StatusOr<InputText> InputText::CreateCopy() const {
  if (std::holds_alternative<std::string>(data_)) {
    return InputText(std::move(std::get<std::string>(data_)));
  } else if (std::holds_alternative<::litert::TensorBuffer>(data_)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone,
                            std::get<::litert::TensorBuffer>(data_).Duplicate());
    return InputText(std::move(tensor_buffer_clone));
  }
  return absl::FailedPreconditionError(
      "The data_ is not a string or a TensorBuffer.");
}

absl::StatusOr<absl::string_view> InputImage::GetRawImageBytes() const {
  if (std::holds_alternative<std::string>(data_)) {
    return absl::string_view(std::get<std::string>(data_));
  }
  if (std::holds_alternative<absl::string_view>(data_)) {
    return std::get<absl::string_view>(data_);
  }
  return absl::FailedPreconditionError(
      "The image is preprocessed and does not have raw image bytes.");
}

absl::StatusOr<const ::litert::TensorBuffer*> InputImage::GetPreprocessedImageTensor()
    const {
  if (std::holds_alternative<::litert::TensorBuffer>(data_)) {
    return &std::get<::litert::TensorBuffer>(data_);
  }
  return absl::FailedPreconditionError(
      "The image is not preprocessed and does not have a tensor.");
}

absl::StatusOr<const absl::flat_hash_map<std::string, ::litert::TensorBuffer>*>
InputImage::GetPreprocessedImageTensorMap() const {
  if (std::holds_alternative<absl::flat_hash_map<std::string, ::litert::TensorBuffer>>(
          data_)) {
    return &std::get<absl::flat_hash_map<std::string, ::litert::TensorBuffer>>(data_);
  }
  return absl::FailedPreconditionError(
      "The image is not preprocessed and does not have a tensor map.");
}

absl::StatusOr<InputImage> InputImage::CreateCopy() const {
  if (std::holds_alternative<std::string>(data_)) {
    return InputImage(std::move(std::get<std::string>(data_)));
  } else if (std::holds_alternative<absl::string_view>(data_)) {
    // Deep copy the string view.
    return InputImage(std::string(std::get<absl::string_view>(data_)));
  } else if (std::holds_alternative<::litert::TensorBuffer>(data_)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone,
                            std::get<::litert::TensorBuffer>(data_).Duplicate());
    return InputImage(std::move(tensor_buffer_clone));
  } else if (std::holds_alternative<
                 absl::flat_hash_map<std::string, ::litert::TensorBuffer>>(data_)) {
    const auto& tensor_buffer_map =
        std::get<absl::flat_hash_map<std::string, ::litert::TensorBuffer>>(data_);
    absl::flat_hash_map<std::string, ::litert::TensorBuffer> tensor_buffer_map_copy;
    for (const auto& [key, value] : tensor_buffer_map) {
      LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone, value.Duplicate());
      tensor_buffer_map_copy.try_emplace(key, std::move(tensor_buffer_clone));
    }
    return InputImage(std::move(tensor_buffer_map_copy));
  }
  return absl::FailedPreconditionError(
      "The data_ is not a string or a TensorBuffer.");
}

absl::StatusOr<absl::string_view> InputAudio::GetRawAudioBytes() const {
  if (std::holds_alternative<std::string>(data_)) {
    return absl::string_view(std::get<std::string>(data_));
  }
  return absl::FailedPreconditionError("The audio is not raw audio bytes.");
}

absl::StatusOr<const ::litert::TensorBuffer*> InputAudio::GetPreprocessedAudioTensor()
    const {
  if (std::holds_alternative<::litert::TensorBuffer>(data_)) {
    return &std::get<::litert::TensorBuffer>(data_);
  }
  return absl::FailedPreconditionError(
      "The audio is not a preprocessed tensor.");
}

absl::StatusOr<absl::Span<const float>> InputAudio::GetPcmFrames() const {
  if (std::holds_alternative<std::vector<float>>(data_)) {
    return std::get<std::vector<float>>(data_);
  }
  return absl::FailedPreconditionError("The audio is not a float vector.");
}

absl::StatusOr<InputAudio> InputAudio::CreateCopy() const {
  if (std::holds_alternative<std::string>(data_)) {
    return InputAudio(std::move(std::get<std::string>(data_)));
  } else if (std::holds_alternative<::litert::TensorBuffer>(data_)) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_clone,
                            std::get<::litert::TensorBuffer>(data_).Duplicate());
    return InputAudio(std::move(tensor_buffer_clone));
  } else if (std::holds_alternative<std::vector<float>>(data_)) {
    return InputAudio(std::get<std::vector<float>>(data_));
  }
  return absl::FailedPreconditionError(
      "The data_ is not a string, TensorBuffer, or float vector.");
}

}  // namespace litert::support
