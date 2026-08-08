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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_UTIL_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_UTIL_IO_TYPES_H_

#include <cstdint>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/util/status_macros.h"

namespace litert::support {

// A container to host the input text.
class InputText {
 public:
  // Constructs an InputText from a raw text string or a TensorBuffer of token
  // ids. The InputText takes ownership of the provided data.
  explicit InputText(std::variant<std::string, ::litert::TensorBuffer> data)
      : data_(std::move(data)) {}

  // Copy constructor.
  InputText(const InputText& other) = delete;
  // Copy assignment operator.
  InputText& operator=(const InputText& other) = delete;
  // Move constructor.
  InputText(InputText&& other) = default;
  // Move assignment operator.
  InputText& operator=(InputText&& other) = default;

  // Returns true if the text is preprocessed into a TensorBuffer.
  bool IsTensorBuffer() const {
    return std::holds_alternative<::litert::TensorBuffer>(data_);
  }

  // Returns the raw text string. Returns an error if the text is preprocessed.
  absl::StatusOr<absl::string_view> GetRawTextString() const;

  // Returns the preprocessed text tensor. Returns an error if the text is
  // not preprocessed.
  absl::StatusOr<const ::litert::TensorBuffer*> GetPreprocessedTextTensor() const;

  // Creates a copy of the InputText.
  // If the text is preprocessed, the copy will be a TensorBuffer shallow copy.
  // Otherwise, the copy will be a string byte deep copy.
  absl::StatusOr<InputText> CreateCopy() const;

 private:
  std::variant<std::string, ::litert::TensorBuffer> data_;
};

inline std::ostream& operator<<(std::ostream& os, const InputText& input_text) {
  if (input_text.IsTensorBuffer()) {
    os << "[TensorBuffer]";
  } else {
    auto raw_text = input_text.GetRawTextString();
    if (raw_text.ok()) {
      os << *raw_text;
    } else {
      os << "Error getting raw text: " << raw_text.status();
    }
  }
  return os;
}

// A container to host the input image.
class InputImage {
 public:
  // Constructs an InputImage from a raw image bytes string or a TensorBuffer of
  // processed image bytes. The InputImage takes ownership of the provided data.
  explicit InputImage(
      std::variant<std::string, absl::string_view, ::litert::TensorBuffer,
                   absl::flat_hash_map<std::string, ::litert::TensorBuffer>>
          data)
      : data_(std::move(data)) {}
  // Useful for testing with const char* or const char[].
  explicit InputImage(const char* data) : data_(absl::string_view(data)) {}

  // Copy constructor.
  InputImage(const InputImage& other) = delete;
  // Copy assignment operator.
  InputImage& operator=(const InputImage& other) = delete;
  // Move constructor.
  InputImage(InputImage&& other) = default;
  // Move assignment operator.
  InputImage& operator=(InputImage&& other) = default;

  // Returns true if the image is preprocessed into a TensorBuffer.
  bool IsTensorBuffer() const {
    return std::holds_alternative<::litert::TensorBuffer>(data_);
  }

  // Returns true if the image is preprocessed into a TensorBuffer map.
  bool IsTensorBufferMap() const {
    return std::holds_alternative<
        absl::flat_hash_map<std::string, ::litert::TensorBuffer>>(data_);
  }

  // Returns the raw image bytes. Returns an error if the image is preprocessed.
  absl::StatusOr<absl::string_view> GetRawImageBytes() const;

  // Returns the preprocessed image tensor. Returns an error if the image is
  // not preprocessed.
  absl::StatusOr<const ::litert::TensorBuffer*> GetPreprocessedImageTensor() const;

  // Returns the preprocessed image tensor map. Returns an error if the image is
  // not preprocessed.
  absl::StatusOr<const absl::flat_hash_map<std::string, ::litert::TensorBuffer>*>
  GetPreprocessedImageTensorMap() const;

  // Creates a copy of the InputImage.
  // If the image is preprocessed, the copy will be a TensorBuffer shallow copy.
  // Otherwise, the copy will be a string byte deep copy.
  absl::StatusOr<InputImage> CreateCopy() const;

 private:
  std::variant<std::string, absl::string_view, ::litert::TensorBuffer,
               absl::flat_hash_map<std::string, ::litert::TensorBuffer>>
      data_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const InputImage& input_image) {
  os << "[InputImage]";
  return os;
}

// A signal to indicate the end of input image.
class InputImageEnd {
 public:
  explicit InputImageEnd() = default;
};

inline std::ostream& operator<<(std::ostream& os,
                                const InputImageEnd& input_image_end) {
  os << "[InputImageEnd]";
  return os;
}

// A container to host the input audio.
class InputAudio {
 public:
  // Constructs an InputAudio from a raw audio bytes string, a TensorBuffer of
  // processed audio bytes, or a vector of float audio samples. The InputAudio
  // takes ownership of the provided data.
  explicit InputAudio(
      std::variant<std::string, ::litert::TensorBuffer, std::vector<float>> data)
      : data_(std::move(data)) {}

  // Copy constructor.
  InputAudio(const InputAudio& other) = delete;
  // Copy assignment operator.
  InputAudio& operator=(const InputAudio& other) = delete;
  // Move constructor.
  InputAudio(InputAudio&& other) = default;
  // Move assignment operator.
  InputAudio& operator=(InputAudio&& other) = default;

  // Returns true if the audio is preprocessed into a TensorBuffer.
  bool IsTensorBuffer() const {
    return std::holds_alternative<::litert::TensorBuffer>(data_);
  }

  // Returns true if the audio is PCM frames.
  bool IsPcmFrames() const {
    return std::holds_alternative<std::vector<float>>(data_);
  }

  // Returns the raw audio bytes. Returns an error if the audio is preprocessed.
  absl::StatusOr<absl::string_view> GetRawAudioBytes() const;

  // Returns the preprocessed audio tensor. Returns an error if the audio is
  // not preprocessed.
  absl::StatusOr<const ::litert::TensorBuffer*> GetPreprocessedAudioTensor() const;

  // Returns the raw audio float vector. Returns an error if the audio is not a
  // float vector.
  absl::StatusOr<absl::Span<const float>> GetPcmFrames() const;

  // Creates a copy of the InputAudio.
  // If the audio is preprocessed, the copy will be a TensorBuffer shallow copy.
  // If the data is a `std::vector<float>`, a deep copy of the vector is made.
  // Otherwise (if it's a string), the copy will be a string byte deep copy.
  absl::StatusOr<InputAudio> CreateCopy() const;

 private:
  std::variant<std::string, ::litert::TensorBuffer, std::vector<float>> data_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const InputAudio& input_audio) {
  os << "[InputAudio]";
  return os;
}

// A signal to indicate the end of input audio.
class InputAudioEnd {
 public:
  explicit InputAudioEnd() = default;
};

inline std::ostream& operator<<(std::ostream& os,
                                const InputAudioEnd& input_audio_end) {
  os << "[InputAudioEnd]";
  return os;
}

// A container to host the input data. Will be extended to support more input
// types in the future.
using InputData = std::variant<InputText, InputImage, InputAudio, InputImageEnd,
                               InputAudioEnd>;

inline std::ostream& operator<<(std::ostream& os, const InputData& input_data) {
  std::visit([&os](const auto& data) { os << data; }, input_data);
  return os;
}

// Creates a copy of the InputData.
inline absl::StatusOr<InputData> CreateInputDataCopy(const InputData& data) {
  if (const auto* input_text = std::get_if<InputText>(&data)) {
    return input_text->CreateCopy();
  } else if (const auto* input_image = std::get_if<InputImage>(&data)) {
    return input_image->CreateCopy();
  } else if (const auto* input_audio = std::get_if<InputAudio>(&data)) {
    return input_audio->CreateCopy();
  } else if (std::get_if<InputAudioEnd>(&data)) {
    return InputAudioEnd();
  } else if (std::get_if<InputImageEnd>(&data)) {
    return InputImageEnd();
  }
  return absl::FailedPreconditionError(
      "The InputData is not a InputText, InputImage, InputAudio, "
      "InputImageEnd, or InputAudioEnd.");
}

// Creates a copy of the InputData vector.
inline absl::StatusOr<std::vector<InputData>> CreateInputDataVectorCopy(
    const std::vector<InputData>& data) {
  std::vector<InputData> copy;
  copy.reserve(data.size());
  for (const auto& input_data : data) {
    ASSIGN_OR_RETURN(auto input_data_copy, CreateInputDataCopy(input_data));
    copy.push_back(std::move(input_data_copy));
  }
  return copy;
}

// The properties of the audio model. These properties are populated by
// inspecting the LiteRT compiled model and provide information about the model
// parameters.
struct VisionExecutorProperties {
  // The number of tokens representing each image fed into the LLM.
  // Note the start of image token is not counted in this number.
  int num_tokens_per_image = 256;

  // The ratio of the input image patch number to the output image patch
  // number. This is used to calculate the number of image tokens fed into the
  // LLM. For example, if the input image has 2520 patches and the
  // patch_num_shrink_factor is 9, the image tokens fed into the LLM will be
  // 2520 / 9 = 280. Only applicable to models that use transformer encoder,
  // a.k.a. Vision Transformer (ViT).
  std::optional<int> patch_num_shrink_factor = std::nullopt;
};

inline std::ostream& operator<<(std::ostream& os,
                                const VisionExecutorProperties& properties) {
  os << "num_tokens_per_image: " << properties.num_tokens_per_image
     << std::endl;
  os << "patch_num_shrink_factor: "
     << (properties.patch_num_shrink_factor.has_value()
             ? absl::StrCat(properties.patch_num_shrink_factor.value())
             : "not set")
     << std::endl;
  return os;
}

// The properties of the audio model. These properties are populated by
// inspecting the LiteRT compiled model and provide information about the model
// type (static or streaming) and the model parameters (chunk size, overlap
// size).
struct AudioExecutorProperties {
  // Whether the audio model is a streaming model.
  bool is_streaming_model = false;

  // The size of each streaming chunk.
  int streaming_chunk_size = 0;

  // The overlap size of each streaming chunk.
  int streaming_chunk_overlap_size = 0;

  // The factor by which the audio is shrunk after encoding. This is used to
  // calculate the number of audio tokens fed into the LLM. For example, if the
  // input audio has 512 frames and the audio_shrink_factor is 16, the audio
  // embeddings will have 512 / 16 = 32 tokens.
  int audio_shrink_factor = 1;
};

inline std::ostream& operator<<(std::ostream& os,
                                const AudioExecutorProperties& properties) {
  os << "is_streaming_model: " << properties.is_streaming_model << std::endl;
  os << "streaming_chunk_size: " << properties.streaming_chunk_size
     << std::endl;
  os << "streaming_chunk_overlap_size: "
     << properties.streaming_chunk_overlap_size << std::endl;
  os << "audio_shrink_factor: " << properties.audio_shrink_factor << std::endl;
  return os;
}

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_UTIL_IO_TYPES_H_
