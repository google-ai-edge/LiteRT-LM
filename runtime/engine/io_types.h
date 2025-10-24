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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/litert_status_util.h"

namespace litert::lm {

// A container to host the input text.
class InputText {
 public:
  // Constructs an InputText from a raw text string or a TensorBuffer of token
  // ids. The InputText takes ownership of the provided data.
  explicit InputText(std::variant<std::string, TensorBuffer> data)
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
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns the raw text string. Returns an error if the text is preprocessed.
  absl::StatusOr<absl::string_view> GetRawTextString() const;

  // Returns the preprocessed text tensor. Returns an error if the text is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedTextTensor() const;

  // Creates a copy of the InputText.
  // If the text is preprocessed, the copy will be a TensorBuffer shallow copy.
  // Otherwise, the copy will be a string byte deep copy.
  absl::StatusOr<InputText> CreateCopy() const;

 private:
  std::variant<std::string, TensorBuffer> data_;
};

// A container to host the input image.
class InputImage {
 public:
  // Constructs an InputImage from a raw image bytes string or a TensorBuffer of
  // processed image bytes. The InputImage takes ownership of the provided data.
  explicit InputImage(std::variant<std::string, TensorBuffer> data)
      : data_(std::move(data)) {}

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
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns the raw image bytes. Returns an error if the image is preprocessed.
  absl::StatusOr<absl::string_view> GetRawImageBytes() const;

  // Returns the preprocessed image tensor. Returns an error if the image is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedImageTensor() const;

  // Creates a copy of the InputImage.
  // If the image is preprocessed, the copy will be a TensorBuffer shallow copy.
  // Otherwise, the copy will be a string byte deep copy.
  absl::StatusOr<InputImage> CreateCopy() const;

 private:
  std::variant<std::string, TensorBuffer> data_;
};

// A container to host the input audio.
class InputAudio {
 public:
  // Constructs an InputAudio from a raw audio bytes string or a TensorBuffer of
  // processed audio bytes. The InputAudio takes ownership of the provided data.
  explicit InputAudio(std::variant<std::string, TensorBuffer> data)
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
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns the raw audio bytes. Returns an error if the audio is preprocessed.
  absl::StatusOr<absl::string_view> GetRawAudioBytes() const;

  // Returns the preprocessed audio tensor. Returns an error if the audio is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedAudioTensor() const;

  // Creates a copy of the InputAudio.
  // If the audio is preprocessed, the copy will be a TensorBuffer shallow copy.
  // Otherwise, the copy will be a string byte deep copy.
  absl::StatusOr<InputAudio> CreateCopy() const;

 private:
  std::variant<std::string, TensorBuffer> data_;
};

// A container to host the input data. Will be extended to support more input
// types in the future.
using InputData = std::variant<InputText, InputImage, InputAudio>;

// Creates a copy of the InputData.
inline absl::StatusOr<InputData> CreateInputDataCopy(const InputData& data) {
  if (const auto* input_text = std::get_if<InputText>(&data)) {
    return input_text->CreateCopy();
  } else if (const auto* input_image = std::get_if<InputImage>(&data)) {
    return input_image->CreateCopy();
  } else if (const auto* input_audio = std::get_if<InputAudio>(&data)) {
    return input_audio->CreateCopy();
  }
  return absl::FailedPreconditionError(
      "The InputData is not a InputText, InputImage, or InputAudio.");
}

// A container to host the model responses.
class Responses {
 public:
  explicit Responses() = default;
  explicit Responses(std::vector<std::string> response_texts,
                     std::vector<float> scores = {})
      : response_texts_(std::move(response_texts)),
        scores_(std::move(scores)) {};

  // Returns the const texts vector.
  const std::vector<std::string>& GetTexts() const { return response_texts_; }

  // Returns the const scores vector.
  const std::vector<float>& GetScores() const { return scores_; }

  // Returns the mutable texts vector.
  std::vector<std::string>& GetMutableTexts() { return response_texts_; };

  // Returns the mutable scores vector.
  std::vector<float>& GetMutableScores() { return scores_; };

 private:
  // The output vector of response tokens (as strings).
  std::vector<std::string> response_texts_;

  // The output vector of scores for each response text. The "score" is pulled
  // from the probability of the last token in the response text.
  std::vector<float> scores_;
};
std::ostream& operator<<(std::ostream& os, const Responses& responses);

// Class to store the data for a single turn of the benchmark. A "turn" is
// defined as a single RunPrefill or RunDecode call.
struct BenchmarkTurnData {
  absl::Duration duration;  // Duration of this entire operation/turn.
  uint64_t num_tokens;      // The number of tokens processed in this turn.
  BenchmarkTurnData(uint64_t tokens, absl::Duration dur);
};
std::ostream& operator<<(std::ostream& os, const BenchmarkTurnData& data);

// Class to store and manage comprehensive performance benchmark information for
// LLMs.
class BenchmarkInfo {
 public:
  explicit BenchmarkInfo(const proto::BenchmarkParams& benchmark_params);
  const proto::BenchmarkParams& GetBenchmarkParams() const;

  // --- Methods to record data ---
  // Time the start and end of a phase in the initialization. The phase name
  // should be a string that uniquely identifies the phase. Otherwise, the
  // methods will return an error.
  absl::Status TimeInitPhaseStart(const std::string& phase_name);
  absl::Status TimeInitPhaseEnd(const std::string& phase_name);
  // Time the start and end of a prefill/decode turn. The num_prefill_tokens
  // should be the number of tokens processed in this turn. The method will
  // return an error if the methods are called out of order (i.e. one end after
  // one start).
  absl::Status TimePrefillTurnStart();
  absl::Status TimePrefillTurnEnd(uint64_t num_prefill_tokens);
  absl::Status TimeDecodeTurnStart();
  absl::Status TimeDecodeTurnEnd(uint64_t num_decode_tokens);
  // Time the duration between two consecutive marks. Useful for profiling the
  // pipeline at a specific point. For example:
  //   RETURN_IF_ERROR(benchmark_info.TimeMarkDelta("sampling"));
  //   ... actual sampling logics ...
  //   RETURN_IF_ERROR(benchmark_info.TimeMarkDelta("sampling"));
  //
  // The method will return the duration as the time delta between the two
  // TimeMarkDelta("sampling") calls. The duration will be stored / recorded for
  // each unique mark name.
  absl::Status TimeMarkDelta(const std::string& mark_name);

  // --- Getters for raw data ---
  const std::map<std::string, absl::Duration>& GetInitPhases() const;
  const std::map<std::string, absl::Duration>& GetMarkDurations() const;

  // --- Calculated metrics and getters for Prefill ---
  uint64_t GetTotalPrefillTurns() const;
  const BenchmarkTurnData& GetPrefillTurn(int turn_index) const;
  double GetPrefillTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for Decode ---
  uint64_t GetTotalDecodeTurns() const;
  const BenchmarkTurnData& GetDecodeTurn(int turn_index) const;
  double GetDecodeTokensPerSec(int turn_index) const;

  // --- Gets the time to the first token ---
  // Note that the first time to token doesn't include the time for
  // initialization. It is the sum of the prefill time for the first turn and
  // the time spent for decoding the first token.
  double GetTimeToFirstToken() const;

 private:
  proto::BenchmarkParams benchmark_params_;

  // Map of phase names to start time.
  std::map<std::string, absl::Time> start_time_map_;
  std::map<std::string, absl::Time> mark_time_map_;
  // The current index of the prefill / decode turn.
  int prefill_turn_index_ = 0;
  int decode_turn_index_ = 0;

  std::map<std::string, absl::Duration> init_phases_;
  std::map<std::string, absl::Duration> mark_durations_;
  std::vector<BenchmarkTurnData> prefill_turns_;
  std::vector<BenchmarkTurnData> decode_turns_;
};
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info);

// Configurations used for a single decode request.
class DecodeConfig {
 public:
  // Creates a default DecodeConfig.
  static DecodeConfig CreateDefault();

  // Sets the optional constraint used to guide the generation.
  // `DecodeConfig` does not take ownership of the `constraint`, which must
  // outlives the single generation process.
  void SetConstraint(Constraint* absl_nullable constraint) {
    constraint_ = constraint;
  }

  // Returns a pointer to the constraint, or nullptr if no constraint is set.
  Constraint* absl_nullable GetConstraint() const { return constraint_; }

 private:
  DecodeConfig() = default;

  Constraint* absl_nullable constraint_ = nullptr;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
