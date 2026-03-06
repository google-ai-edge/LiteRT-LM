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
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/status_macros.h"

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
      std::variant<std::string, absl::string_view, TensorBuffer,
                   absl::flat_hash_map<std::string, TensorBuffer>>
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
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns true if the image is preprocessed into a TensorBuffer map.
  bool IsTensorBufferMap() const {
    return std::holds_alternative<
        absl::flat_hash_map<std::string, TensorBuffer>>(data_);
  }

  // Returns the raw image bytes. Returns an error if the image is preprocessed.
  absl::StatusOr<absl::string_view> GetRawImageBytes() const;

  // Returns the preprocessed image tensor. Returns an error if the image is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedImageTensor() const;

  // Returns the preprocessed image tensor map. Returns an error if the image is
  // not preprocessed.
  absl::StatusOr<const absl::flat_hash_map<std::string, TensorBuffer>*>
  GetPreprocessedImageTensorMap() const;

  // Creates a copy of the InputImage.
  // If the image is preprocessed, the copy will be a TensorBuffer shallow copy.
  // Otherwise, the copy will be a string byte deep copy.
  absl::StatusOr<InputImage> CreateCopy() const;

 private:
  std::variant<std::string, absl::string_view, TensorBuffer,
               absl::flat_hash_map<std::string, TensorBuffer>>
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
      std::variant<std::string, TensorBuffer, std::vector<float>> data)
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

  // Returns true if the audio is PCM frames.
  bool IsPcmFrames() const {
    return std::holds_alternative<std::vector<float>>(data_);
  }

  // Returns the raw audio bytes. Returns an error if the audio is preprocessed.
  absl::StatusOr<absl::string_view> GetRawAudioBytes() const;

  // Returns the preprocessed audio tensor. Returns an error if the audio is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedAudioTensor() const;

  // Returns the raw audio float vector. Returns an error if the audio is not a
  // float vector.
  absl::StatusOr<absl::Span<const float>> GetPcmFrames() const;

  // Creates a copy of the InputAudio.
  // If the audio is preprocessed, the copy will be a TensorBuffer shallow copy.
  // If the data is a `std::vector<float>`, a deep copy of the vector is made.
  // Otherwise (if it's a string), the copy will be a string byte deep copy.
  absl::StatusOr<InputAudio> CreateCopy() const;

 private:
  std::variant<std::string, TensorBuffer, std::vector<float>> data_;
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

// A struct that holds the scoring output for a single option.
struct ScorerOutput {
  // The score of the option text.
  // NOTE: this is the sum of the scores for each token in the option text.
  double score;
  // Character length of the option text.
  std::optional<int> option_text_char_length;
  // Token length of the option text.
  std::optional<int> option_text_token_length;
};

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

// The state of the task.
enum class TaskState {
  kUnknown,                 // The task is in an unknown state.
  kCreated,                 // The task is created and waiting for other
                            // dependent tasks.
                            // For example, the decode task is waiting for the
                            // prefill task to be done.
  kQueued,                  // The task is queued to be processed.
                            // For example, the decode task is queued to be
                            // processed after the prefill task is done.
  kProcessing,              // The task is being processed.
  kDone,                    // The task is done.
  kMaxNumTokensReached,     // The task is done because the max number of tokens
                            // is reached.
  kFailed,                  // The task is failed.
  kDependentTaskFailed,     // The task was cancelled because a dependent task
                            // failed.
  kCancelled,               // The task is cancelled.
  kDependentTaskCancelled,  // The task was cancelled because a dependent task
                            // was cancelled.
  kLastCallbackQueued,      // The last callback is queued to be called.
                            // This is internal state, will not be called to the
                            // user callback.
};
std::ostream& operator<<(std::ostream& os, const TaskState& task_state);

bool IsTaskEndState(const TaskState& task_state);

// A container to host the model responses.
class Responses {
 public:
  explicit Responses(TaskState task_state,
                     std::vector<std::string> response_texts = {},
                     std::vector<float> scores = {},
                     std::vector<int> token_lengths = {})
      : task_state_(task_state),
        response_texts_(std::move(response_texts)),
        scores_(std::move(scores)) {
    if (!token_lengths.empty()) {
      token_lengths_ = std::move(token_lengths);
    }
  };

  // Returns the task state.
  const TaskState& GetTaskState() const { return task_state_; }

  // Sets the task state.
  void SetTaskState(TaskState task_state) { task_state_ = task_state; }

  // Returns the const texts vector.
  const std::vector<std::string>& GetTexts() const { return response_texts_; }

  // Returns the const scores vector.
  const std::vector<float>& GetScores() const { return scores_; }

  // Returns the mutable texts vector.
  std::vector<std::string>& GetMutableTexts() { return response_texts_; };

  // Returns the mutable scores vector.
  std::vector<float>& GetMutableScores() { return scores_; };

  // Returns the const token lengths vector.
  const std::optional<std::vector<int>>& GetTokenLengths() const {
    return token_lengths_;
  }

  // Returns the mutable token lengths vector.
  std::optional<std::vector<int>>& GetMutableTokenLengths() {
    return token_lengths_;
  };

  // Returns the const token scores vector.
  const std::optional<std::vector<std::vector<float>>>& GetTokenScores() const {
    return token_scores_;
  }

  // Returns the mutable token scores vector.
  std::optional<std::vector<std::vector<float>>>& GetMutableTokenScores() {
    return token_scores_;
  };

 private:
  // The state of the task.
  TaskState task_state_;

  // The output vector of response tokens (as strings).
  std::vector<std::string> response_texts_;

  // The output vector of scores for each response text. The "score" is pulled
  // from the probability of the last token in the response text.
  std::vector<float> scores_;

  // The output vector of token lengths for each response text. Optional.
  std::optional<std::vector<int>> token_lengths_;

  // The output vector of token scores for each response text. Optional.
  std::optional<std::vector<std::vector<float>>> token_scores_;
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

  enum class InitPhase {
    kModelAssets,
    kLlmMetadata,
    kExecutor,
    kTokenizer,
    kSession,
    kConversation,
    kTotal,
  };
  static constexpr absl::string_view InitPhaseToString(InitPhase phase) {
    switch (phase) {
      case InitPhase::kModelAssets:
        return "Init Model assets";
      case InitPhase::kLlmMetadata:
        return "Init LLM metadata";
      case InitPhase::kExecutor:
        return "Init Executor";
      case InitPhase::kTokenizer:
        return "Init Tokenizer";
      case InitPhase::kSession:
        return "Init Session";
      case InitPhase::kConversation:
        return "Init Conversation";
      case InitPhase::kTotal:
        return "Init Total";
    }
  }

  // Time the start and end of an init phase. The method will return an error
  // if the methods are called out of order (i.e. one end after one start).
  // Each phase can only be timed once, and the subsequent calls will return
  // error.
  absl::Status TimeInitPhaseStart(InitPhase phase);
  absl::Status TimeInitPhaseEnd(InitPhase phase);

  // An alternative to TimeInitPhaseStart and TimeInitPhaseEnd. Allows directly
  // recording the duration of a phase. This is useful when the BenchmarkInfo
  // object is not available to mark the start time as needed.
  absl::Status InitPhaseRecord(InitPhase phase, absl::Duration duration);

  // Time the start and end of a prefill/decode turn. The num_prefill_tokens
  // should be the number of tokens processed in this turn. The method will
  // return an error if the methods are called out of order (i.e. one end after
  // one start).
  absl::Status TimePrefillTurnStart();
  absl::Status TimePrefillTurnEnd(uint64_t num_prefill_tokens);
  absl::Status TimeDecodeTurnStart();
  absl::Status TimeDecodeTurnEnd(uint64_t num_decode_tokens);
  absl::Status TimeTextToTokenIdsStart();
  absl::Status TimeTextToTokenIdsEnd(uint64_t num_tokens);
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
  absl::StatusOr<BenchmarkTurnData> GetPrefillTurn(int turn_index) const;
  double GetPrefillTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for Decode ---
  uint64_t GetTotalDecodeTurns() const;
  absl::StatusOr<BenchmarkTurnData> GetDecodeTurn(int turn_index) const;
  double GetDecodeTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for TextToTokenIds ---
  uint64_t GetTotalTextToTokenIdsTurns() const;
  absl::StatusOr<BenchmarkTurnData> GetTextToTokenIdsTurn(int turn_index) const;

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
  // The current index of the prefill / decode / text_to_token_ids turn.
  int prefill_turn_index_ = 0;
  int decode_turn_index_ = 0;
  int text_to_token_ids_turn_index_ = 0;

  std::map<std::string, absl::Duration> init_phases_;
  std::map<std::string, absl::Duration> mark_durations_;
  std::vector<BenchmarkTurnData> prefill_turns_;
  std::vector<BenchmarkTurnData> decode_turns_;
  std::vector<BenchmarkTurnData> text_to_token_ids_turns_;
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

  // Sets the max output tokens.
  void SetMaxOutputTokens(int max_output_tokens) {
    max_output_tokens_ = max_output_tokens;
  }

  // Returns the max output tokens.
  std::optional<int> GetMaxOutputTokens() const { return max_output_tokens_; }

 private:
  DecodeConfig() = default;

  Constraint* absl_nullable constraint_ = nullptr;
  std::optional<int> max_output_tokens_ = std::nullopt;
};

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

std::ostream& operator<<(std::ostream& os,
                         const VisionExecutorProperties& properties);

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

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorProperties& properties);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
