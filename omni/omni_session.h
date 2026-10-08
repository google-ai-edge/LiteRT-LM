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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_SESSION_H_

#include <memory>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/stage.h"

namespace litert::omni {

// Pure interface for a unified session processing multimodal inputs (`Input`)
// and producing multimodal outputs (`Output`).
class OmniSession {
 public:
  using EndOfInput = ::litert::omni::EndOfInput;
  using TextInput = ::litert::omni::TextInput;
  using AudioInputMetadata = ::litert::omni::AudioInputMetadata;
  using AudioInput = ::litert::omni::AudioInput;
  using ImageGenInputMetadata = ::litert::omni::ImageGenInputMetadata;
  using Input = ::litert::omni::Input;

  // Base stage producing `Input` items for an `OmniSession`.
  class InputSource : public SingleThreadedStageWithDeque<Input> {
   public:
    ~InputSource() override = default;
  };

  using EndOfOutput = ::litert::omni::EndOfOutput;
  using TextOutput = ::litert::omni::TextOutput;
  using AudioOutput = ::litert::omni::AudioOutput;
  using ImageOutput = ::litert::omni::ImageOutput;
  using Output = ::litert::omni::Output;
  using OutputCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<Output>)>;

  virtual ~OmniSession() = default;

  // Resets session state for a new stream, just like the state created by the
  // factory.
  virtual void Reset() = 0;

  // Flushes remaining buffered output at the end of a stream. It's different
  // from `Reset()` in that it does not reset the session's internal state.
  virtual absl::StatusOr<Output> Flush() = 0;

  // Synchronously processes the pipeline and returns one `Output` chunk per
  // call. Returns `absl::NotFoundError` when no output is ready yet (more
  // input is needed or `Flush()` should be called), and
  // `absl::OutOfRangeError` when the input source is exhausted and the stream
  // has ended.
  virtual absl::StatusOr<Output> ProcessNext() = 0;

  // Asynchronously processes inputs from the session's `InputSource` using the
  // underlying session's thread pool and emits `Output` chunks to `callback`.
  // Returns `absl::AlreadyExistsError` if async processing is already active.
  virtual absl::Status ProcessAsync(OutputCallback callback) = 0;
};

// An `OmniSession::InputSource` implementation that queues `OmniSession::Input`
// chunks for consumption by an `OmniSession`.
//
// Note: When used with `ProcessAsync()`, callers should push initial inputs
// before starting `ProcessAsync()`, as an empty input queue with no running
// stages signals `absl::OutOfRangeError` (end of stream) to the scheduler.
class PushInputSource : public OmniSession::InputSource {
 public:
  PushInputSource() = default;
  ~PushInputSource() override = default;

  // Appends an `OmniSession::Input` chunk to be consumed by the session.
  // Always returns `absl::OkStatus()` for an unbounded in-memory queue
  // (returns `absl::Status` to allow future bounded-queue / validation errors).
  absl::Status PushInput(OmniSession::Input input) {
    PushOutput(std::move(input));
    return absl::OkStatus();
  }

  // Signals the end of the input stream by pushing `OmniSession::EndOfInput`.
  void Finish() { PushOutput(OmniSession::EndOfInput{}); }

 protected:
  // Data has already been pushed into the output queue in PushInput().
  bool NeedScheduleInternal() const override { return false; }

  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    return absl::OkStatus();
  }
};

// Pure interface for creating `OmniSession` instances bound to an
// `OmniSession::InputSource`.
class OmniSessionFactory {
 public:
  virtual ~OmniSessionFactory() = default;

  virtual absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source) = 0;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_SESSION_H_
