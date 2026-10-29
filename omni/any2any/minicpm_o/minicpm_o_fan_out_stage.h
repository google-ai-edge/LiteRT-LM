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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FAN_OUT_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FAN_OUT_STAGE_H_

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

// Fan-out stage that routes `OmniSession::Input` items to modality queues.
class MiniCpmOInputFanOutStage : public FanOutStage<OmniSession::Input> {
 public:
  // Constructs a `MiniCpmOInputFanOutStage` pulling inputs from `upstream` and
  // registering output queues for `TextInput`, `AudioInput`,
  // `AudioInputMetadata`, `ImageInput`, `EndOfInputTurn`, and `EndOfInput`.
  //
  // Args:
  //   upstream: Non-null upstream stage producing `OmniSession::Input` items.
  explicit MiniCpmOInputFanOutStage(
      Stage<OmniSession::Input>* absl_nonnull upstream);
  ~MiniCpmOInputFanOutStage() override = default;

  // Returns the output queue for `TextInput` items.
  //
  // Returns:
  //   Reference to the `SequencedVariantQueue` storing `Sequenced<TextInput>`.
  SequencedVariantQueue<OmniSession::Input, TextInput>& TextQueue();

  // Returns the output queue for `AudioInput` items.
  //
  // Returns:
  //   Reference to the `SequencedVariantQueue` storing `Sequenced<AudioInput>`.
  SequencedVariantQueue<OmniSession::Input, AudioInput>& AudioQueue();

  // Returns the output queue for `AudioInputMetadata` items.
  //
  // Returns:
  //   Reference to the `SequencedVariantQueue` storing
  //   `Sequenced<AudioInputMetadata>`.
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata>&
  AudioInputMetadataQueue();

  // Returns the output queue for `ImageInput` items.
  //
  // Returns:
  //   Reference to the `SequencedVariantQueue` storing `Sequenced<ImageInput>`.
  SequencedVariantQueue<OmniSession::Input, ImageInput>& ImageQueue();

  // Returns the output queue for `EndOfInputTurn` control items.
  //
  // Returns:
  //   Reference to the `SequencedVariantQueue` storing
  //   `Sequenced<EndOfInputTurn>`.
  SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>& EndOfTurnQueue();

  // Returns the output queue for `EndOfInput` end-of-stream control items.
  //
  // Returns:
  //   Reference to the `SequencedVariantQueue` storing `Sequenced<EndOfInput>`.
  SequencedVariantQueue<OmniSession::Input, EndOfInput>& EndOfInputQueue();

 protected:
  absl::Status RouteInput(OmniSession::Input input) override;

 private:
  SequencedVariantQueue<OmniSession::Input, TextInput>* text_queue_;
  SequencedVariantQueue<OmniSession::Input, AudioInput>* audio_queue_;
  SequencedVariantQueue<OmniSession::Input, AudioInputMetadata>*
      audio_input_metadata_queue_;
  SequencedVariantQueue<OmniSession::Input, ImageInput>* image_queue_;
  SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>* end_of_turn_queue_;
  SequencedVariantQueue<OmniSession::Input, EndOfInput>* end_of_input_queue_;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_FAN_OUT_STAGE_H_
