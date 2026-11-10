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

#include "omni/any2any/minicpm_o/minicpm_o_fan_out_stage.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {

MiniCpmOInputFanOutStage::MiniCpmOInputFanOutStage(
    Stage<OmniSession::Input>* absl_nonnull upstream, bool has_audio_encoder,
    bool has_vision_encoder)
    : FanOutStage<OmniSession::Input>(upstream),
      text_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, TextInput>>())),
      audio_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, AudioInput>>())),
      audio_input_metadata_queue_(AddOutputQueue(
          std::make_unique<SequencedVariantQueue<OmniSession::Input,
                                                 AudioInputMetadata>>())),
      image_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, ImageInput>>())),
      end_of_turn_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>>())),
      end_of_input_queue_(AddOutputQueue(
          std::make_unique<
              SequencedVariantQueue<OmniSession::Input, EndOfInput>>())),
      has_audio_encoder_(has_audio_encoder),
      has_vision_encoder_(has_vision_encoder) {}

SequencedVariantQueue<OmniSession::Input, TextInput>&
MiniCpmOInputFanOutStage::TextQueue() {
  return *text_queue_;
}

SequencedVariantQueue<OmniSession::Input, AudioInput>&
MiniCpmOInputFanOutStage::AudioQueue() {
  return *audio_queue_;
}

SequencedVariantQueue<OmniSession::Input, AudioInputMetadata>&
MiniCpmOInputFanOutStage::AudioInputMetadataQueue() {
  return *audio_input_metadata_queue_;
}

SequencedVariantQueue<OmniSession::Input, ImageInput>&
MiniCpmOInputFanOutStage::ImageQueue() {
  return *image_queue_;
}

SequencedVariantQueue<OmniSession::Input, EndOfInputTurn>&
MiniCpmOInputFanOutStage::EndOfTurnQueue() {
  return *end_of_turn_queue_;
}

SequencedVariantQueue<OmniSession::Input, EndOfInput>&
MiniCpmOInputFanOutStage::EndOfInputQueue() {
  return *end_of_input_queue_;
}

absl::Status MiniCpmOInputFanOutStage::RouteInput(OmniSession::Input input) {
  if (std::holds_alternative<ImageGenInputMetadata>(input)) {
    return absl::UnimplementedError(
        "ImageGenInputMetadata is not yet supported in "
        "MiniCpmOInputFanOutStage.");
  }
  if (!has_audio_encoder_ &&
      (std::holds_alternative<AudioInput>(input) ||
       std::holds_alternative<AudioInputMetadata>(input))) {
    return absl::FailedPreconditionError(
        "Audio input is not supported because audio_encoder is not loaded.");
  }
  if (!has_vision_encoder_ && std::holds_alternative<ImageInput>(input)) {
    return absl::FailedPreconditionError(
        "Image input is not supported because vision_encoder is not loaded.");
  }
  return FanOutStage<OmniSession::Input>::RouteInput(std::move(input));
}

}  // namespace litert::omni::any2any
