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
    Stage<OmniSession::Input>* absl_nonnull upstream)
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
              SequencedVariantQueue<OmniSession::Input, EndOfInput>>())) {}

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
  if (std::holds_alternative<ImageInput>(input) ||
      std::holds_alternative<ImageGenInputMetadata>(input)) {
    return absl::UnimplementedError(
        "ImageInput is not yet supported in MiniCpmOInputFanOutStage.");
  }
  return FanOutStage<OmniSession::Input>::RouteInput(std::move(input));
}

}  // namespace litert::omni::any2any
