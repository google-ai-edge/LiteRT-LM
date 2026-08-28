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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_ACOUSTIC_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_ACOUSTIC_STAGE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/kokoro/kokoro_io_types.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/kokoro/phonemizer.h"

namespace litert::omni::tts {

// Text Frontend & Unified Acoustic Prediction for Kokoro-82M TTS
class KokoroAcousticStage
    : public SingleThreadedStageWithDeque<KokoroAcousticOutput> {
 public:
  // Creates a KokoroAcousticStage instance and initializes its resources.
  //
  // args
  // - text_source: Stage that provides the input text string.
  // - config: Model configuration for Kokoro-82M TTS.
  // - model_folder: Directory containing model assets and phonemizer data.
  // - resources: Shared ModelResources container with compiled models.
  //
  // returns
  // - Unique pointer to the created KokoroAcousticStage on success, or an error
  //   status on failure.
  static absl::StatusOr<std::unique_ptr<KokoroAcousticStage>> Create(
      Stage<std::string>* absl_nonnull text_source,
      const KokoroModelConfig& config, absl::string_view model_folder,
      std::shared_ptr<ModelResources> absl_nonnull resources);

  ~KokoroAcousticStage() override = default;

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_.HasOutput();
  }

  // Executes one step of acoustic stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  struct InputIndices {
    size_t phoneme_ids = 0;
    size_t speaker_style = 1;
    size_t phoneme_length = 2;
  };

  struct OutputIndices {
    size_t acoustic_features = 0;
    size_t pitch_contour = 1;
    size_t energy_contour = 2;
    size_t speech_frame_length = 3;
  };

  KokoroAcousticStage(Stage<std::string>* text_source, KokoroModelConfig config,
                      absl::string_view model_folder,
                      std::shared_ptr<ModelResources> resources)
      : text_source_(*text_source),
        config_(std::move(config)),
        model_folder_(std::move(model_folder)),
        resources_(std::move(resources)) {}

  Stage<std::string>& text_source_;
  KokoroModelConfig config_;
  std::string model_folder_;
  std::shared_ptr<ModelResources> resources_;

  std::shared_ptr<CompiledModel> acoustic_model_;
  std::vector<float> voice_pack_;
  std::unique_ptr<KokoroPhonemizer> phonemizer_;

  InputIndices input_indices_;
  OutputIndices output_indices_;

  std::vector<TensorBuffer> acoustic_input_buffers_;
  std::vector<TensorBuffer> acoustic_output_buffers_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_ACOUSTIC_STAGE_H_
