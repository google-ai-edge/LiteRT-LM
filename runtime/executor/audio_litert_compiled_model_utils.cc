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

#include "runtime/executor/audio_litert_compiled_model_utils.h"

#include <algorithm>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kPrevMaskName = "prev_mask";
constexpr absl::string_view kFeatureStatesNamePattern = "feature_state";
constexpr absl::string_view kSegmentMaskName = "segment_mask";

bool IsStreamingEncoder(const std::vector<absl::string_view>& input_names) {
  // A huristic to check if the model is a streaming model by checking if the
  // input names contain the prev_mask name.
  return std::any_of(input_names.begin(), input_names.end(),
                     [](absl::string_view input_name) {
                       return absl::StrContains(input_name, kPrevMaskName);
                     });
}
}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorProperties& properties) {
  os << "IsStreamingModel: " << properties.is_streaming_model << std::endl;
  os << "StreamingChunkSize: " << properties.streaming_chunk_size << std::endl;
  os << "StreamingChunkOverlapSize: " << properties.streaming_chunk_overlap_size
     << std::endl;
  return os;
}

absl::StatusOr<AudioExecutorProperties>
GetAudioExecutorPropertiesFromModelResources(ModelResources& model_resources) {
  AudioExecutorProperties properties;
  ASSIGN_OR_RETURN(
      auto audio_encoder_model,
      model_resources.GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw));
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          audio_encoder_model->GetSignatureInputNames());
  properties.is_streaming_model = IsStreamingEncoder(input_names);
  if (properties.is_streaming_model) {
    // Get the feature states tensor type and use it to get the overlap size.
    std::string feature_states_name =
        absl::StrCat(kFeatureStatesNamePattern, "_0");
    LITERT_ASSIGN_OR_RETURN(
        auto feature_states_tensor_type,
        audio_encoder_model->GetInputTensorType(0, feature_states_name),
        _ << "The Audio Streaming Encoder model must have a feature_states "
             "input "
             "buffer.");
    // The overlap size is the number of elements in the feature states tensor,
    // which is 3 for gemma3n.
    LITERT_ASSIGN_OR_RETURN(properties.streaming_chunk_overlap_size,
                            feature_states_tensor_type.Layout().NumElements());

    // Get the segment mask tensor type and use it to get the chunk size.
    LITERT_ASSIGN_OR_RETURN(
        auto segment_mask_tensor_type,
        audio_encoder_model->GetInputTensorType(0, kSegmentMaskName),
        _ << "The Audio Streaming Encoder model must have a segment_mask input "
             "buffer.");
    properties.streaming_chunk_size =
        segment_mask_tensor_type.Layout().Dimensions()
            [segment_mask_tensor_type.Layout().Dimensions().size() - 1];
  }
  return properties;
}
}  // namespace litert::lm
