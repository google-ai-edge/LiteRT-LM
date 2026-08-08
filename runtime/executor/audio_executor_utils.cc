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

#include "runtime/executor/audio_executor_utils.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kPrevPrefix = "prev_";
constexpr absl::string_view kFeatureStatesNamePattern = "feature_state";
constexpr absl::string_view kSegmentMaskName = "segment_mask";
constexpr absl::string_view kMaskName = "mask";
constexpr absl::string_view kFeaturesName = "features";

bool IsStreamingEncoder(const std::vector<absl::string_view>& input_names,
                        bool has_adapter) {
  // A huristic to check if the model is a streaming model by checking if the
  // input names contain the prev_mask name.
  return std::any_of(input_names.begin(), input_names.end(),
                     [](absl::string_view input_name) {
                       return absl::StrContains(input_name, kPrevPrefix);
                     }) ||
         // For audio encoder models without an adapter, we assume the model is
         // streamable because there is no internal states for such models, and
         // it can naturally support both streaming and non-streaming modes.
         !has_adapter;
}

}  // namespace

absl::StatusOr<AudioExecutorProperties>
GetAudioExecutorPropertiesFromModelResources(ModelResources& model_resources) {
  AudioExecutorProperties properties;
  ABSL_ASSIGN_OR_RETURN(
      auto audio_encoder_model,
      model_resources.GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw));
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          audio_encoder_model->GetSignatureInputNames());
  auto audio_adapter_model_or =
      model_resources.GetTFLiteModel(ModelType::kTfLiteAudioAdapter);
  bool has_adapter = audio_adapter_model_or.ok();
  properties.is_streaming_model = IsStreamingEncoder(input_names, has_adapter);

  absl::string_view mask_name = properties.is_streaming_model && has_adapter
                                    ? kSegmentMaskName
                                    : kMaskName;

  bool has_mask = std::find(input_names.begin(), input_names.end(),
                            mask_name) != input_names.end();

  int input_sequence_length = 0;
  if (has_mask) {
    LITERT_ASSIGN_OR_RETURN(
        auto mask_tensor_type,
        audio_encoder_model->GetInputTensorType(0, mask_name));
    LITERT_ASSIGN_OR_RETURN(input_sequence_length,
                            mask_tensor_type.Layout().NumElements());
  } else {
    LITERT_ASSIGN_OR_RETURN(auto input_tensor_type,
                            audio_encoder_model->GetInputTensorType(0, 0));
    const auto& dims = input_tensor_type.Layout().Dimensions();
    if (dims.size() < 2) {
      return absl::InvalidArgumentError(
          "Input tensor at index 0 must have at least 2 dimensions");
    }
    input_sequence_length = dims[dims.size() - 2];
  }

  int output_sequence_length;
  if (has_adapter) {
    LITERT_ASSIGN_OR_RETURN(
        auto adapter_output_tensor_type,
        (*audio_adapter_model_or)->GetOutputTensorType(0, 0));
    output_sequence_length =
        adapter_output_tensor_type.Layout().Dimensions()
            [adapter_output_tensor_type.Layout().Dimensions().size() - 2];
  } else {
    if (properties.is_streaming_model) {
      LITERT_ASSIGN_OR_RETURN(
          auto features_tensor_type,
          audio_encoder_model->GetOutputTensorType(0, kFeaturesName));
      output_sequence_length =
          features_tensor_type.Layout()
              .Dimensions()[features_tensor_type.Layout().Dimensions().size() -
                            2];
    } else {
      LITERT_ASSIGN_OR_RETURN(auto output_tensor_type,
                              audio_encoder_model->GetOutputTensorType(0, 0));
      output_sequence_length =
          output_tensor_type.Layout()
              .Dimensions()[output_tensor_type.Layout().Dimensions().size() -
                            2];
    }
  }

  if (properties.is_streaming_model && has_adapter) {
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

    properties.streaming_chunk_size = input_sequence_length;

    properties.audio_shrink_factor =
        (input_sequence_length - properties.streaming_chunk_overlap_size) /
        output_sequence_length;
  } else {
    properties.audio_shrink_factor =
        input_sequence_length / output_sequence_length;
    if (properties.is_streaming_model) {
      properties.streaming_chunk_size = input_sequence_length;
    } else {
      properties.streaming_chunk_size = 0;
    }
  }
  return properties;
}
}  // namespace litert::lm
