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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MULTIMODAL_PROCESSOR_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MULTIMODAL_PROCESSOR_HELPER_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/engine/io_types.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"

namespace litert::lm {

using ImagePreprocessor = ::litert::support::ImagePreprocessor;
using AudioPreprocessor = ::litert::support::AudioPreprocessor;
using ImagePreprocessParameter = ::litert::support::ImagePreprocessParameter;

// One frame of a `{"type": "video"}` content item. A frame only refers to its
// JSON object; pixel data is read when the frame is preprocessed, so the video
// item must outlive the frame.
struct VideoFrame {
  // The frame object in the video item's `frames` array. Not owned.
  const nlohmann::ordered_json* item = nullptr;
  // Whether the frame carries raw RGB888 HWC `pixels` (otherwise it carries an
  // encoded image as `blob` or `path`).
  bool is_raw = false;
  // Width and height of a raw frame, in pixels. 0 for encoded frames.
  int width = 0;
  int height = 0;
  // Presentation time in the source video, in seconds.
  double timestamp = 0.0;
};

// Loads the frames of a video content item:
//   {"type": "video",
//    "frames": [{"blob" | "path": ..., "timestamp": 0.0},
//               {"pixels": ..., "width": W, "height": H, "timestamp": 0.5}],
//    "fps": 2.0}
// `pixels` is raw RGB888 HWC data as a base64 string or binary JSON value.
//
// Only the frame metadata is validated and read; no pixel data is decoded or
// copied.
//
// args:
//   video_item: The video content item. Must outlive the returned frames.
//   default_fps: Frame rate used to place frames without a `timestamp` (at
//     `index / fps`) when the item has no `fps`.
// returns:
//   The frames in order, or an InvalidArgumentError for a malformed item.
absl::StatusOr<std::vector<VideoFrame>> LoadVideoFrames(
    const nlohmann::ordered_json& video_item, double default_fps);
// The returned frames refer to `video_item`, so it cannot be a temporary.
absl::StatusOr<std::vector<VideoFrame>> LoadVideoFrames(
    nlohmann::ordered_json&& video_item, double default_fps) = delete;

// Preprocesses the frames of one video.
//
// Frames are decoded one at a time into reused scratch buffers, so only one
// decoded frame is resident at a time. All frames must resize to the same size
// (their patches are paired or compared across frames); this is checked from
// the frame dimensions (raw frames) or image header (encoded frames) before
// the frame's pixels are decoded.
//
// args:
//   image_preprocessor: The image preprocessor.
//   frames: The frames, e.g. from LoadVideoFrames().
//   parameter: The image preprocessing parameters for every frame.
// returns:
//   One preprocessed image per frame, or an InvalidArgumentError if a frame is
//   malformed or the frames resize to different sizes.
absl::StatusOr<std::vector<InputImage>> PreprocessVideoFrames(
    ImagePreprocessor& image_preprocessor,
    const std::vector<VideoFrame>& frames,
    const ImagePreprocessParameter& parameter);

// Creates the input of a two-frame vision encoder call: the tensors of
// `frame_0` (e.g. `images`, `positions_xy`) plus the `images` tensor of
// `frame_1` as `images_t1`. Tensor buffers are shared, not copied.
//
// args:
//   frame_0: First preprocessed (patchified) frame.
//   frame_1: Second preprocessed frame, of the same size as `frame_0`.
// returns:
//   The combined image, or an error if the frames are not patchified images.
absl::StatusOr<InputImage> CreateTemporalPairImage(const InputImage& frame_0,
                                                   const InputImage& frame_1);

// Gives every patchified still image in `input_data` an `images_t1` input
// equal to `images` (a still image repeats its frame over the temporal patch).
// Images that already have `images_t1` are unchanged.
//
// args:
//   input_data: The processed prompt inputs, updated in place.
// returns:
//   OK, or an error if a tensor buffer cannot be shared.
absl::Status AddRepeatedTemporalFrame(std::vector<InputData>& input_data);

// Configuration defining how the prompt is parsed and formatted
struct MultimodalPromptProcessingConfig {
  // Regex for matching the delimiters between different modalities.
  std::string delimiter_regex;
  // Regex for matching the image token placeholders.
  std::string image_token_regex;
  // Regex for matching the audio token placeholders.
  std::string audio_token_regex;

  // Image formatting
  // The token that indicates the beginning of an image.
  std::string boi_token;
  // The token that indicates the end of an image.
  std::string eoi_token;
  // The prefix to add before the image token.
  std::string image_prefix;
  // The suffix to add after the image token.
  std::string image_suffix;
  // Whether to add the end of image token after the image content.
  bool add_image_end = false;

  // Audio formatting
  // The token that indicates the beginning of an audio.
  std::string boa_token;
  // The token that indicates the end of an audio.
  std::string eoa_token;
  // The prefix to add before the audio token.
  std::string audio_prefix;
  // The suffix to add after the audio token.
  std::string audio_suffix;
  // Whether to add the end of audio token after the audio content.
  bool add_audio_end = true;

  // Video formatting
  // Regex for matching the video placeholders. Each match is replaced with the
  // inputs that `VideoInputExpander::ExpandVideo()` returns for the next
  // `"type": "video"` content item. Empty means video is not supported.
  std::string video_token_regex;
};

// Regexes matching template-emitted multimodal placeholder tokens for each
// modality supported by a ModelDataProcessor. Empty strings indicate that the
// processor does not split rendered prompts on placeholders for that modality.
struct MultimodalTokenRegexes {
  std::string image_token_regex;
  std::string audio_token_regex;
  std::string video_token_regex;

  bool IsEmpty() const {
    return image_token_regex.empty() && audio_token_regex.empty() &&
           video_token_regex.empty();
  }
};

// Per-instance runtime nonces that replace template-emitted multimodal
// placeholders when multimodal delimiter sanitization is enabled.
struct MultimodalDelimiterNonces {
  std::string delimiter_regex;
  std::string image_token_regex;
  std::string audio_token_regex;
  std::string video_token_regex;
  // Exact runtime nonce placeholder for single-literal image markers (used by
  // processors such as MiniCpmVDataProcessor that split via literal search).
  std::string image_nonce_token;
};

// Replaces the multimodal delimiter regexes in `config` with `nonces` when
// runtime nonces are configured.
inline void ApplyMultimodalDelimiterNonces(
    MultimodalPromptProcessingConfig& config,
    const std::optional<MultimodalDelimiterNonces>& nonces) {
  if (!nonces.has_value()) {
    return;
  }
  config.delimiter_regex = nonces->delimiter_regex;
  config.image_token_regex = nonces->image_token_regex;
  config.audio_token_regex = nonces->audio_token_regex;
  config.video_token_regex = nonces->video_token_regex;
}

// Rewrites multimodal placeholder literals in a prompt template source to
// per-instance random 128-bit runtime nonces and maps rendered nonces back to
// their original placeholder tokens when formatting debug/test strings.
class MultimodalDelimiterSanitizer {
 public:
  // Creates a sanitizer for `template_source` and `token_regexes` using a
  // freshly generated random 128-bit runtime nonce.
  static absl::StatusOr<MultimodalDelimiterSanitizer> Create(
      absl::string_view template_source,
      const MultimodalTokenRegexes& token_regexes);

  // Returns the template source with media placeholder literals replaced by
  // runtime nonces.
  const std::string& rewritten_template() const { return rewritten_template_; }

  // Returns the runtime nonce configuration for the ModelDataProcessor.
  const MultimodalDelimiterNonces& nonces() const { return nonces_; }

  // Returns the placeholder emitted for `"image"` items by
  // `ApplyFallbackPromptTemplate` when Minijinja is disabled.
  absl::string_view fallback_image_placeholder() const {
    return fallback_image_placeholder_;
  }

  // Returns the placeholder emitted for `"audio"` items by
  // `ApplyFallbackPromptTemplate` when Minijinja is disabled.
  absl::string_view fallback_audio_placeholder() const {
    return fallback_audio_placeholder_;
  }

  // Replaces any runtime nonce placeholders in `rendered` back with their
  // original template tokens so callers never see internal runtime nonces.
  std::string RestoreDelimiters(std::string rendered) const;

 private:
  std::string nonce_hex_;
  std::string rewritten_template_;
  MultimodalDelimiterNonces nonces_;
  std::string fallback_image_placeholder_ = "<start_of_image>";
  std::string fallback_audio_placeholder_ = "<start_of_audio>";
  std::vector<std::pair<std::string, std::string>> restore_replacements_;
};

// Context passed to `VideoInputExpander::ExpandVideo()`.
struct VideoExpandContext {
  // Base image preprocessing parameters before applying `visual_token_budget`
  // or `dry_run`.
  const ImagePreprocessParameter& base_image_params;
  // Optional per-video visual token budget from the caller.
  std::optional<int> visual_token_budget = std::nullopt;
  // Whether to only dry run preprocessing (e.g. for token counting).
  bool dry_run = false;
};

// Expands one `"type": "video"` content item into model inputs. Implemented by
// model data processors that support video, which own the model-specific
// frame grouping, video-level budget, and prompt format.
class VideoInputExpander {
 public:
  virtual ~VideoInputExpander() = default;

  // Expands `video_item` into the inputs that replace its placeholder.
  //
  // args:
  //   video_item: The `{"type": "video", "frames": [...], ...}` content item.
  //   context: Base image preprocessing parameters, optional per-video visual
  //     token budget, and dry-run flag.
  // returns:
  //   The InputData sequence (text and preprocessed images) for the video, or
  //   an error if the item is malformed or the model cannot encode video.
  virtual absl::StatusOr<std::vector<InputData>> ExpandVideo(
      const nlohmann::ordered_json& video_item,
      const VideoExpandContext& context) const = 0;
};

// Per-call options for `ProcessMultimodalPrompt()`.
struct MultimodalPromptProcessingOptions {
  // Base image preprocessing parameters (required if `image_preprocessor` is
  // non-null).
  std::optional<ImagePreprocessParameter> image_params = std::nullopt;
  // Optional visual token budget (per image for still images, per video for
  // video items).
  std::optional<int> visual_token_budget = std::nullopt;
  // Whether to only dry run the media preprocessing (see
  // `ImagePreprocessParameter::SetDryRun()` and
  // `AudioPreprocessor::PreprocessDryRun()`). The resulting InputData carries
  // enough information to count tokens (see `CalculateInputDataTokens()`), but
  // must not be fed to the model.
  bool dry_run = false;
  // Expands `"type": "video"` items matched by `config.video_token_regex`.
  // If null, video items are rejected.
  const VideoInputExpander* video_expander = nullptr;
};

// The core generic preprocessing loop.
//
// args:
//   rendered_template_prompt: The rendered template prompt with placeholders
//     for images, audio, and video.
//   messages: The messages to be processed.
//   image_preprocessor: The image preprocessor to use for preprocessing images.
//   audio_preprocessor: The audio preprocessor to use for preprocessing audio.
//   config: The configuration for multimodal prompt processing.
//   options: Per-call image parameters, token budget, dry-run flag, and
//     optional video expander.
// returns:
//   A vector of InputData objects representing the preprocessed prompt, or an
//   error if preprocessing fails or placeholders do not match items.
absl::StatusOr<std::vector<InputData>> ProcessMultimodalPrompt(
    absl::string_view rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    ImagePreprocessor* image_preprocessor,
    AudioPreprocessor* audio_preprocessor,
    const MultimodalPromptProcessingConfig& config,
    const MultimodalPromptProcessingOptions& options);

// Convenience overload of `ProcessMultimodalPrompt()` for callers without
// video support.
//
// args:
//   rendered_template_prompt: The rendered template prompt with placeholders
//     for images and audio.
//   messages: The messages to be processed.
//   image_preprocessor: The image preprocessor to use for preprocessing images.
//   audio_preprocessor: The audio preprocessor to use for preprocessing audio.
//   config: The configuration for multimodal prompt processing.
//   image_params: The parameters for image preprocessing.
//   visual_token_budget: Optional budget for maximum visual tokens per image.
//   dry_run: Whether to only dry run the media preprocessing.
// returns:
//   A vector of InputData objects representing the preprocessed prompt, or an
//   error if preprocessing fails or placeholders do not match items.
absl::StatusOr<std::vector<InputData>> ProcessMultimodalPrompt(
    absl::string_view rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    ImagePreprocessor* image_preprocessor,
    AudioPreprocessor* audio_preprocessor,
    const MultimodalPromptProcessingConfig& config,
    std::optional<ImagePreprocessParameter> image_params = std::nullopt,
    std::optional<int> visual_token_budget = std::nullopt,
    bool dry_run = false);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MULTIMODAL_PROCESSOR_HELPER_H_
