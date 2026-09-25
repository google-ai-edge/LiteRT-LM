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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/text_source.h"
#include "omni/tts/tts_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::tts {

// Suffix of a GenericBinaryData section holding a text normalization rule
// table. What precedes it is the language code, so the Mandarin table ships as
// the "cmn-textnorm" section. See omni/tts/text_normalizer.h for the format.
constexpr absl::string_view kTextNormSectionSuffix = "-textnorm";

// Suffix of a GenericBinaryData section holding a packed CJK lexicon blob
// (e.g. "ja-lexicon", "zh-lexicon").
constexpr absl::string_view kLexiconSectionSuffix = "-lexicon";

// Compiles and populates all Kokoro-82M LiteRT models into shared
// ModelResources.
//
// When `resources` carries a .litertlm container, the models are compiled from
// the container's TF_LITE_ACOUSTIC / TF_LITE_VOCODER sections, espeak-ng data
// packaged in the container is unpacked into `cache_dir` and recorded in
// `config.espeak_data_dir`, and any `<lang>-textnorm` or `<lang>-lexicon`
// sections are recorded as zero-copy views in `config.text_norm_rules` and
// `config.cjk_lexicons`.
//
// args
// - config: Kokoro model configuration. `espeak_data_dir`, `text_norm_rules`,
//   and `cjk_lexicons` are populated when the corresponding sections are
//   present in the model container.
// - model_folder: Path to the directory containing the Kokoro models.
// - cache_dir: Path to the directory for caching model data.
// - backend: Backend to use for model execution.
// - num_threads: Number of threads to use for model execution.
// - env: LiteRT environment.
// - resources: ModelResources to add compiled models to.
//
// returns
// - absl::OkStatus on success, or error status on failure.
absl::Status InitKokoroResources(KokoroModelConfig& config,
                                 absl::string_view model_folder,
                                 absl::string_view cache_dir,
                                 lm::Backend backend, int num_threads,
                                 ::litert::Environment& env,
                                 ModelResources& resources);

// Revises `text_chunk_config` for Kokoro's target bucket capacity.
TextChunkConfig ReviseTextChunkConfigForKokoro(
    const KokoroModelConfig& config, TextChunkConfig text_chunk_config);

// Instantiates all stage components for a Kokoro TTS inference session.
//
// args
// - config: Kokoro model configuration.
// - model_folder: Path to the directory containing the Kokoro models.
// - text_source: StreamTextSource providing text chunks for the session.
// - resources: Shared ModelResources container with compiled models.
//
// returns
// - TtsSession::Components containing all stage components on success, or
// error status on failure.
absl::StatusOr<TtsSession::Components> CreateKokoroComponents(
    const KokoroModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<StreamTextSource> absl_nonnull text_source,
    std::shared_ptr<ModelResources> resources);

// Returns the list of available Kokoro voice profile names (e.g. "af_heart",
// "ef_dora", "zf_xiaobei") discovered from GenericBinaryData sections in
// `lm_resources` or voice files (*.bin) in `model_folder/voices` /
// `model_folder`.
//
// args
// - model_folder: Optional path to the directory containing voice `.bin` files.
// - lm_resources: Optional pointer to a loaded `.litertlm` `ModelResources`
//   container to inspect for embedded voice sections.
//
// returns
// - Sorted vector of discovered Kokoro voice identifiers (without `.bin`
//   suffixes).
std::vector<std::string> GetAvailableKokoroVoices(
    absl::string_view model_folder = "",
    const lm::ModelResources* lm_resources = nullptr);

// Returns the canonical Kokoro (`espeak-ng`) language code corresponding to a
// Kokoro voice identifier (e.g. "ef_dora" -> "es", "if_sara" -> "it",
// "af_heart" -> "en-us", "zf_xiaobei" -> "cmn").
//
// args
// - voice_name: Kokoro voice identifier (e.g. "af_heart", "zf_xiaobei").
//
// returns
// - Canonical language code (e.g. "en-us", "es", "cmn"), or an empty string if
//   `voice_name` does not match a known Kokoro voice prefix.
std::string GetKokoroVoiceLanguage(absl::string_view voice_name);

// Returns the default Kokoro voice profile name for a given language code (e.g.
// "es" -> "ef_dora", "it" -> "if_sara", "cmn" -> "zf_xiaobei",
// "en-us" -> "af_heart").
//
// args
// - language_code: Canonical Kokoro language code or BCP-47 tag (defaults to
//   English US when empty).
//
// returns
// - Default voice identifier for the specified language (e.g. "af_heart").
std::string GetDefaultKokoroVoice(absl::string_view language_code = "");

// Converts a BCP-47 language tag or language name (e.g. "en-US", "en-GB", "es",
// "zh-CN", "pt-BR") into the corresponding canonical Kokoro (`espeak-ng`)
// language code (e.g. "en-us", "en-gb", "es", "cmn", "pt-br").
//
// args
// - language: BCP-47 language tag, ISO code, or language name.
//
// returns
// - Canonical Kokoro language code, or an empty string if `language` is empty
//   or not supported by Kokoro.
std::string ToKokoroLanguageCode(absl::string_view language);

// Converts a canonical Kokoro (`espeak-ng`) language code (e.g. "en-us",
// "en-gb", "es", "cmn", "pt-br") to its canonical BCP-47 language tag (e.g.
// "en-US", "en-GB", "es", "zh-CN", "pt-BR").
//
// args
// - kokoro_code: Canonical Kokoro language code (e.g. "cmn", "en-us").
//
// returns
// - Canonical BCP-47 language tag (e.g. "zh-CN", "en-US"), or an empty string
//   if `kokoro_code` is unrecognized.
std::string KokoroCodeToBcp47(absl::string_view kokoro_code);

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_KOKORO_FACTORY_H_
