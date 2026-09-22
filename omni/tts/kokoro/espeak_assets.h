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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_ESPEAK_ASSETS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_ESPEAK_ASSETS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"

namespace litert::omni::tts::kokoro {

// Name of the GenericBinaryData section holding the espeak-ng data archive.
inline constexpr char kEspeakNgSectionName[] = "espeak-ng";

// Name of the directory espeak-ng data is unpacked into.
inline constexpr char kEspeakDataDirName[] = "espeak-ng-data";

// Unpacks the espeak-ng data archive stored in a .litertlm container onto disk
// so that espeak-ng, which only reads from the filesystem, can use it.
//
// The archive must be a zip of the espeak-ng data files with stored
// (uncompressed) entries. A leading "espeak-ng-data/" path component is
// stripped, so both layouts are accepted.
//
// Extraction happens at most once per destination: when the destination
// already contains the "phontab" table file, the existing data is reused. The
// archive is unpacked into a temporary sibling directory that is renamed into
// place, so an interrupted run never leaves a partially unpacked directory
// behind.
//
// args
// - lm_resources: Container to read the espeak-ng section from.
// - cache_dir: Directory to unpack into; the system temporary directory is
//   used when empty.
//
// returns
// - Path of the unpacked espeak-ng data directory, a NotFound error if the
//   container has no espeak-ng section, or another error if unpacking failed.
absl::StatusOr<std::string> UnpackEspeakDataFromLitertLm(
    lm::ModelResources& lm_resources, absl::string_view cache_dir);

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_ESPEAK_ASSETS_H_
