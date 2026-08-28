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

#include "omni/tts/kokoro/common.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

absl::StatusOr<std::vector<float>> LoadVoiceEmbedding(
    absl::string_view model_dir, absl::string_view voice_identifier) {
  std::vector<float> embed(kVoicePackSize, 0.0f);

  std::vector<std::filesystem::path> candidate_paths;
  std::filesystem::path base_path = std::string(model_dir);
  if (!voice_identifier.empty()) {
    std::string voice_str(voice_identifier);
    candidate_paths.push_back(voice_str);
    candidate_paths.push_back((base_path / voice_str));
    candidate_paths.push_back((base_path / "voices" / voice_str));
    candidate_paths.push_back(
        (base_path / "voices" / absl::StrCat(voice_identifier, ".bin")));
  }
  candidate_paths.push_back((base_path / "voices" / "af_heart"));
  candidate_paths.push_back((base_path / "voices" / "af_heart.bin"));

  for (const auto& path : candidate_paths) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(embed.data()),
                kVoicePackSize * sizeof(float));
      if (file.gcount() ==
          static_cast<std::streamsize>(kVoicePackSize * sizeof(float))) {
        ABSL_LOG(INFO) << "Loaded voice pack embedding (" << kVoicePackSize
                       << " floats) from: " << path;
        return embed;
      }
    }
  }

  return absl::NotFoundError(
      absl::StrCat("Voice embedding '", voice_identifier,
                   "' not found or incomplete in model_dir: ", model_dir));
}

}  // namespace litert::omni::tts::kokoro
