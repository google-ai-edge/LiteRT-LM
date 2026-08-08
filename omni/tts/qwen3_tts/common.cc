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

#include "omni/tts/qwen3_tts/common.h"

#include <cctype>
#include <string>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts {

absl::StatusOr<int> GetLanguageId(absl::string_view language) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, int>>
      kLanguageMap({
          {"chinese", 2055},  {"english", 2050},    {"german", 2053},
          {"italian", 2070},  {"portuguese", 2071}, {"spanish", 2054},
          {"japanese", 2064}, {"korean", 2064},     {"french", 2061},
          {"russian", 2069},
      });
  std::string lang_lower = std::string(language);
  absl::c_transform(lang_lower, lang_lower.begin(), ::tolower);
  auto it = kLanguageMap->find(lang_lower);
  if (it == kLanguageMap->end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported language: ", language));
  }
  return it->second;
}

}  // namespace litert::omni::tts
