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

#include "schema/capabilities/capabilities_c.h"

#include <fstream>
#include <ios>

#include "schema/capabilities/capabilities.h"

struct LiteRtLmLoadedFile {
  std::ifstream stream;
};

extern "C" {

LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path) {
  if (litertlm_path == nullptr) {
    return nullptr;
  }
  auto* file = new LiteRtLmLoadedFile;
  file->stream.open(litertlm_path, std::ios::binary);
  if (!file->stream.is_open()) {
    delete file;
    return nullptr;
  }
  return file;
}

void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file) {
  delete loaded_file;
}

bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file) {
  if (loaded_file == nullptr) {
    return false;
  }
  auto result = litert::lm::schema::capabilities::HasSpeculativeDecodingSupport(
      loaded_file->stream);
  if (!result.ok()) {
    return false;
  }
  return *result;
}

}  // extern "C"
