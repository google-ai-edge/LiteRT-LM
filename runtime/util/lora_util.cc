// Copyright 2025 The ODML Authors.
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

#include "runtime/util/lora_util.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

constexpr LazyRE2 kLoRAInputNamePattern = {
    "^(?:(?:query|key|value|post)_w_prime_(?:left|right)|"
    "lora_atten_(?:q|k|v|o)_(?:a|b)_prime_weight)_\\d+$"};

uint64_t AlignByN(uint64_t number, uint64_t n) {
  const uint64_t q = number / n;
  return (number % n == 0 ? q : q + 1) * n;
}

}  // namespace

// Gets an offset and size which will be valid to pass to
// MemoryMappedFile.
absl::StatusOr<std::unique_ptr<MemoryMappedFileWithAutoAlignment>>
MemoryMappedFileWithAutoAlignment::Create(ScopedFile::PlatformFile file,
                                          uint64_t offset, uint64_t size,
                                          absl::string_view key) {
  const size_t kAlignment = MemoryMappedFile::GetOffsetAlignment();
  uint64_t aligned_offset = (offset / kAlignment) * kAlignment;

  uint64_t map_size;
  if (size == 0) {
    map_size = 0;  // Map to the end of the file.
  } else {
    map_size = AlignByN(offset - aligned_offset + size, kAlignment);
  }

  ASSIGN_OR_RETURN(auto region, MemoryMappedFile::Create(file, aligned_offset,
                                                         map_size, key));

  uint64_t internal_offset = offset - aligned_offset;
  uint64_t final_size;
  if (size == 0) {
    final_size = region->length() - internal_offset;
  } else {
    final_size = size;
  }

  return absl::WrapUnique(new MemoryMappedFileWithAutoAlignment(
      std::move(region), internal_offset, final_size));
}

// Returns true if the given name is a LoRA input name for the model.
// The LoRA name is in the format of
// "(query|key|value|post)_w_prime_(left|right)_[0-num_layers)" or
// "lora_atten_(q|k|v|o)_(a|b)_prime_weight_[0-num_layers)".
bool IsLoRAInputName(absl::string_view name) {
  return RE2::FullMatch(name, *kLoRAInputNamePattern);
}

}  // namespace litert::lm
