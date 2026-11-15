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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SYNC_SHA256_H_
#define THIRD_PARTY_ODML_LITERT_LM_SYNC_SHA256_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm::sync {

// =============================================================================
// FROZEN IMPLEMENTATION - DO NOT MODIFY
//
// This is a minimal, self-contained implementation of the FIPS 180-4 SHA-256
// specification for integrity checksum verification of model sections and
// files. It has zero external dependencies (no OpenSSL / BoringSSL) to ensure
// portability across on-device targets (Android, iOS, WebAssembly, desktop).
//
// Do not modify cryptographic primitives or mathematical constants.
// Any changes must be verified against NIST CAVP / FIPS 180-4 test vectors.
// =============================================================================

// Lightweight, dependency-free incremental SHA-256 hasher.
class Sha256Hasher {
 public:
  Sha256Hasher() { Reset(); }

  void Reset();
  void Update(const void* data, size_t length);
  void Update(absl::string_view data) { Update(data.data(), data.size()); }

  // Finalizes the SHA-256 digest and returns the raw 32-byte hash.
  std::array<uint8_t, 32> Final();

  // Finalizes the SHA-256 digest and returns the lowercase 64-character hex
  // digest string.
  std::string FinalHex();

 private:
  void Transform(const uint8_t* chunk);

  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> buffer_{};
  uint64_t bit_len_ = 0;
  size_t buf_len_ = 0;
};

// Convenience helper to compute hex SHA-256 string for data.
std::string ComputeSha256Hex(absl::string_view data);

}  // namespace litert::lm::sync

#endif  // THIRD_PARTY_ODML_LITERT_LM_SYNC_SHA256_H_
