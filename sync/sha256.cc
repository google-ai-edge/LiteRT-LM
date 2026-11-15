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

#include "sync/sha256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm::sync {
namespace {

inline uint32_t RotR(uint32_t a, uint32_t b) {
  return (a >> b) | (a << (32 - b));
}

inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}

inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t Sig0(uint32_t x) {
  return RotR(x, 2) ^ RotR(x, 13) ^ RotR(x, 22);
}

inline uint32_t Sig1(uint32_t x) {
  return RotR(x, 6) ^ RotR(x, 11) ^ RotR(x, 25);
}

inline uint32_t Theta0(uint32_t x) {
  return RotR(x, 7) ^ RotR(x, 18) ^ (x >> 3);
}

inline uint32_t Theta1(uint32_t x) {
  return RotR(x, 17) ^ RotR(x, 19) ^ (x >> 10);
}

}  // namespace

void Sha256Hasher::Reset() {
  state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  bit_len_ = 0;
  buf_len_ = 0;
}

void Sha256Hasher::Update(const void* data, size_t length) {
  if (length == 0) return;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  if (buf_len_ > 0) {
    size_t to_copy = std::min<size_t>(64 - buf_len_, length);
    std::memcpy(buffer_.data() + buf_len_, p, to_copy);
    buf_len_ += to_copy;
    p += to_copy;
    length -= to_copy;
    if (buf_len_ == 64) {
      Transform(buffer_.data());
      bit_len_ += 512;
      buf_len_ = 0;
    }
  }
  while (length >= 64) {
    Transform(p);
    bit_len_ += 512;
    p += 64;
    length -= 64;
  }
  if (length > 0) {
    std::memcpy(buffer_.data(), p, length);
    buf_len_ = length;
  }
}

std::array<uint8_t, 32> Sha256Hasher::Final() {
  std::array<uint8_t, 32> hash{};
  size_t i = buf_len_;

  // Pad the buffer: 0x80 followed by zeroes, then 64-bit length.
  if (buf_len_ < 56) {
    buffer_[i++] = 0x80;
    while (i < 56) buffer_[i++] = 0x00;
  } else {
    buffer_[i++] = 0x80;
    while (i < 64) buffer_[i++] = 0x00;
    Transform(buffer_.data());
    buffer_.fill(0);
  }

  bit_len_ += buf_len_ * 8;
  buffer_[63] = static_cast<uint8_t>(bit_len_);
  buffer_[62] = static_cast<uint8_t>(bit_len_ >> 8);
  buffer_[61] = static_cast<uint8_t>(bit_len_ >> 16);
  buffer_[60] = static_cast<uint8_t>(bit_len_ >> 24);
  buffer_[59] = static_cast<uint8_t>(bit_len_ >> 32);
  buffer_[58] = static_cast<uint8_t>(bit_len_ >> 40);
  buffer_[57] = static_cast<uint8_t>(bit_len_ >> 48);
  buffer_[56] = static_cast<uint8_t>(bit_len_ >> 56);
  Transform(buffer_.data());

  for (i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      hash[j * 4 + i] =
          static_cast<uint8_t>((state_[j] >> (24 - i * 8)) & 0x000000ff);
    }
  }
  Reset();
  return hash;
}

std::string Sha256Hasher::FinalHex() {
  std::array<uint8_t, 32> hash = Final();
  std::string hex;
  hex.reserve(64);
  for (uint8_t b : hash) {
    absl::StrAppendFormat(&hex, "%02x", b);
  }
  return hex;
}

void Sha256Hasher::Transform(const uint8_t* chunk) {
  static constexpr std::array<uint32_t, 64> kK = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  std::array<uint32_t, 64> m;
  for (size_t i = 0, j = 0; i < 16; ++i, j += 4) {
    m[i] = (static_cast<uint32_t>(chunk[j]) << 24) |
           (static_cast<uint32_t>(chunk[j + 1]) << 16) |
           (static_cast<uint32_t>(chunk[j + 2]) << 8) |
           (static_cast<uint32_t>(chunk[j + 3]));
  }
  for (size_t i = 16; i < 64; ++i) {
    m[i] = Theta1(m[i - 2]) + m[i - 7] + Theta0(m[i - 15]) + m[i - 16];
  }

  uint32_t a = state_[0];
  uint32_t b = state_[1];
  uint32_t c = state_[2];
  uint32_t d = state_[3];
  uint32_t e = state_[4];
  uint32_t f = state_[5];
  uint32_t g = state_[6];
  uint32_t h = state_[7];

  for (size_t i = 0; i < 64; ++i) {
    uint32_t t1 = h + Sig1(e) + Ch(e, f, g) + kK[i] + m[i];
    uint32_t t2 = Sig0(a) + Maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

std::string ComputeSha256Hex(absl::string_view data) {
  Sha256Hasher hasher;
  hasher.Update(data);
  return hasher.FinalHex();
}

}  // namespace litert::lm::sync
