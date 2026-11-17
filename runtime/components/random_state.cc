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

#include "runtime/components/random_state.h"

#include <cstdint>
#include <random>

namespace litert::lm {

namespace {

// Seeds a lenience stream with the session seed and the stream's salt, so that
// the streams of one seed differ from each other.
std::mt19937_64 SaltedEngine(int32_t seed, uint64_t salt) {
  return std::mt19937_64(static_cast<uint64_t>(static_cast<uint32_t>(seed)) ^
                         salt);
}

}  // namespace

RandomState RandomState::FromSeed(int32_t seed) {
  return RandomState{
      .sampling = std::default_random_engine(seed),
      .drafter_lenience = SaltedEngine(seed, /*salt=*/0x4c454e49),  // "LENI"
      .draft_verifier_lenience =
          SaltedEngine(seed, /*salt=*/0x44524654),  // "DRFT"
  };
}

}  // namespace litert::lm
