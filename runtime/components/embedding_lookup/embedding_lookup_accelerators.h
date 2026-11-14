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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_ACCELERATORS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_ACCELERATORS_H_

namespace litert::lm {

// Returns whether the embedding lookup models are allowed to run on the NPU in
// addition to the CPU. This is the case on Android, except on Intel devices
// (x86 and x86_64), where the lookups run on the CPU only.
constexpr bool UseNpuForEmbeddingLookup() {
#if defined(__ANDROID__) && !defined(__x86_64__) && !defined(__i386__)
  return true;
#else
  return false;
#endif
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_EMBEDDING_LOOKUP_EMBEDDING_LOOKUP_ACCELERATORS_H_
