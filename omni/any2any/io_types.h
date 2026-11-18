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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_

#include <vector>

namespace litert::omni::any2any {

// Continuous audio embeddings produced by a single encoder chunk invocation.
struct EncodedAudioChunk {
  // Flattened `[valid_tokens, llm_hidden_size]` float32 embeddings.
  std::vector<float> embeddings;
  // Number of valid embedding tokens in this chunk.
  int valid_tokens = 0;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_IO_TYPES_H_
