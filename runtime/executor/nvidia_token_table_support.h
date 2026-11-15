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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_TOKEN_TABLE_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_TOKEN_TABLE_SUPPORT_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

// The text embedder's token table as the NVIDIA dispatch takes it for the
// decode step that feeds itself (its token_table_c_api.h): the INT4 or INT2
// table [vocab, dim] of the embedder's tfl.embedding_lookup, its scales (one
// FP32 per row, or one FP16 per block of scale_block columns) and the
// multiplier of the tfl.mul that follows the lookup. The pointers are the
// embedder's loaded weights; a table kept out of memory (an external buffer
// the model file holds) has a null pointer and the tensor's place in the
// model, from which RegisterNvidiaTokenTable finds its bytes in the file.
struct NvidiaTokenTable {
  uint32_t vocab = 0;
  uint32_t dim = 0;
  uint32_t bits = 4;
  const void* table = nullptr;
  size_t table_bytes = 0;  // vocab * dim * bits / 8, whether loaded or not
  uint32_t table_subgraph = 0;
  uint32_t table_tensor = 0;
  const void* scales = nullptr;
  size_t scales_bytes = 0;
  uint32_t scale_block = 0;
  float multiplier = 1.0f;
};

// The token table of an embedder model, or nullopt: no lookup of an INT4 or
// INT2 table with such scales in the model.
absl::StatusOr<std::optional<NvidiaTokenTable>> DescribeNvidiaTokenTable(
    const litert::Model& embedder);

// The per-layer embedder's tables (the Gemma E models): the tables of the
// tfl.embedding_lookups whose outputs a tfl.concatenation joins, in its
// order, each INT4 or INT2 [vocab, width] with one FP32 scale per row and no
// zero points, and the multiplier of a tfl.mul after the concatenation (1
// when the concatenation is the output). The pointers are the model's loaded
// weights.
struct NvidiaPerLayerTables {
  uint32_t vocab = 0;
  uint32_t width = 0;
  uint32_t bits = 4;
  std::vector<const void*> tables;
  size_t table_bytes = 0;  // each
  std::vector<const float*> row_scales;
  float multiplier = 1.0f;
};

// The per-layer tables of a per-layer embedder model, or nullopt: no such
// concatenation of lookups, tables of another kind, or something other than
// a multiply by a scalar after the concatenation.
absl::StatusOr<std::optional<NvidiaPerLayerTables>>
DescribeNvidiaPerLayerTables(const litert::Model& per_layer_embedder);

// Registers the text embedder's token table of `resources`, and the
// per-layer embedder's tables when there is one, with the NVIDIA dispatch
// library the runtime has loaded, for a decode step that awaits them, and
// tells the dispatch the local attention's sliding window when the executor
// metadata has one. Whether the dispatch took the token table: false when
// there is nothing to do (no NVIDIA dispatch loaded, no text embedder or no
// token table in it, no decode step awaiting one of its shape). The dispatch
// takes the tables for the decode step of the model the calling thread
// created last and reads them from `resources` when that step first runs in
// a decode loop: call this on the thread that created the executor's model,
// right after creating it, with resources that outlive the executor.
absl::StatusOr<bool> RegisterNvidiaTokenTable(
    const litert::Environment& env, ModelResources& resources,
    const proto::ExecutorMetadata* executor_metadata);

// RegisterNvidiaTokenTable for the executor: a failure costs the decode step
// that feeds itself, not the executor, and is logged here.
void RegisterNvidiaTokenTables(
    const litert::Environment& env, ModelResources& resources,
    const proto::ExecutorMetadata* executor_metadata);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NVIDIA_TOKEN_TABLE_SUPPORT_H_
