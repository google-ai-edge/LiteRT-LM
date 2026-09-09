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

#include "runtime/util/streamed_weights_manager.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {
namespace {

ModelType g_currently_compiling_model = ModelType::kUnknown;

// Global map of stored weights streams, mapped by ModelType.
std::unordered_map<ModelType, std::shared_ptr<DataStream>>&
GetStoredWeightsStreams() {
  static auto* const m =
      new std::unordered_map<ModelType, std::shared_ptr<DataStream>>();
  return *m;
}

}  // namespace

void SetCurrentlyCompilingModel(ModelType model_type) {
  g_currently_compiling_model = model_type;
}

ModelType GetCurrentlyCompilingModel() { return g_currently_compiling_model; }

void StoreWeightsStream(ModelType model_type,
                        std::shared_ptr<DataStream> stream) {
  GetStoredWeightsStreams()[model_type] = std::move(stream);
}

absl::Status ReadStoredWeights(int model_type_int, uint64_t offset,
                               uint64_t size, void* buffer) {
  ModelType model_type = static_cast<ModelType>(model_type_int);
  auto& streams = GetStoredWeightsStreams();
  auto it = streams.find(model_type);
  if (it == streams.end() || it->second == nullptr) {
    return absl::NotFoundError(absl::StrCat(
        "Stored weights stream not found for model type: ", model_type_int));
  }
  return it->second->ReadAndDiscard(buffer, offset, size);
}

absl::Status ClearStoredWeightsStream(ModelType model_type) {
  auto& streams = GetStoredWeightsStreams();
  auto it = streams.find(model_type);
  if (it != streams.end()) {
    if (it->second) {
      (void)it->second->Discard(0, UINT64_MAX);
    }
    streams.erase(it);
  }
  return absl::OkStatus();
}

absl::Status ClearStoredWeightsStreams() {
  auto& streams = GetStoredWeightsStreams();
  for (auto& [model_type, stream] : streams) {
    if (stream) {
      (void)stream->Discard(0, UINT64_MAX);
    }
  }
  streams.clear();
  return absl::OkStatus();
}

}  // namespace litert::lm
