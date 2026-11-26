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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_READ_DATA_FN_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_READ_DATA_FN_UTIL_H_

#include <cstdint>
#include <memory>

#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {

// Creates a ReadDataFn that can be used to construct a ModelData object from
// a DataStream.
//
//   `section_data_stream`: The DataStream to read from. Usually corresponds to
//     a TFLite model section.
//   `section_size`: The size of the section.
odml::infra::llm_utils::ModelData::ReadDataFn MakeReadDataFn(
    std::shared_ptr<DataStream> section_data_stream, uint64_t section_size);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_READ_DATA_FN_UTIL_H_
