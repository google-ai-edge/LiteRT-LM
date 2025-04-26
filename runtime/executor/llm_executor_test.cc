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

#include "runtime/executor/llm_executor.h"

#include <atomic>
#include <cstdint>
#include <sstream>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {
namespace {

TEST(LlmExecutorTest, InputsPrint) {
  struct alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
    int32_t d[6] = {1, 2, 3, 4, 5, 6};
  } data;

  auto token_ids = ::litert::TensorBuffer::CreateFromHostMemory(
      ::litert::RankedTensorType(
          ::litert::ElementType::Int32,
          ::litert::Layout(::litert::Dimensions({3, 2}))),
      data.d, 6 * sizeof(int32_t));
  EXPECT_TRUE(token_ids.HasValue());

  Inputs inputs = {.text_input = TextInput{.token_ids = std::move(*token_ids)}};
  std::stringstream oss;
  oss << inputs;
  EXPECT_EQ(oss.str(),
            "token_ids: token_ids: TensorBuffer: [[1, 2], [3, 4], [5, 6]] "
            "shape=(3, 2)\n\n");
}

TEST(LlmExecutorTest, PrefillQueryParamsPrint) {
  std::atomic_bool cancel = true;
  PrefillQueryParams params = {
      .current_step = 10, .wait_for_completion = true, .cancel = &cancel};
  std::stringstream oss;
  oss << params;
  EXPECT_EQ(oss.str(), "current_step: 10\nwait_for_completion: 1\ncancel: 1\n");

  // Test with a null cancel flag.
  oss.str("");
  params.cancel = nullptr;
  oss << params;
  EXPECT_EQ(oss.str(), "current_step: 10\nwait_for_completion: 1");
}

}  // namespace
}  // namespace litert::lm
