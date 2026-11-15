# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# KV Cache
# ==============================================================================
add_litertlm_library(runtime_executor_litert_state STATIC
  state.cc
)
add_library(LiteRTLM::Runtime::Executor::Litert::State
    ALIAS runtime_executor_litert_state)

target_include_directories(runtime_executor_litert_state
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_litert_state
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_executor_litert_libs INTERFACE)
add_library(LiteRTLM::Runtime::Executor::Litert ALIAS runtime_executor_litert_libs)

target_link_libraries(runtime_executor_litert_libs INTERFACE
  LiteRTLM::Runtime::Executor::Litert::State
)