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
# LLM LiteRT NPU Compiled Model Executor Utils
# ==============================================================================
add_litertlm_library(runtime_executor_npu_llm_litert_npu_compiled_model_executor_utils STATIC
  llm_litert_npu_compiled_model_executor_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
  ALIAS runtime_executor_npu_llm_litert_npu_compiled_model_executor_utils)
add_library(LiteRTLM::Runtime::Executor::LlmNpuCompiledModelUtils
  ALIAS runtime_executor_npu_llm_litert_npu_compiled_model_executor_utils)

target_include_directories(runtime_executor_npu_llm_litert_npu_compiled_model_executor_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_npu_llm_litert_npu_compiled_model_executor_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Executor::LitertCompiledModelExecutorUtils
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::ModelResourcesTask
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT NPU Rope
# ==============================================================================
add_litertlm_library(runtime_executor_npu_llm_litert_npu_rope STATIC
  llm_litert_npu_rope.cc
)
add_library(LiteRTLM::Runtime::Executor::Npu::Rope
  ALIAS runtime_executor_npu_llm_litert_npu_rope)

target_include_directories(runtime_executor_npu_llm_litert_npu_rope
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_npu_llm_litert_npu_rope
  PUBLIC
    LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT NPU Mask
# ==============================================================================
add_litertlm_library(runtime_executor_npu_llm_litert_npu_mask STATIC
  llm_litert_npu_mask.cc
)
add_library(LiteRTLM::Runtime::Executor::Npu::Mask
  ALIAS runtime_executor_npu_llm_litert_npu_mask)

target_include_directories(runtime_executor_npu_llm_litert_npu_mask
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_npu_llm_litert_npu_mask
  PUBLIC
    LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT NPU KV Cache
# ==============================================================================
add_litertlm_library(runtime_executor_npu_llm_litert_npu_kv_cache STATIC
  llm_litert_npu_kv_cache.cc
)
add_library(LiteRTLM::Runtime::Executor::Npu::KVCache
  ALIAS runtime_executor_npu_llm_litert_npu_kv_cache)

target_include_directories(runtime_executor_npu_llm_litert_npu_kv_cache
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_npu_llm_litert_npu_kv_cache
  PUBLIC
    LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Generated::Protobuf
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# LLM LiteRT NPU Embedder
# ==============================================================================
add_litertlm_library(runtime_executor_npu_llm_litert_npu_embedder STATIC
  llm_litert_npu_embedder.cc
)
add_library(LiteRTLM::Runtime::Executor::Npu::Embedder
  ALIAS runtime_executor_npu_llm_litert_npu_embedder)

target_include_directories(runtime_executor_npu_llm_litert_npu_embedder
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_npu_llm_litert_npu_embedder
  PUBLIC
    LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
    LiteRTLM::Runtime::Components::ModelResourcesInterface
    LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorProcessedTokens
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_executor_npu_libs INTERFACE)
add_library(LiteRTLM::Runtime::Executor::Npu ALIAS runtime_executor_npu_libs)

target_link_libraries(runtime_executor_npu_libs INTERFACE
  LiteRTLM::Runtime::Executor::Npu::CompiledModelExecutorUtils
  LiteRTLM::Runtime::Executor::Npu::KVCache
  LiteRTLM::Runtime::Executor::Npu::Rope
  LiteRTLM::Runtime::Executor::Npu::Mask
  LiteRTLM::Runtime::Executor::Npu::Embedder
)

