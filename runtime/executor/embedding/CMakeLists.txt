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
# Embedding Executor Settings
# ==============================================================================
add_litertlm_library(runtime_executor_embedding_executor_settings STATIC
  embedding_executor_settings.cc
)
add_library(LiteRTLM::Runtime::Executor::EmbeddingExecutorSettings ALIAS runtime_executor_embedding_executor_settings)

target_include_directories(runtime_executor_embedding_executor_settings
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_embedding_executor_settings
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_executor_embedding_libs INTERFACE)
add_library(LiteRTLM::Runtime::Executor::Embedding ALIAS runtime_executor_embedding_libs)

target_link_libraries(runtime_executor_embedding_libs INTERFACE
  LiteRTLM::Runtime::Executor::EmbeddingExecutorSettings
)
