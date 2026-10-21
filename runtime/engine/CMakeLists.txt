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
# Engine Interface
# ==============================================================================
add_litertlm_library(runtime_engine_engine_interface INTERFACE)
add_library(LiteRTLM::Runtime::Engine::Interface ALIAS runtime_engine_engine_interface)

target_include_directories(runtime_engine_engine_interface
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_engine_interface
  INTERFACE
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Components::TokenizerInterface
    LITERTLM_DEPS
)

# ==============================================================================
# CPU Affinity Utils
# ==============================================================================
add_litertlm_library(runtime_engine_cpu_affinity_utils STATIC
  cpu_affinity_utils.cc
)
add_library(LiteRTLM::Runtime::Engine::CpuAffinityUtils ALIAS runtime_engine_cpu_affinity_utils)

target_include_directories(runtime_engine_cpu_affinity_utils
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_cpu_affinity_utils
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Engine Settings
# ==============================================================================
add_litertlm_library(runtime_engine_engine_settings STATIC
  engine_settings.cc
)
add_library(LiteRTLM::Runtime::Engine::Settings ALIAS runtime_engine_engine_settings)

target_include_directories(runtime_engine_engine_settings
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_engine_settings
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Executor
    LiteRTLM::Runtime::Util
    LiteRTLM::Schema::Core
    LITERTLM_DEPS
)

# ==============================================================================
# Engine Settings
# ==============================================================================
add_litertlm_library(runtime_engine_embedding_engine_settings STATIC
  embedding_engine_settings.cc
)
add_library(LiteRTLM::Runtime::Engine::EmbeddingEngineSettings ALIAS runtime_engine_embedding_engine_settings)

target_include_directories(runtime_engine_embedding_engine_settings
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_embedding_engine_settings
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Executor
    LiteRTLM::Runtime::Util
    LITERTLM_DEPS
)

# ==============================================================================
# IO Types
# ==============================================================================
add_litertlm_library(runtime_engine_io_types STATIC
  io_types.cc
)
add_library(LiteRTLM::Runtime::Engine::IoTypes ALIAS runtime_engine_io_types)

target_include_directories(runtime_engine_io_types
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_io_types
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Util

    LITERTLM_DEPS
)

# ==============================================================================
# Engine Lib
# ==============================================================================
add_litertlm_library(runtime_engine_litert_lm_lib STATIC
  litert_lm_lib.cc
)
add_library(LiteRTLM::Runtime::Engine::Lib ALIAS runtime_engine_litert_lm_lib)

target_include_directories(runtime_engine_litert_lm_lib
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_litert_lm_lib
  PUBLIC
    LiteRTLM::Runtime::Engine::Interface
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation
    LiteRTLM::Runtime::Core
    LiteRTLM::Runtime::Executor
    LiteRTLM::Runtime::Framework
    LiteRTLM::Runtime::Util
    LITERTLM_DEPS
)

# ==============================================================================
# Shared Flags
# ==============================================================================
add_litertlm_library(runtime_engine_shared_flags STATIC
  shared_flags.cc
)
add_library(LiteRTLM::Runtime::Engine::SharedFlags ALIAS runtime_engine_shared_flags)

target_include_directories(runtime_engine_shared_flags
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_shared_flags
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Embedding Engine Interface
# ==============================================================================
add_litertlm_library(runtime_engine_embedding_engine_interface INTERFACE)
add_library(LiteRTLM::Runtime::Engine::EmbeddingEngineInterface ALIAS runtime_engine_embedding_engine_interface)

target_include_directories(runtime_engine_embedding_engine_interface
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_embedding_engine_interface
  INTERFACE
    LiteRTLM::Runtime::Engine::EmbeddingEngineSettings
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Components::TokenizerInterface
    LiteRTLM::Runtime::Executor::ModelSignatureUtils
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_engine_libs INTERFACE)
add_library(LiteRTLM::Runtime::Engine ALIAS runtime_engine_libs)

target_link_libraries(runtime_engine_libs INTERFACE
  LiteRTLM::Runtime::Engine::Interface
  LiteRTLM::Runtime::Engine::EmbeddingEngineInterface
  LiteRTLM::Runtime::Engine::CpuAffinityUtils
  LiteRTLM::Runtime::Engine::Settings
  LiteRTLM::Runtime::Engine::EmbeddingEngineSettings
  LiteRTLM::Runtime::Engine::IoTypes
  LiteRTLM::Runtime::Engine::Lib
  LiteRTLM::Runtime::Engine::SharedFlags
)