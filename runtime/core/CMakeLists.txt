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
#  Tasks
# ==============================================================================
add_litertlm_library(runtime_core_tasks STATIC
  tasks.cc
)
add_library(LiteRTLM::Runtime::Core::Tasks ALIAS runtime_core_tasks)

target_include_directories(runtime_core_tasks
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_tasks
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutor
    LiteRTLM::Runtime::Framework::Threadpool
    LiteRTLM::Runtime::Util::ConvertTensorBuffer

    LITERTLM_DEPS
)

# ==============================================================================
# Pipeline
# ==============================================================================
add_litertlm_library(runtime_core_pipeline STATIC
  pipeline.cc
)
add_library(LiteRTLM::Runtime::Core::Pipeline ALIAS runtime_core_pipeline)

target_include_directories(runtime_core_pipeline
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_pipeline
  PUBLIC
    runtime_core_tasks
    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::ScoringCpuUtil
    LiteRTLM::Runtime::Components::StopTokenDetector
    LiteRTLM::Runtime::Components::TokenIdUtil
    LiteRTLM::Runtime::Components::TokenizerInterface
    LiteRTLM::Runtime::Components::TopPCpuSampler
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorSettings
    LiteRTLM::Runtime::Executor::LlmLitertCompiledModelExecutor
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LITERTLM_DEPS
)

# ==============================================================================
# Session Utils
# ==============================================================================
add_litertlm_library(runtime_core_session_utils STATIC
  session_utils.cc
)
add_library(LiteRTLM::Runtime::Core::SessionUtils ALIAS runtime_core_session_utils)

target_include_directories(runtime_core_session_utils
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_session_utils
  PUBLIC
    LiteRTLM::Runtime::Components::TokenizerInterface
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes
)

# ==============================================================================
# Session Advanced
# ==============================================================================
add_litertlm_library(runtime_core_session_advanced STATIC
  session_advanced.cc
)
add_library(LiteRTLM::Runtime::Core::SessionAdvanced ALIAS runtime_core_session_advanced)

target_include_directories(runtime_core_session_advanced
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_session_advanced
  PUBLIC
    LiteRTLM::Runtime::Core::Pipeline
    LiteRTLM::Runtime::Core::SessionUtils
    LiteRTLM::Runtime::Core::Tasks

    LiteRTLM::Runtime::Components::SamplerInterface
    LiteRTLM::Runtime::Components::SamplerFactory
    LiteRTLM::Runtime::Components::StopTokenDetector
    LiteRTLM::Runtime::Components::TokenizerInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface

    LiteRTLM::Runtime::Engine::Interface
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes

    LiteRTLM::Runtime::Executor::AudioExecutorInterface
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LiteRTLM::Runtime::Executor::LlmExecutorInterface
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Executor::VisionExecutorInterface

    LiteRTLM::Runtime::Framework::Threadpool
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::ExecutorDataUtil
    LiteRTLM::Runtime::Util::ModelTypeUtils
    LiteRTLM::Runtime::Util::TensorBufferUtil

    LITERTLM_DEPS
)

# ==============================================================================
# Engine Advanced Impl
# ==============================================================================
add_litertlm_library(runtime_core_engine_advanced_impl STATIC
  engine_advanced_impl.cc
)
add_library(LiteRTLM::Runtime::Core::EngineAdvancedImpl ALIAS runtime_core_engine_advanced_impl)

target_include_directories(runtime_core_engine_advanced_impl
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_engine_advanced_impl
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Executor
    LiteRTLM::Runtime::Framework::Threadpool
    LiteRTLM::Runtime::Util
    LITERTLM_DEPS
)

# ==============================================================================
# Embedding Engine Impl
# ==============================================================================
add_litertlm_library(runtime_core_embedding_engine_impl STATIC
  embedding_engine_impl.cc
)
add_library(LiteRTLM::Runtime::Core::EmbeddingEngineImpl ALIAS runtime_core_embedding_engine_impl)

target_include_directories(runtime_core_embedding_engine_impl
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_embedding_engine_impl
  PUBLIC
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Executor
    LiteRTLM::Runtime::Framework::Threadpool
    LiteRTLM::Runtime::Util
    LITERTLM_DEPS
)

# ==============================================================================
# Cached Session
# ==============================================================================
add_litertlm_library(runtime_core_cached_session STATIC
  cached_session.cc
)
add_library(LiteRTLM::Runtime::Core::CachedSession
  ALIAS runtime_core_cached_session)

target_include_directories(runtime_core_cached_session
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_core_cached_session
  PUBLIC
    LiteRTLM::Runtime::Core::PrefixCache
    LiteRTLM::Runtime::Core::SessionUtils
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Prefix Cache
# ==============================================================================
add_litertlm_library(runtime_core_prefix_cache STATIC
  prefix_cache.cc
)
add_library(LiteRTLM::Runtime::Core::PrefixCache ALIAS runtime_core_prefix_cache)

target_include_directories(runtime_core_prefix_cache
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_core_prefix_cache
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Facade
# ==============================================================================
add_library(runtime_core_libs INTERFACE)
add_library(LiteRTLM::Runtime::Core ALIAS runtime_core_libs)

target_link_libraries(runtime_core_libs INTERFACE
  LiteRTLM::Runtime::Core::EngineAdvancedImpl
  LiteRTLM::Runtime::Core::EmbeddingEngineImpl
  LiteRTLM::Runtime::Core::Pipeline
  LiteRTLM::Runtime::Core::SessionAdvanced
  LiteRTLM::Runtime::Core::SessionUtils
  LiteRTLM::Runtime::Core::Tasks
  LiteRTLM::Runtime::Core::CachedSession
  LiteRTLM::Runtime::Core::PrefixCache
)