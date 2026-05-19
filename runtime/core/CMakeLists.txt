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


set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# 1. Tasks
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
    LiteRTLM::Runtime::Components::Sampler::Interface
    LiteRTLM::Runtime::Components::ScoringCpuUtil
    LiteRTLM::Runtime::Components::StopTokenDetector
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint

    runtime_engine_io_types

    runtime_executor_llm_executor
    runtime_executor_llm_executor_io_types
    runtime_executor_llm_executor_settings
    runtime_executor_llm_litert_compiled_model_executor

    runtime_framework_threadpool
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# 2. Pipeline
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
    LiteRTLM::Runtime::Components::Sampler::Interface
    LiteRTLM::Runtime::Components::ScoringCpuUtil
    LiteRTLM::Runtime::Components::StopTokenDetector
    LiteRTLM::Runtime::Components::TokenIdUtil
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Components::Sampler::TopP
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    runtime_engine_io_types
    runtime_executor_executor_settings_base
    runtime_executor_llm_executor
    runtime_executor_llm_executor_io_types
    runtime_executor_llm_executor_settings
    runtime_executor_llm_litert_compiled_model_executor
    runtime_util_convert_tensor_buffer
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# 3. Session Utils
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
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    runtime_engine_engine_settings
    runtime_engine_io_types
    runtime_util_litert_status_util
)

# ==============================================================================
# 4. Session Basic
# ==============================================================================
add_litertlm_library(runtime_core_session_basic STATIC
  session_basic.cc
)
add_library(LiteRTLM::Runtime::Core::SessionBasic ALIAS runtime_core_session_basic)

target_include_directories(runtime_core_session_basic
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_session_basic
  PUBLIC
    runtime_core_pipeline
    runtime_core_session_utils
    runtime_core_tasks

    LiteRTLM::Runtime::Components::Sampler::Interface
    LiteRTLM::Runtime::Components::Sampler::Factory
    LiteRTLM::Runtime::Components::StopTokenDetector
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint

    runtime_engine_engine_interface
    runtime_engine_engine_settings
    runtime_engine_io_types

    runtime_executor_audio_executor
    runtime_executor_executor_settings_base
    runtime_executor_llm_executor
    runtime_executor_llm_executor_io_types
    runtime_executor_vision_executor

    runtime_framework_threadpool
    runtime_util_convert_tensor_buffer
    runtime_util_executor_data_util
    runtime_util_litert_status_util
    runtime_util_model_type_utils
    runtime_util_tensor_buffer_util

    LITERTLM_DEPS
)

# ==============================================================================
# 5. Session Factory
# ==============================================================================
add_litertlm_library(runtime_core_session_factory STATIC
  session_factory.cc
)
add_library(LiteRTLM::Runtime::Core::SessionFactory ALIAS runtime_core_session_factory)

target_include_directories(runtime_core_session_factory
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_session_factory
  PUBLIC
    runtime_core_session_basic
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    runtime_engine_engine_interface
    runtime_engine_engine_settings
    runtime_engine_io_types
    runtime_executor_audio_executor
    runtime_executor_llm_executor
    runtime_executor_vision_executor
    runtime_framework_threadpool
    runtime_util_litert_status_util
    LITERTLM_DEPS
)


# ==============================================================================
# 6. Engine Impl
# ==============================================================================
add_litertlm_library(runtime_core_engine_impl STATIC
  engine_impl.cc
  ../util/litert_util.cc
)
add_library(LiteRTLM::Runtime::Core::EngineImpl ALIAS runtime_core_engine_impl)

target_include_directories(runtime_core_engine_impl
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_engine_impl
  PUBLIC
    runtime_core_session_factory
    LiteRTLM::Runtime::Components::ModelResources::Interface
    runtime_engine_engine_interface
    runtime_engine_engine_settings
    runtime_engine_io_types
    runtime_executor_audio_executor
    runtime_executor_audio_executor_settings
    runtime_executor_audio_litert_compiled_model_executor
    runtime_executor_executor_settings_base
    runtime_executor_litert_compiled_model_executor_utils
    runtime_executor_llm_executor
    runtime_executor_llm_executor_settings
    runtime_executor_llm_litert_compiled_model_executor
    runtime_executor_magic_number_configs_helper
    runtime_executor_vision_executor
    runtime_executor_vision_litert_compiled_model_executor
    runtime_framework_threadpool
    runtime_executor_llm_litert_compiled_model_executor_factory
    runtime_util_file_format_util
    runtime_util_litert_status_util

    LITERTLM_DEPS

)

# ==============================================================================
# 7. Engine Impl (CPU Only)
# ==============================================================================
add_litertlm_library(runtime_core_engine_impl_cpu_only STATIC
  engine_impl.cc
  ../util/litert_util.cc
)
add_library(LiteRTLM::Runtime::Core::EngineImplCPU ALIAS runtime_core_engine_impl_cpu_only)

target_include_directories(runtime_core_engine_impl_cpu_only
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_core_engine_impl_cpu_only
  PUBLIC
    runtime_core_session_factory
    LiteRTLM::Runtime::Components::ModelResources::Interface
    runtime_engine_engine_interface
    runtime_engine_engine_settings
    runtime_engine_io_types
    runtime_executor_audio_executor
    runtime_executor_audio_executor_settings
    runtime_executor_audio_litert_compiled_model_executor
    runtime_executor_executor_settings_base
    runtime_executor_litert_compiled_model_executor_utils
    runtime_executor_llm_executor
    runtime_executor_llm_executor_settings
    runtime_executor_llm_litert_compiled_model_executor
    runtime_executor_magic_number_configs_helper
    runtime_executor_vision_executor
    runtime_executor_vision_litert_compiled_model_executor
    runtime_framework_threadpool
    runtime_executor_llm_litert_compiled_model_executor_factory
    runtime_util_file_format_util
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# 8. Facade
# ==============================================================================
add_library(runtime_core_libs INTERFACE)
add_library(LiteRTLM::Runtime::Core ALIAS runtime_core_libs)

target_link_libraries(runtime_core_libs INTERFACE
  LiteRTLM::Runtime::Core::EngineImpl
  LiteRTLM::Runtime::Core::EngineImplCPU
  LiteRTLM::Runtime::Core::Pipeline
  LiteRTLM::Runtime::Core::SessionBasic
  LiteRTLM::Runtime::Core::SessionFactory
  LiteRTLM::Runtime::Core::SessionUtils
  LiteRTLM::Runtime::Core::Tasks
)