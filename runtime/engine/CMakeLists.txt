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
# 1. Engine Interface
# ==============================================================================
add_litertlm_library(runtime_engine_engine_interface INTERFACE)
add_library(LiteRTLM::Runtime::Engine::Interface ALIAS runtime_engine_engine_interface)

target_include_directories(runtime_engine_engine_interface
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_engine_interface
  INTERFACE
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# 2. Engine Settings
# ==============================================================================
add_litertlm_library(runtime_engine_engine_settings STATIC
  engine_settings.cc
  cpu_affinity_utils.cc
)
add_library(LiteRTLM::Runtime::Engine::Settings ALIAS runtime_engine_engine_settings)

target_include_directories(runtime_engine_engine_settings
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_engine_settings
  PUBLIC
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    runtime_executor_audio_executor_settings
    runtime_executor_executor_settings_base
    runtime_executor_llm_executor_settings
    runtime_executor_vision_executor_settings
    runtime_util_litert_status_util
    runtime_util_model_type_utils

    LITERTLM_DEPS
)

# ==============================================================================
# 3. IO Types
#    Bazel: cc_library(name = "io_types" ...)
# ==============================================================================
add_litertlm_library(runtime_engine_io_types STATIC
  io_types.cc
)
add_library(LiteRTLM::Runtime::Engine::IoTypes ALIAS runtime_engine_io_types)

target_include_directories(runtime_engine_io_types
  PRIVATE
    ${CMAKE_BINARY_DIR}
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}
)

target_link_libraries(runtime_engine_io_types
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    runtime_util_litert_status_util

    LITERTLM_DEPS
)

# ==============================================================================
# 4. Engine Lib (The Core Logic)
#    Bazel: cc_library(name = "litert_lm_lib" ...)
# ==============================================================================
add_litertlm_library(runtime_engine_litert_lm_lib STATIC
  litert_lm_lib.cc
)
add_library(LiteRTLM::Runtime::Engine::Lib ALIAS runtime_engine_litert_lm_lib)

target_include_directories(runtime_engine_litert_lm_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/json/include
)

target_link_libraries(runtime_engine_litert_lm_lib
  PUBLIC
    # Internal Modules (Bazel deps)
    LiteRTLM::Runtime::Conversation
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Core::EngineImpl
    LiteRTLM::Runtime::Engine::Interface
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes
    runtime_executor_executor_settings_base
    runtime_executor_llm_executor_settings
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# 5. Shared Flags
# ==============================================================================
add_litertlm_library(runtime_engine_shared_flags STATIC
  shared_flags.cc
)
add_library(LiteRTLM::Runtime::Engine::SharedFlags ALIAS runtime_engine_shared_flags)

target_include_directories(runtime_engine_shared_flags
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_engine_shared_flags
  PUBLIC
    LITERTLM_DEPS
)


if(_unverified_targets)
  # ==============================================================================
  # 6. Advanced Main Executable
  # ==============================================================================

  set(MEMORY_USAGE_MONITOR_SRC "${TFLITE_SRC_DIR}/profiling/memory_usage_monitor.cc")

  set_source_files_properties(${MEMORY_USAGE_MONITOR_SRC} PROPERTIES GENERATED TRUE)

  add_litertlm_executable(runtime_engine_litert_lm_advanced_main
    litert_lm_advanced_main.cc
    ${MEMORY_USAGE_MONITOR_SRC}
  )

  target_link_options(runtime_engine_litert_lm_advanced_main PRIVATE
    "LINKER:--export-dynamic-symbol=LiteRt*"
  )
  target_include_directories(runtime_engine_litert_lm_advanced_main PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
    ${TENSORFLOW_SOURCE_DIR}
  )

  target_link_libraries(runtime_engine_litert_lm_advanced_main
    PUBLIC
        LiteRTLM::Runtime::Engine::IoTypes
        LiteRTLM::Runtime::Engine::Lib
        LiteRTLM::Runtime::Engine::SharedFlags
        LiteRTLM::Schema::Core

        runtime_util_litert_status_util
        LITERTLM_DEPS
  )

  # ==============================================================================
  # 7. Main Executable (The Target We Care About)
  # ==============================================================================
  add_litertlm_executable(litert_lm_main
    litert_lm_main.cc
  )
  target_link_options(litert_lm_main PRIVATE ${LITERTLM_UNIFIED_LINK_SPEC})
  target_link_options(litert_lm_main PRIVATE
    "LINKER:--export-dynamic-symbol=LiteRt*"
  )

  target_include_directories(litert_lm_main PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/json/include
  )

  target_link_libraries(litert_lm_main
    PUBLIC
      LiteRTLM::Runtime::Engine::Interface
      LiteRTLM::Runtime::Engine::Settings
      LiteRTLM::Runtime::Engine::IoTypes
      LiteRTLM::Runtime::Conversation
      LiteRTLM::Runtime::Conversation::IoTypes
      LiteRTLM::Runtime::Core::EngineImpl
      runtime_executor_executor_settings_base
      runtime_executor_llm_executor_settings
      runtime_util_litert_status_util
      LITERTLM_DEPS
  )
  add_litertlm_executable(litert_lm_main
      litert_lm_main.cc
  )
  target_link_options(litert_lm_main PRIVATE
    "LINKER:--export-dynamic-symbol=LiteRt*"
  )
  target_include_directories(litert_lm_main PRIVATE
      ${LITERTLM_INCLUDE_PATHS}
      ${LITERT_INCLUDE_PATHS}
      ${THIRD_PARTY_DIR}/json/include
  )

  # Android Specifics
  if(ANDROID)
    target_link_libraries(litert_lm_main PRIVATE EGL GLESv3)
  endif()
endif()


# ==============================================================================
# 8. Folder Facade
# ==============================================================================
add_library(runtime_engine_libs INTERFACE)
add_library(LiteRTLM::Runtime::Engine ALIAS runtime_engine_libs)

target_link_libraries(runtime_engine_libs INTERFACE
  LiteRTLM::Runtime::Engine::Interface
  LiteRTLM::Runtime::Engine::Settings
  LiteRTLM::Runtime::Engine::IoTypes
  LiteRTLM::Runtime::Engine::Lib
  LiteRTLM::Runtime::Engine::SharedFlags
)