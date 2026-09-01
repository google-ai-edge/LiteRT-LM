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
# Audio Executor Interface
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor INTERFACE)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorInterface
ALIAS runtime_executor_audio_executor)

target_include_directories(runtime_executor_audio_executor
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Audio Executor Base Interface
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor_base INTERFACE)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorBaseInterface
  ALIAS runtime_executor_audio_executor_base)

target_include_directories(runtime_executor_audio_executor_base
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_executor_base
  INTERFACE
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Audio Executor Settings
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor_settings STATIC
  audio_executor_settings.cc
)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorSettings
  ALIAS runtime_executor_audio_executor_settings)

target_include_directories(runtime_executor_audio_executor_settings
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_executor_settings
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
    LITERTLM_DEPS
)

# ==============================================================================
# Audio Executor Utils
# ==============================================================================
add_litertlm_library(runtime_executor_audio_executor_utils STATIC
  audio_executor_utils.cc
)
add_library(LiteRTLM::Runtime::Executor::AudioExecutorUtils
  ALIAS runtime_executor_audio_executor_utils)

target_include_directories(runtime_executor_audio_executor_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERT_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_executor_audio_executor_utils
  PUBLIC
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_executor_audio_libs INTERFACE)
add_library(LiteRTLM::Runtime::Executor::Audio ALIAS runtime_executor_audio_libs)

target_link_libraries(runtime_executor_audio_libs INTERFACE
  LiteRTLM::Runtime::Executor::AudioExecutorInterface
  LiteRTLM::Runtime::Executor::AudioExecutorBaseInterface
  LiteRTLM::Runtime::Executor::AudioExecutorSettings
  LiteRTLM::Runtime::Executor::AudioExecutorUtils
)
