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

add_subdirectory(context_handler)
add_subdirectory(processed_context)
add_subdirectory(utils)

# ==============================================================================
# Resource Manager
# ==============================================================================
add_litertlm_library(runtime_framework_resource_manager_manager STATIC
  resource_manager.cc
)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::Manager ALIAS runtime_framework_resource_manager_manager)

target_include_directories(runtime_framework_resource_manager_manager
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_resource_manager_manager
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

# ==============================================================================
# Serial Execution Manager
# ==============================================================================
add_litertlm_library(runtime_framework_resource_manager_serial_execution_manager STATIC
  serial_execution_manager.cc
)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::SerialExecutionManager ALIAS runtime_framework_resource_manager_serial_execution_manager)

target_include_directories(runtime_framework_resource_manager_serial_execution_manager
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_resource_manager_serial_execution_manager
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

# ==============================================================================
# Threaded Execution Manager
# ==============================================================================
add_litertlm_library(runtime_framework_resource_manager_threaded_execution_manager STATIC
  threaded_execution_manager.cc
)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::ThreadedExecutionManager ALIAS runtime_framework_resource_manager_threaded_execution_manager)

target_include_directories(runtime_framework_resource_manager_threaded_execution_manager
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_resource_manager_threaded_execution_manager
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_framework_resource_libs INTERFACE)
add_library(LiteRTLM::Runtime::Framework::ResourceManager ALIAS runtime_framework_resource_libs)

target_link_libraries(runtime_framework_resource_libs INTERFACE
  LiteRTLM::Runtime::Framework::ResourceManager::Utils
  LiteRTLM::Runtime::Framework::ResourceManager::ContextHandler
  LiteRTLM::Runtime::Framework::ResourceManager::ProcessedContext
  LiteRTLM::Runtime::Framework::ResourceManager::Manager
  LiteRTLM::Runtime::Framework::ResourceManager::SerialExecutionManager
  LiteRTLM::Runtime::Framework::ResourceManager::ThreadedExecutionManager
)