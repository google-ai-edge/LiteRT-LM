# Copyright 2026 Google LC.
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
# Movable Mutex Lock
# ==============================================================================
add_litertlm_library(runtime_framework_resource_manager_utils_movable_mutex_lock INTERFACE)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::Utils::MovableMutexLock
  ALIAS runtime_framework_resource_manager_utils_movable_mutex_lock)

target_include_directories(runtime_framework_resource_manager_utils_movable_mutex_lock
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Resource Manager Utils
# ==============================================================================
add_litertlm_library(runtime_framework_resource_manager_utils_manager_utils STATIC
  resource_manager_utils.cc
)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::Utils::ResourceManagerUtils
  ALIAS runtime_framework_resource_manager_utils_manager_utils)

target_include_directories(runtime_framework_resource_manager_utils_manager_utils
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_resource_manager_utils_manager_utils
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_framework_resource_manager_utils_libs INTERFACE)
add_library(LiteRTLM::Runtime::Framework::ResourceManager::Utils
  ALIAS runtime_framework_resource_manager_utils_libs)

target_link_libraries(runtime_framework_resource_manager_utils_libs INTERFACE
  LiteRTLM::Runtime::Framework::ResourceManager::Utils::ResourceManagerUtils
  LiteRTLM::Runtime::Framework::ResourceManager::Utils::MovableMutexLock
)