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

add_subdirectory(resource_management)

# ==============================================================================
# Thread Options Interface
# ==============================================================================
add_litertlm_library(runtime_framework_thread_options INTERFACE)
add_library(LiteRTLM::Runtime::Framework::ThreadOptions ALIAS runtime_framework_thread_options)

target_include_directories(runtime_framework_thread_options
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Execution Queue
# ==============================================================================
add_litertlm_library(runtime_framework_execution_queue STATIC
  execution_queue.cc
)
add_library(LiteRTLM::Runtime::Framework::ExecutionQueue ALIAS runtime_framework_execution_queue)

target_include_directories(runtime_framework_execution_queue
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_execution_queue
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

# ==============================================================================
# Threadpool
# ==============================================================================

set(THREADPOOL_SRCS
  threadpool.cc
  worker_thread.cc
)

if(WIN32 OR LITERTLM_USE_STD_THREAD)
  list(APPEND THREADPOOL_SRCS worker_thread_std_thread.cc)
else()
  list(APPEND THREADPOOL_SRCS worker_thread_pthread.cc)
endif()

add_litertlm_library(runtime_framework_threadpool STATIC
  ${THREADPOOL_SRCS}
)
add_library(LiteRTLM::Runtime::Framework::Threadpool ALIAS runtime_framework_threadpool)

target_include_directories(runtime_framework_threadpool
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_framework_threadpool
  PUBLIC
    LiteRTLM::Runtime::Framework::ThreadOptions
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_framework_libs INTERFACE)
add_library(LiteRTLM::Runtime::Framework ALIAS runtime_framework_libs)

target_link_libraries(runtime_framework_libs INTERFACE
  LiteRTLM::Runtime::Framework::ThreadOptions
  LiteRTLM::Runtime::Framework::ExecutionQueue
  LiteRTLM::Runtime::Framework::Threadpool
  LiteRTLM::Runtime::Framework::ResourceManager
)