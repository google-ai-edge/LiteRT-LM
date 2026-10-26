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
# Host Memory
# ==============================================================================
# Exactly one host_memory_<os>.cc implements the platform contract for the
# target OS. Keep this in sync with the select() in BUILD.
if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(HOST_MEMORY_SRCS host_memory_linux.cc)
elseif(WIN32)
  set(HOST_MEMORY_SRCS host_memory_windows.cc)
elseif(APPLE)
  set(HOST_MEMORY_SRCS host_memory_apple.cc)
else()
  set(HOST_MEMORY_SRCS host_memory_unsupported.cc)
endif()

add_litertlm_library(runtime_util_memory_profiling_host_memory STATIC
  ${HOST_MEMORY_SRCS}
)
add_library(LiteRTLM::Runtime::Util::MemoryProfiling::HostMemory
  ALIAS runtime_util_memory_profiling_host_memory)

target_include_directories(runtime_util_memory_profiling_host_memory
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_memory_profiling_host_memory
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Accelerator Memory Monitor
# ==============================================================================
# accelerator_memory_monitor.cc is platform-neutral; exactly one
# accelerator_memory_<os>.cc implements the platform contract for the target
# OS. The dma-buf reader (accelerator_memory_dmabuf.cc) is Android's building
# block; it is built on Linux too so that its test can run without a device.
# Keep this in sync with the select() in BUILD.
set(ACCELERATOR_MEMORY_MONITOR_SRCS accelerator_memory_monitor.cc)
if(ANDROID)
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS
    accelerator_memory_android.cc
    accelerator_memory_dmabuf.cc
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS
    accelerator_memory_dmabuf.cc
    accelerator_memory_unsupported.cc
  )
else()
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS accelerator_memory_unsupported.cc)
endif()

add_litertlm_library(runtime_util_memory_profiling_accelerator_memory_monitor STATIC
  ${ACCELERATOR_MEMORY_MONITOR_SRCS}
)
add_library(LiteRTLM::Runtime::Util::MemoryProfiling::AcceleratorMemoryMonitor
  ALIAS runtime_util_memory_profiling_accelerator_memory_monitor)

target_include_directories(runtime_util_memory_profiling_accelerator_memory_monitor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_memory_profiling_accelerator_memory_monitor
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_util_memory_profiling_libs INTERFACE)
add_library(LiteRTLM::Runtime::Util::MemoryProfiling ALIAS runtime_util_memory_profiling_libs)

target_link_libraries(runtime_util_memory_profiling_libs INTERFACE
  LiteRTLM::Runtime::Util::MemoryProfiling::HostMemory
  LiteRTLM::Runtime::Util::MemoryProfiling::AcceleratorMemoryMonitor
)
