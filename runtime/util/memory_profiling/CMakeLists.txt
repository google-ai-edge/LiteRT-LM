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
# accelerator_memory_<os>.cc (.mm on Apple platforms) implements the platform
# contract for the target OS. Linux and Android share the /proc/self/fdinfo
# reader (fdinfo.cc), the dma-buf reader built on it
# (accelerator_memory_dmabuf.cc), the /proc/self/{maps,status,smaps_rollup}
# readers (proc_mappings.cc) and the Android combination rule
# (accelerator_memory_android_sources.cc). Keep this in sync with the select()
# in BUILD.
set(ACCELERATOR_MEMORY_MONITOR_SRCS accelerator_memory_monitor.cc)
if(ANDROID)
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS
    accelerator_memory_android.cc
    accelerator_memory_android_sources.cc
    accelerator_memory_dmabuf.cc
    fdinfo.cc
    proc_mappings.cc
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS
    accelerator_memory_android_sources.cc
    accelerator_memory_dmabuf.cc
    accelerator_memory_linux.cc
    accelerator_memory_nvml.cc
    fdinfo.cc
    proc_mappings.cc
  )
elseif(WIN32)
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS accelerator_memory_windows.cc)
elseif(APPLE)
  # Objective-C++: it reads MTLDevice. The rest of the build is C / C++.
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  list(APPEND ACCELERATOR_MEMORY_MONITOR_SRCS accelerator_memory_apple.mm)
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
if(WIN32)
  # dxgi: IDXGIAdapter3::QueryVideoMemoryInfo for per-process video memory
  # usage. gdi32: D3DKMTQueryStatistics for the part of it that is charged to
  # the process's private memory footprint.
  target_link_libraries(runtime_util_memory_profiling_accelerator_memory_monitor
    PUBLIC dxgi gdi32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # dlopen() of libnvidia-ml.so at run time.
  target_link_libraries(runtime_util_memory_profiling_accelerator_memory_monitor
    PUBLIC ${CMAKE_DL_LIBS})
elseif(APPLE)
  # Metal: MTLDevice.currentAllocatedSize. Foundation: NSArray / NSString.
  target_link_libraries(runtime_util_memory_profiling_accelerator_memory_monitor
    PUBLIC "-framework Metal" "-framework Foundation")
endif()

# ==============================================================================
# Memory Snapshot
# ==============================================================================
add_litertlm_library(runtime_util_memory_profiling_memory_snapshot STATIC
  memory_snapshot.cc
)
add_library(LiteRTLM::Runtime::Util::MemoryProfiling::MemorySnapshot
  ALIAS runtime_util_memory_profiling_memory_snapshot)

target_include_directories(runtime_util_memory_profiling_memory_snapshot
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_memory_profiling_memory_snapshot
  PUBLIC
    LiteRTLM::Runtime::Util::MemoryProfiling::HostMemory
    LiteRTLM::Runtime::Util::MemoryProfiling::AcceleratorMemoryMonitor
    LITERTLM_DEPS
)

# ==============================================================================
# Memory Monitor
# ==============================================================================
add_litertlm_library(runtime_util_memory_profiling_memory_monitor STATIC
  memory_monitor.cc
)
add_library(LiteRTLM::Runtime::Util::MemoryProfiling::MemoryMonitor
  ALIAS runtime_util_memory_profiling_memory_monitor)

target_include_directories(runtime_util_memory_profiling_memory_monitor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_util_memory_profiling_memory_monitor
  PUBLIC
    LiteRTLM::Runtime::Util::MemoryProfiling::AcceleratorMemoryMonitor
    LiteRTLM::Runtime::Util::MemoryProfiling::MemorySnapshot
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
  LiteRTLM::Runtime::Util::MemoryProfiling::MemorySnapshot
  LiteRTLM::Runtime::Util::MemoryProfiling::MemoryMonitor
)
