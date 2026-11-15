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
# Accelerator Memory Monitor
# ==============================================================================
# The Android dma-buf (NPU) accelerator memory monitor the CLIs use today. The
# full memory measurement library lives in LiteRT
# (litert/core/util/memory_profiling, litert_memory_profiling); this is a
# cut-down copy kept until the CLIs move onto that library's MemoryMonitor.
add_litertlm_library(runtime_util_memory_profiling_accelerator_memory_monitor STATIC
  accelerator_memory_monitor.cc
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
  LiteRTLM::Runtime::Util::MemoryProfiling::AcceleratorMemoryMonitor
)
