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

add_compile_options(
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-w>
    $<$<CXX_COMPILER_ID:MSVC>:/W0>
)

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_MODULES_DIR}/generators/generate_protobuf.cmake")

if(WIN32)
    # Match Bazel's XNNPACK build by compiling only production microkernels
    # (xnnpack-microkernels-prod); non-prod microkernels are only used by
    # XNNPACK's own tests and benchmarks.
    set(XNNPACK_BUILD_ALL_MICROKERNELS OFF CACHE BOOL "Build only production XNNPACK microkernels on Windows" FORCE)
endif()

# Ported from LiteRT's litert/CMakeLists.txt (LiteRT-LM builds TFLite itself).
# MSVC's cl.exe targeting ARM64 ships none of the ACLE headers or intrinsics
# these microkernel families need: <arm_fp16.h>, <arm_bf16.h> and <arm_sve.h>
# do not exist, and arm64_neon.h declares no bf16 intrinsics. XNNPACK selects
# them from the target ISA alone, so a cl.exe build fails to compile them.
# clang-cl has all of them, so only cl.exe is gated.
# ARM64 I8MM (without BF16) uses <arm_neon.h> intrinsics that cl.exe supports
# and is kept enabled via patch_xnnpack_windows_arm64.cmake.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _litertlm_tflite_processor)
    string(TOLOWER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" _litertlm_tflite_compiler_arch)
    if(_litertlm_tflite_processor MATCHES "^(arm64|aarch64)$"
       OR _litertlm_tflite_compiler_arch STREQUAL "arm64")
        message(STATUS
            "[LiteRTLM] cl.exe on ARM64 -- disabling XNNPACK FP16/BF16/SVE2/SME "
            "microkernels (no ACLE support); keeping DotProd and I8MM enabled.")
        foreach(_litertlm_xnn_off
                XNNPACK_ENABLE_ARM_FP16_SCALAR
                XNNPACK_ENABLE_ARM_FP16_VECTOR
                XNNPACK_ENABLE_ARM_BF16
                XNNPACK_ENABLE_ARM_SVE2
                XNNPACK_ENABLE_ARM_SME
                XNNPACK_ENABLE_ARM_SME2
                XNNPACK_ENABLE_ARM_SME_ACLE)
            set(${_litertlm_xnn_off} OFF CACHE BOOL "Unsupported by cl.exe on ARM64" FORCE)
        endforeach()
    endif()
endif()

set(_tflite_shims_dir "${LITERTLM_TFLITE_PACKAGE_DIR}/shims")
include("${_tflite_shims_dir}/build_tree_shim.cmake")

# --- Abseil ---
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_ABSL_AGGREGATE_PATH}")
generate_absl_aggregate()

include("${LITERTLM_GTEST_CONFIG_PATH}")
include("${LITERTLM_TFLITE_CONFIG_PATH}")

# --- Protobuf ---
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_PROTOBUF_AGGREGATE_PATH}")
generate_protobuf_aggregate()
if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf ALIAS LiteRTLM::protobuf::libprotobuf)
endif()
if(NOT TARGET protobuf::protobuf)
    add_library(protobuf::protobuf ALIAS LiteRTLM::protobuf::libprotobuf)
endif()

set(Protobuf_INCLUDE_DIR "${LITERTLM_PROTOBUF_INCLUDE_DIRS}" CACHE INTERNAL "")
set(Protobuf_LIBRARIES LiteRTLM::protobuf::libprotobuf CACHE INTERNAL "")
set(Protobuf_PROTOC_EXECUTABLE "${LITERTLM_HOST_PROTOC}" CACHE INTERNAL "")
set(Protobuf_FOUND TRUE CACHE INTERNAL "")
set(PROTOBUF_FOUND TRUE CACHE INTERNAL "")

if(NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${LITERTLM_HOST_PROTOC}"
    )
endif()

# --- Flatbuffers ---
include("${LITERTLM_FLATBUFFERS_CONFIG_PATH}")
include("${LITERTLM_FLATBUFFERS_AGGREGATE_PATH}")
generate_flatbuffers_aggregate()

set(FIXED_FLATC    "${LITERTLM_FLATC_EXECUTABLE}" CACHE INTERNAL "Forced" FORCE)
set(FLATC_TARGET                 "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(FLATC_BIN                    "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(FLATBUFFERS_FLATC_EXECUTABLE "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(flatbuffers_FLATC_EXECUTABLE "${FIXED_FLATC}" CACHE INTERNAL "Forced" FORCE)
set(TFLITE_HOST_TOOLS_DIR        "${FIXED_FLATC}" CACHE PATH     "Forced" FORCE)
set(FLATC_PATHS                  "${FIXED_FLATC}" CACHE STRING   "Forced" FORCE)
set(flatbuffers_FOUND            TRUE             CACHE INTERNAL "Forced" FORCE)
set(FlatBuffers_FOUND            TRUE             CACHE INTERNAL "Forced" FORCE)

if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers INTERFACE IMPORTED GLOBAL)
    set_target_properties(flatbuffers::flatbuffers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
    )
endif()

if(NOT TARGET flatc)
    add_executable(flatc IMPORTED GLOBAL)
    set_target_properties(flatc PROPERTIES 
        IMPORTED_LOCATION "${LITERTLM_FLATC_EXECUTABLE}"
    )
endif()

include_directories(
    "${LITERTLM_ABSL_INCLUDE_DIR}"
    "${LITERTLM_PROTOBUF_INCLUDE_DIR}"
    "${LITERTLM_PROTOBUF_INSTALL_DIR}/include"
    "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
    "${LITERTLM_TENSORFLOW_SRC_DIR}"
    "${LITERTLM_TFLITE_SRC_DIR}"
)

message(STATUS "[LiteRTLM] Injecting missing CMakeLists into profiling/...")

file(GLOB PROFILING_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/profiling/*.cc")
list(FILTER PROFILING_SRCS EXCLUDE REGEX "_test\\.cc$")
if(WIN32)
    # Android-only (includes <dlfcn.h>); platform_profiler.cc only references
    # it under __ANDROID__.
    list(FILTER PROFILING_SRCS EXCLUDE REGEX "atrace_profiler\\.cc$")
endif()

set(STATS_CALC_SRC 
    "${LITERTLM_TENSORFLOW_SRC_DIR}/third_party/xla/xla/tsl/util/stats_calculator.cc")

if(EXISTS "${STATS_CALC_SRC}")
    message(STATUS "[LiteRTLM] Found stats_calculator at: ${STATS_CALC_SRC}")
    list(APPEND PROFILING_SRCS "${STATS_CALC_SRC}")
else()
    set(STATS_CALC_FALLBACK 
        "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/core/util/stats_calculator.cc")
    if(EXISTS "${STATS_CALC_FALLBACK}")
         list(APPEND PROFILING_SRCS "${STATS_CALC_FALLBACK}")
    else()
         message(FATAL_ERROR "[LiteRT-LM] CRITICAL: Could not find stats_calculator.cc in XLA or Core paths.\nChecked:\n  ${STATS_CALC_SRC}\n  ${STATS_CALC_FALLBACK}")
    endif()
endif()

set(LITERTLM_PROTO_FILES
    "${LITERTLM_TFLITE_SRC_DIR}/profiling/proto/profiling_info.proto"
    "${LITERTLM_TFLITE_SRC_DIR}/profiling/proto/model_runtime_info.proto"
)

add_library(tflite_profiling STATIC ${PROFILING_SRCS})
if(CMAKE_GENERATOR MATCHES "Ninja"
   AND EXISTS "${LITERTLM_TFLITE_SRC_DIR}/profiling/proto/CMakeLists.txt")
    # Older TFLite revisions' profiling/proto/CMakeLists.txt already adds custom
    # commands for profiling_info.proto and model_runtime_info.proto at the same
    # output paths; Ninja rejects duplicate build rules for the same file.
    target_link_libraries(tflite_profiling PUBLIC
        profiling_info_proto
        model_runtime_info_proto
    )
else()
    generate_protobuf(tflite_profiling ${LITERTLM_TENSORFLOW_SRC_DIR})
endif()

target_link_libraries(tflite_profiling PRIVATE
    LiteRTLM::absl::absl
    LiteRTLM::protobuf::libprotobuf
)

target_include_directories(tflite_profiling PUBLIC
    "${CMAKE_BINARY_DIR}"
    "${LITERTLM_TENSORFLOW_SRC_DIR}"
    "${LITERTLM_ABSL_INCLUDE_DIR}"
    "${LITERTLM_PROTOBUF_INCLUDE_DIR}"
)

install(TARGETS tflite_profiling
    ARCHIVE DESTINATION lib
)