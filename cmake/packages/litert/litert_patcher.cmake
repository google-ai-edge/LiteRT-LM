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

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_LITERT_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_FLATBUFFERS_CONFIG_PATH}")
include("${LITERTLM_TFLITE_CONFIG_PATH}")

set(ROOT_LIST "${LITERTLM_LITERT_SRC_DIR}/CMakeLists.txt")

if(EXISTS "${LITERTLM_LITERT_SRC_DIR}/../tflite")
    file(REMOVE_RECURSE "${LITERTLM_LITERT_SRC_DIR}/../tflite")
endif()

if(EXISTS "${LITERTLM_LITERT_SRC_DIR}/../tensor")
    file(REMOVE_RECURSE "${LITERTLM_LITERT_SRC_DIR}/../tensor")
endif()

if(EXISTS "${LITERTLM_TFLITE_SRC_DIR}")
    file(COPY "${LITERTLM_TFLITE_SRC_DIR}/" 
     DESTINATION "${LITERTLM_LITERT_SRC_DIR}/../tflite")
endif()

if(EXISTS "${LITERTLM_TENSORFLOW_SOURCE_DIR}")
file(COPY "${LITERTLM_TENSORFLOW_SOURCE_DIR}/" 
     DESTINATION "${LITERTLM_LITERT_SRC_DIR}/../tensor")
endif()

file(COPY_FILE "${LITERTLM_LITERT_PACKAGE_DIR}/shims/CMakeLists-shim.txt"
     "${ROOT_LIST}")
file(REMOVE "${LITERTLM_LITERT_SRC_DIR}/../support/preprocessor/CMakeLists.txt")

patch_file_content("${ROOT_LIST}" 
    "# Add TFLite as a subdirectory"
    "# Add TFLite as a subdirectory\nif(FALSE)"
    FALSE
)

patch_file_content("${ROOT_LIST}"
    "add_subdirectory(compiler_plugin)"
    "add_subdirectory(compiler)"
    FALSE
)

patch_file_content("${ROOT_LIST}"
    "# Add TFLite as a subdirectory[^@]+# Set default TensorFlow source directory if not found" 
    "\n" 
    TRUE
)

patch_file_content("${LITERTLM_LITERT_SRC_DIR}/../tensor/**/CMakeLists.txt"
    "add_subdirectory(examples//"
    "#add_subdirectory(examples//"
    FALSE
)

set(LITERTLM_BYPASS_PATHS
    "${LITERTLM_LITERT_SRC_DIR}/third_party/tensorflow/CMakeLists.txt"
    "${LITERTLM_LITERT_SRC_DIR}/tflite/CMakeLists.txt"
    "${LITERTLM_LITERT_SRC_DIR}/tflite/tools/cmake/CMakeLists.txt"
)
foreach(TARGET_PATH ${LITERTLM_BYPASS_PATHS})
    get_filename_component(TARGET_DIR "${TARGET_PATH}" DIRECTORY)
    if(NOT EXISTS "${TARGET_DIR}")
        file(MAKE_DIRECTORY "${TARGET_DIR}")
    endif()

    message(STATUS "[LiteRTLM] Bypassing conflicting build path: ${TARGET_PATH}")
    file(WRITE "${TARGET_PATH}" "# Path bypassed by LiteRT-LM to prevent dependency collisions.\n")
endforeach()

message(STATUS "[LiteRTLM] Patching LiteRT CMakeLists.txt files...")
file(GLOB_RECURSE ALL_CMAKELISTS 
    "${LITERTLM_LITERT_SRC_DIR}/../*.cmake" 
    "${LITERTLM_LITERT_SRC_DIR}/../**/CMakeLists.txt")


foreach(C_FILE ${ALL_CMAKELISTS})
    if("${C_FILE}" STREQUAL "${ROOT_LIST}")
        continue()
    endif()
    patch_file_content("${C_FILE}"
        "absl::[a-zA-Z0-9_]+" 
        "LiteRTLM::absl::shim" 
        TRUE)
    patch_file_content("${C_FILE}"
        "[^\" ]*/_deps/flatbuffers-build/libflatbuffers.a"
        "LiteRTLM::flatbuffers::flatbuffers"
        TRUE)
    patch_file_content("${C_FILE}"
        "flatbuffers-build/libflatbuffers.a"
        "LiteRTLM::flatbuffers::flatbuffers"
        FALSE)
    patch_file_content("${C_FILE}"
        "set(FLATC_EXECUTABLE \"\")"
        "\n"
        FALSE
    )
    patch_file_content("${C_FILE}"
        "TFLITE_FLATBUFFERS_LIB"
        "LiteRTLM::flatbuffers::flatbuffers"
        FALSE)
    patch_file_content("${C_FILE}" 
        "find_program\\(FLATC_EXECUTABLE[^\\)]+\\)"
        "# [LiteRTLM] Suppressed: Using Global Shim"
        TRUE)
    patch_file_content("${C_FILE}"
        "set\\(FLATC_EXECUTABLE \\$<TARGET_FILE:flatc>\\)"
        "set(FLATC_EXECUTABLE flatc)"
        TRUE)
    patch_file_content("${C_FILE}"
        "FetchContent_Declare\\([^\\)]+\\)"
        "# [LiteRTLM] Suppressed: External fetch prohibited"
        TRUE)
    patch_file_content("${C_FILE}"
        "FetchContent_MakeAvailable\\([^\\)]+\\)"
        "# [LiteRTLM] Suppressed: Using Global Manifest"
        TRUE)
endforeach()

patch_file_content("${LITERTLM_LITERT_SRC_DIR}/runtime/compiled_model.cc"
    " return litert_cpu_buffer_requirements"
    "return litert::Expected<const LiteRtTensorBufferRequirementsT*>(litert_cpu_buffer_requirements)"
    FALSE
)
patch_file_content("${LITERTLM_LITERT_SRC_DIR}/c/litert_environment.cc"
    "env->SetGpuEnvironment\\(std::move\\(gpu_env\\)\\)\\);[ \n\r]*\\}"
    "env->SetGpuEnvironment(std::move(gpu_env)));\n  }\n#endif"
    TRUE
)
patch_file_content("${LITERTLM_LITERT_SRC_DIR}/cc/internal/litert_runtime_builtin.cc"
    ".litert_gpu_environment_create = LiteRtGpuEnvironmentCreate,"
    "#if !defined(LITERT_DISABLE_GPU)\n  .litert_gpu_environment_create = LiteRtGpuEnvironmentCreate,\n#else\n  .litert_gpu_environment_create = nullptr,\n#endif"
    FALSE
)


# file(READ "${LITERTLM_LITERT_PACKAGE_DIR}/shims/litert_cc_options_shim.cmake" litert_cc_options_shim_CONTENT)
# patch_file_content("${LITERTLM_LITERT_SRC_DIR}/cc/options/CMakeLists.txt"
#     "cmake_minimum_required\\(VERSION 3.20\\).*"
#     "${LITERTLM_litert_cc_options_shim_CONTENT}"
#     TRUE
# )

set(GPU_INJECTION_STR
    "\${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/gl_interop.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/convert.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/data_type.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/gpu_info.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/memory_management.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/memory_management/greedy_by_breadth_assignment.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/memory_management/internal.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/memory_management/types.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/operations.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/common/shape.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/cl_command_queue.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/cl_event.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/cl_memory.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/opencl_wrapper.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/util.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/api.cc
    \${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/cl_kernel.cc)"
)
patch_file_content("${LITERTLM_LITERT_SRC_DIR}/c/CMakeLists.txt"
    "\${LITERTLM_TFLITE_SOURCE_DIR}/delegates/gpu/cl/gl_interop.cc)"
    "${GPU_INJECTION_STR}"
    FALSE
)

# --- Decouple Vendor Dependencies
set(V_LIST "${LITERTLM_LITERT_SRC_DIR}/vendors/CMakeLists.txt")
if(EXISTS "${V_LIST}")
    file(READ "${V_LIST}" V_CONTENT)
    string(FIND "${V_CONTENT}" "if(VENDOR STREQUAL \"MediaTek\")" START_POS)
    if(NOT START_POS EQUAL -1)
        string(FIND "${V_CONTENT}" "add_custom_target(mediatek_schema_gen" ANCHOR_POS)
        if(NOT ANCHOR_POS EQUAL -1)
            message(STATUS "[LiteRTLM] Decoupling Vendor Dependencies...")
            string(SUBSTRING "${V_CONTENT}" ${ANCHOR_POS} -1 POST_ANCHOR)
            string(FIND "${POST_ANCHOR}" "endif()" ENDIF_REL_POS)
            math(EXPR END_POS "${ANCHOR_POS} + ${ENDIF_REL_POS} + 7")
            string(SUBSTRING "${V_CONTENT}" 0 ${START_POS} PRE_BLOCK)
            string(SUBSTRING "${V_CONTENT}" ${END_POS} -1 POST_BLOCK)
            set(INJECTION "\n# [LiteRTLM] MediaTek Logic Virtualized\ninclude(\"${LITERTLM_LITERT_VENDOR_SHIM_PATH}\")\n")
            file(WRITE "${V_LIST}" "${PRE_BLOCK}${INJECTION}${POST_BLOCK}")
        endif()
    endif()
endif()

# --- Generate build_config.h
message(STATUS "[LiteRTLM] Generating build_config.h...")
set(LITERTLM_LITERT_GEN_DIR "${LITERTLM_LITERT_SRC_DIR}/build_common")

if(NOT EXISTS "${LITERTLM_LITERT_GEN_DIR}")
    file(MAKE_DIRECTORY "${LITERTLM_LITERT_GEN_DIR}")
endif()

cmake_to_c_bool(LITERT_BUILD_CONFIG_DISABLE_GPU_VAL C_DISABLE_GPU)
cmake_to_c_bool(LITERT_BUILD_CONFIG_DISABLE_NPU_VAL C_DISABLE_NPU)

set(BUILD_CONFIG_CONTENT "/* Generated by LiteRTLM Patcher */
#ifndef LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_

#define LITERT_BUILD_CONFIG_DISABLE_GPU ${C_DISABLE_GPU}
#define LITERT_BUILD_CONFIG_DISABLE_NPU ${C_DISABLE_NPU}

#endif  /* LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_ */\n")

file(WRITE "${LITERTLM_LITERT_GEN_DIR}/build_config.h" "${BUILD_CONFIG_CONTENT}")

set(LAYOUT_HDR "${LITERTLM_TFLITE_SRC_DIR}/../litert/cc/litert_layout.h")

# Neutralize constexpr constructor to avoid redefinition errors.
if(EXISTS "${LAYOUT_HDR}")
    message(STATUS "[LiteRTLM] Neutralizing constexpr violation in litert_layout.h")

    file(READ "${LAYOUT_HDR}" CONTENT)
    string(REPLACE "constexpr LiteRTLayout" "inline LiteRTLayout"
        MODIFIED_CONTENT "${CONTENT}")

    if(NOT "${CONTENT}" STREQUAL "${MODIFIED_CONTENT}")
        file(WRITE "${LAYOUT_HDR}" "${MODIFIED_CONTENT}")
        message(STATUS "[LiteRTLM] Successfully converted constexpr constructor to inline.")
    else()
        message(WARNING "[LiteRTLM] Patch target not found in litert_layout.h. Check version compatibility.")
    endif()
endif()


set(_tensor_example_cmakelists
    "${LITERTLM_LITERT_SRC_DIR}/../tensor/examples/gemma3/CMakeLists.txt"
    "${LITERTLM_LITERT_SRC_DIR}/../tensor/examples/segmentation/CMakeLists.txt"
)
foreach(_tensor_cmakelist IN LISTS _tensor_example_cmakelists)
    set(_cmakelist_path "${_tensor_cmakelist}")
    if(EXISTS "${_cmakelist_path}")
        message(STATUS "[LiteRTLM] Neutralizing ${_tensor_cmakelist}")
        file(READ "${_cmakelist_path}" CONTENT)
        file(WRITE "${_cmakelist_path}" "return()\n${CONTENT}")
    endif()
endforeach()

message(STATUS "[LiteRTLM] Patching Phase Complete.")
