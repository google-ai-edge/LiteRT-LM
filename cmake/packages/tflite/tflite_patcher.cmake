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

message(STATUS "[LiteRTLM] Patching TFLite source at: ${LITERTLM_TENSORFLOW_SRC_DIR}")
include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_TFLITE_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_FLATBUFFERS_CONFIG_PATH}")

set(ROOT_LIST "${LITERTLM_TFLITE_SRC_DIR}/CMakeLists.txt")

file(COPY_FILE "${LITERTLM_TFLITE_PACKAGE_DIR}/shims/CMakeLists-shim.txt"
     "${ROOT_LIST}")


patch_file_content("${ROOT_LIST}"
    "project(tensorflow-lite C CXX)"
    "project(tensorflow-lite C CXX)\n
    include(${LITERTLM_TFLITE_PACKAGE_DIR}/tflite_config.cmake)\n
    include(${LITERTLM_TFLITE_PACKAGE_DIR}/tflite_shim.cmake)\n"
    FALSE
)

file(GLOB_RECURSE ALL_CMAKELISTS 
    "${LITERTLM_TFLITE_SRC_DIR}/../*.cmake" 
    "${LITERTLM_TFLITE_SRC_DIR}/../**/CMakeLists.txt")

foreach(C_FILE ${ALL_CMAKELISTS})
    patch_file_content("${C_FILE}"
        "absl::[a-zA-Z0-9_]+" 
        "LiteRTLM::absl::shim" 
        TRUE)
    patch_file_content("${C_FILE}"
        "protobuf::[a-zA-Z0-9_]+" 
        "LiteRTLM::protobuf::shim" 
        TRUE)
    patch_file_content("${C_FILE}"
        "flatbuffers::[a-zA-Z0-9_]+" 
        "LiteRTLM::flatbuffers::shim" 
        TRUE)
endforeach()

patch_file_content("${ROOT_LIST}"
    "find_program\\(FLATC_BIN flatc HINTS \\\${FLATC_PATHS}\\)"
    "set(FLATC_BIN \"${LITERTLM_FLATC_EXECUTABLE}\" CACHE FILEPATH \"Forced by LiteRT-LM\")"
    TRUE
)

set(CONFIG_GEN_H "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/lite/acceleration/configuration/configuration_generated.h")
if(EXISTS "${CONFIG_GEN_H}")
    file(READ "${CONFIG_GEN_H}" CONTENT)
    string(REGEX REPLACE "FLATBUFFERS_VERSION_MAJOR == [0-9]+" "FLATBUFFERS_VERSION_MAJOR >= 25" CONTENT "${CONTENT}")
    file(WRITE "${CONFIG_GEN_H}" "${CONTENT}")
endif()

if(EXISTS "${LITERTLM_PROJECT_ROOT}/cmake/patches/converter.zip")
    message(STATUS "[LITERTLM] Extracting converter.zip...")
    file(ARCHIVE_EXTRACT INPUT "${LITERTLM_PROJECT_ROOT}/cmake/patches/converter.zip" DESTINATION "${LITERTLM_TFLITE_SRC_DIR}")
endif()

set(SCHEMA_GEN_H "${LITERTLM_TENSORFLOW_SRC_DIR}/tensorflow/compiler/mlir/lite/schema/schema_generated.h")
if(EXISTS "${SCHEMA_GEN_H}")
    file(READ "${SCHEMA_GEN_H}" CONTENT)
    string(REPLACE "FLATBUFFERS_VERSION_MAJOR == 24"
           "FLATBUFFERS_VERSION_MAJOR >= 24" CONTENT "${CONTENT}")
    string(REPLACE "FLATBUFFERS_VERSION_MINOR == 3"
           "FLATBUFFERS_VERSION_MINOR >= 0" CONTENT "${CONTENT}")
    string(REPLACE "FLATBUFFERS_VERSION_REVISION == 25"
           "FLATBUFFERS_VERSION_REVISION >= 0" CONTENT "${CONTENT}")
    file(WRITE "${SCHEMA_GEN_H}" "${CONTENT}")
endif()

set(_downloader_modules
    "tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake"
    "tensorflow/lite/tools/cmake/modules/protobuf.cmake"
    "tensorflow/lite/tools/cmake/modules/flatbuffers.cmake"
)
foreach(_mod IN LISTS _downloader_modules)
    set(_mod_path "${_mod}")
    if(EXISTS "${_mod_path}")
        message(STATUS "[LiteRTLM] Neutralizing ${_mod}")
        file(READ "${_mod_path}" CONTENT)
        file(WRITE "${_mod_path}" "return()\n${CONTENT}")
    endif()
endforeach()

# --- XNNPACK ---
message(STATUS "[LiteRTLM] Patching XNNPACK...")
if(EXISTS "${LITERTLM_XNNPACK_MOD_FILE}")
    message(STATUS "[LiteRTLM] Applying XNNPACK shim...")
    file(APPEND "${LITERTLM_XNNPACK_MOD_FILE}"
        "\n# [LiteRTLM] Automated Fix\ninclude(${LITERTLM_TFLITE_PACKAGE_DIR}/shims/xnnpack_shim.cmake)\n")
endif()

if(EXISTS "${LITERTLM_XNNPACK_DELEGATE_CMAKELISTS}")
    message(STATUS "[LiteRTLM] Patching XNNPACK delegate custom command (Native)...")
    file(READ "${LITERTLM_XNNPACK_DELEGATE_CMAKELISTS}" CONTENT)
    string(REPLACE "\"\${FLATBUFFERS_FLATC_EXECUTABLE}\""
        "\"${LITERTLM_FLATC_EXECUTABLE}\""
        CONTENT "${CONTENT}"
    )
    string(REPLACE "\"\${FLATC_TARGET}\""
    "\"${LITERTLM_FLATC_EXECUTABLE}\""
    CONTENT "${CONTENT}")
    file(WRITE "${LITERTLM_XNNPACK_DELEGATE_CMAKELISTS}" "${CONTENT}")
endif()

message(STATUS "[LiteRTLM] Performing manual XNNPACK schema generation...")
execute_process(
    COMMAND "${LITERTLM_FLATC_EXECUTABLE}" -c
            -o "${LITERTLM_TFLITE_SRC_DIR}/"
            --gen-mutable --gen-object-api
            "${LITERTLM_TFLITE_SRC_DIR}/delegates/xnnpack/weight_cache_schema.fbs"
    RESULT_VARIABLE manual_gen_res
)

if(NOT manual_gen_res EQUAL 0)
    message(FATAL_ERROR "LITERTLM: Manual flatc generation failed! Path: ${LITERTLM_FLATC_EXECUTABLE}")
endif()

message(STATUS "[LiteRTLM] Neutralizing XNNPACK custom command...")
patch_delete_block("${LITERTLM_XNNPACK_CMAKELISTS}" "add_custom_command\\(" "\\)")

file(MAKE_DIRECTORY "${LITERTLM_TFLITE_SRC_DIR}/delegates/xnnpack")
file(COPY "${LITERTLM_TFLITE_SRC_DIR}/weight_cache_schema_generated.h"
     DESTINATION "${LITERTLM_TFLITE_BUILD_DIR}/tensorflow/lite/delegates/xnnpack")

set(_PROTO_RECORDS
    "tensorflow/lite/profiling/proto/CMakeLists.txt:profiling_info.proto"
    "tensorflow/lite/tools/benchmark/proto/CMakeLists.txt:benchmark_result.proto"
)

foreach(RECORD ${_PROTO_RECORDS})
    string(REPLACE ":" ";" FIELDS ${RECORD})
    list(GET FIELDS 0 PLIST)
    list(GET FIELDS 1 PFILE)
    set(TARGET_LIST "${LITERTLM_TENSORFLOW_SRC_DIR}/${PLIST}")

    if(EXISTS "${TARGET_LIST}")
        message(STATUS "[LiteRT-LM] Applying patch to ${PLIST} (Native)...")
        get_filename_component(PDIR "${PLIST}" DIRECTORY)

        file(READ "${TARGET_LIST}" CONTENT)

        string(REGEX REPLACE "--proto_path=[^ \t\r\n\"]*"
               "--proto_path=${LITERTLM_TENSORFLOW_SRC_DIR}" CONTENT "${CONTENT}")
        string(REGEX REPLACE " [^ \t\r\n\"]*${PFILE}"
               " ${LITERTLM_TENSORFLOW_SRC_DIR}/${PDIR}/${PFILE}" CONTENT "${CONTENT}")
        string(REPLACE "tflite/"
               "tensorflow/lite/" CONTENT "${CONTENT}")
        file(WRITE "${TARGET_LIST}" "${CONTENT}")
    endif()
endforeach()
