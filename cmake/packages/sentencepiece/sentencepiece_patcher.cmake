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
include("${LITERTLM_SENTENCEPIECE_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")

message(STATUS "[LiteRTLM] Patching SentencePiece source at: ${LITERTLM_SENTENCEPIECE_SRC_DIR}")

set(ROOT_LIST "${LITERTLM_SENTENCEPIECE_SRC_DIR}/CMakeLists.txt")
set(MATCH_STR "if \\(SPM_ABSL_PROVIDER STREQUAL \"module\"\\)[^@]*if\\(SPM_ENABLE_BENCHMARK\\)")

patch_file_content("${ROOT_LIST}"
    ${MATCH_STR}
    "if\(SPM_ENABLE_BENCHMARK\)"
    TRUE)

file(GLOB_RECURSE ALL_CMAKELISTS 
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/../*.cmake" 
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/../**/CMakeLists.txt")

foreach(C_FILE ${ALL_CMAKELISTS})
    if("${C_FILE}" STREQUAL "${ROOT_LIST}")
        continue()
    endif()
    patch_file_content("${C_FILE}"
        "absl::[a-zA-Z0-9_]+" 
        "LiteRTLM::absl::shim" 
        TRUE)
endforeach()

file(REMOVE_RECURSE "${LITERTLM_SENTENCEPIECE_SRC_DIR}/third_party/abseil-cpp")
file(REMOVE_RECURSE "${LITERTLM_SENTENCEPIECE_SRC_DIR}/third_party/absl")
file(REMOVE_RECURSE "${LITERTLM_SENTENCEPIECE_SRC_DIR}/third_party/protobuf")
file(REMOVE_RECURSE "${LITERTLM_SENTENCEPIECE_SRC_DIR}/third_party/protobuf-lite")

message(STATUS "[LiteRTLM] Performing header canonicalization in ${LITERTLM_SENTENCEPIECE_SRC_DIR}...")

file(GLOB_RECURSE SP_SOURCES
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/src/*.cc"
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/src/*.h"
)

foreach(FILE_PATH ${SP_SOURCES})
    file(READ "${FILE_PATH}" CONTENT)

    if(CONTENT MATCHES "SharedBitGen")
        string(REPLACE "absl::SharedBitGen" "absl::BitGen" CONTENT "${CONTENT}")
        if(NOT CONTENT MATCHES "#include \"absl/random/random.h\"")
            string(REPLACE "#include \"absl/strings/string_view.h\""
                           "#include \"absl/strings/string_view.h\"\n#include \"absl/random/random.h\""
                           CONTENT "${CONTENT}")
        endif()
        file(WRITE "${FILE_PATH}" "${CONTENT}")
        message(STATUS "  - Patched: ${FILE_PATH}")
    endif()
endforeach()

file(GLOB_RECURSE ALL_FILES
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/*.h"
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/*.cc"
    "${LITERTLM_SENTENCEPIECE_SRC_DIR}/*.cpp"
)

foreach(FILE_PATH ${ALL_FILES})
    file(READ "${FILE_PATH}" FILE_CONTENT)
    set(MODIFIED FALSE)

    if(FILE_CONTENT MATCHES "third_party/absl/")
        string(REPLACE "third_party/absl/" "absl/" FILE_CONTENT "${FILE_CONTENT}")
        set(MODIFIED TRUE)
    endif()

    if(FILE_CONTENT MATCHES "third_party/protobuf-lite/")
        string(REPLACE "third_party/protobuf-lite/" "google/protobuf/" FILE_CONTENT "${FILE_CONTENT}")
        set(MODIFIED TRUE)
    endif()

    if(MODIFIED)
        file(WRITE "${FILE_PATH}" "${FILE_CONTENT}")
    endif()
endforeach()

file(READ ${ROOT_LIST} ROOT_CONTENT)

string(REPLACE "project(sentencepiece VERSION \${SPM_VERSION} LANGUAGES C CXX)"
    "project(sentencepiece VERSION \${SPM_VERSION} LANGUAGES C CXX)\ninclude(${LITERTLM_SENTENCEPIECE_ROOT_SHIM_PATH})"
    ROOT_CONTENT "${ROOT_CONTENT}")

string(REPLACE
    "option(SPM_USE_BUILTIN_PROTOBUF \"Use builtin protobuf\" ON)"
    "option(SPM_USE_BUILTIN_PROTOBUF \"Use builtin protobuf\" OFF)"
    ROOT_CONTENT "${ROOT_CONTENT}")

string(REPLACE "set(CMAKE_CXX_STANDARD 17)" "set(CMAKE_CXX_STANDARD 20)" ROOT_CONTENT "${ROOT_CONTENT}")
file(WRITE "${LITERTLM_SENTENCEPIECE_SRC_DIR}/CMakeLists.txt" ${ROOT_CONTENT})

file(READ "${LITERTLM_SENTENCEPIECE_SRC_DIR}/src/CMakeLists.txt" SRC_CONTENT)
message(STATUS "[LiteRTLM] Redirecting SentencePiece internal Protobuf paths...")
string(REPLACE "\${CMAKE_CURRENT_SOURCE_DIR}/../third_party/protobuf-lite"
               "\${PROTO_SRC_DIR}"
               SRC_CONTENT "${SRC_CONTENT}")
string(REPLACE "\${CMAKE_CURRENT_SOURCE_DIR}/../third_party/absl/flags/flag.cc"
               "\${ABSL_SRC_DIR}/flags/internal/flag.cc"
               SRC_CONTENT "${SRC_CONTENT}")
string(REPLACE
    "include_directories(\${CMAKE_CURRENT_SOURCE_DIR}/../third_party)"
    "include_directories(\${CMAKE_CURRENT_SOURCE_DIR}/../third_party)\ninclude_directories(${LITERTLM_ABSL_INLUDE_DIR})\ninclude_directories(${LITERTLM_PROTO_INCLUDE_DIR})"
    SRC_CONTENT "${SRC_CONTENT}")
string(REPLACE "if (SPM_USE_BUILTIN_PROTOBUF)" "if (FALSE) # Forced by LiteRTLM" SRC_CONTENT "${SRC_CONTENT}")
string(REPLACE "if (SPM_USE_EXTERNAL_ABSL)" "if (TRUE) # Forced by LiteRTLM" SRC_CONTENT "${SRC_CONTENT}")
string(REPLACE "\${ABSL_STRINGS_SRCS}" "" SRC_CONTENT "${SRC_CONTENT}")
string(REPLACE "\${ABSL_FLAGS_SRCS}" "" SRC_CONTENT "${SRC_CONTENT}")

set(SRC_CONTENT ${SRC_CONTENT})
set(SRC_SHIM_INCLUDE "include(${LITERTLM_SENTENCEPIECE_SRC_SHIM_PATH})")
file(WRITE "${LITERTLM_SENTENCEPIECE_SRC_DIR}/src/CMakeLists.txt" ${SRC_SHIM_INCLUDE}${SRC_CONTENT})
