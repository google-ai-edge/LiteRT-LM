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

message(STATUS "[LiteRTLM] Patching tokenizers source at: ${LITERTLM_TOKENIZERS_SRC_DIR}")
include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_TOKENIZERS_CONFIG_PATH}")
include("${LITERTLM_SENTENCEPIECE_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_FLATBUFFERS_CONFIG_PATH}")

set(ROOT_LIST "${LITERTLM_TOKENIZERS_SRC_DIR}/CMakeLists.txt")

patch_file_content("${ROOT_LIST}"
    "project(tokenizers_cpp C CXX)"
    "project(tokenizers_cpp C CXX)\n
    include(${LITERTLM_TOKENIZERS_PACKAGE_DIR}/tokenizers_config.cmake)\n
    include(${LITERTLM_TOKENIZERS_PACKAGE_DIR}/tokenizers_shim.cmake)\n"
    FALSE
)

patch_file_content("${ROOT_LIST}"
    "add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)"
    "#add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)"
    FALSE
)

patch_file_content("${ROOT_LIST}"
    "target_include_directories(tokenizers_cpp PRIVATE sentencepiece/src)"
    "target_include_directories(tokenizers_cpp PRIVATE ${LITERTLM_SENTENCEPIECE_INCLUDE_PATHS})"
    FALSE
)

patch_file_content("${ROOT_LIST}"
    "set(CMAKE_CXX_STANDARD 17)"
    "set(CMAKE_CXX_STANDARD 20)"
    FALSE
)

# Windows ARM64: the Rust crate bundled with tokenizers-cpp must be built for
# the same target as the rest of the build, not for the Rust toolchain's
# default one (x64 when cross compiling from an x64 host). Seed
# TOKENIZERS_CPP_CARGO_TARGET in tokenizers-cpp/CMakeLists.txt with our target.
if(CMAKE_HOST_WIN32 AND "${LITERTLM_RUST_TARGET}" MATCHES "^aarch64")
    file(READ "${ROOT_LIST}" _tokenizers_root_content)
    set(_tokenizers_cargo_target_init "set(TOKENIZERS_CPP_CARGO_TARGET \"\")")
    string(FIND "${_tokenizers_root_content}" "${_tokenizers_cargo_target_init}" _tokenizers_cargo_target_pos)
    if(_tokenizers_cargo_target_pos EQUAL -1)
        message(FATAL_ERROR
            "[LiteRTLM] tokenizers-cpp: could not find '${_tokenizers_cargo_target_init}' in "
            "${ROOT_LIST}; cannot pin Rust target to ${LITERTLM_RUST_TARGET}.")
    else()
        string(REPLACE "${_tokenizers_cargo_target_init}"
            "set(TOKENIZERS_CPP_CARGO_TARGET \"${LITERTLM_RUST_TARGET}\")"
            _tokenizers_root_content "${_tokenizers_root_content}")
        file(WRITE "${ROOT_LIST}" "${_tokenizers_root_content}")
        message(STATUS "[LiteRTLM] tokenizers-cpp: Rust target pinned to ${LITERTLM_RUST_TARGET}")
    endif()
endif()

file(GLOB_RECURSE ALL_SOURCE_FILES
    "${LITERTLM_TOKENIZERS_SRC_DIR}/../**/*.h" 
    "${LITERTLM_TOKENIZERS_SRC_DIR}/../**/*.cc")

foreach(_src_file ${ALL_SOURCE_FILES})
  message(VERBOSE "[LiteRTLM] Patching ${_src_file}...")
  patch_file_content("${_src_file}"
      "third_party/absl"
      "absl"
      FALSE)
endforeach()

file(GLOB_RECURSE ALL_CMAKELISTS 
    "${LITERTLM_TOKENIZERS_SRC_DIR}/../*.cmake" 
    "${LITERTLM_TOKENIZERS_SRC_DIR}/../**/CMakeLists.txt")

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
    patch_file_content("${C_FILE}"
        "sentencepiece::[a-zA-Z0-9_]+" 
        "LiteRTLM::sentencepiece::shim" 
        TRUE)
endforeach()

patch_file_content("${ROOT_LIST}"
    "find_program\\(FLATC_BIN flatc HINTS \\\${FLATC_PATHS}\\)"
    "set(FLATC_BIN \"${LITERTLM_FLATC_EXECUTABLE}\" CACHE FILEPATH \"Forced by LiteRT-LM\")"
    TRUE
)
