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

include("${LITERTLM_GENERATORS_DIR}/generate_protobuf.cmake")
include("${LITERTLM_GENERATORS_DIR}/generate_cxxbridge.cmake")

function(generate_src_files OUTPUT_CLEAN_PATHS)
    set(RAW_FILES ${ARGN})
    set(CLEANED_PATHS_OUT "")

    foreach(RAW_FILE IN ITEMS ${RAW_FILES})
        get_filename_component(FILE_NAME ${RAW_FILE} NAME)
        get_filename_component(FILE_DIR ${RAW_FILE} DIRECTORY)

        file(RELATIVE_PATH REL_PATH "${LITERTLM_PROJECT_ROOT}" "${FILE_DIR}")

        set(GEN_DIR "${LITERTLM_GENERATED_SRC_DIR}/${REL_PATH}")
        file(MAKE_DIRECTORY "${GEN_DIR}")

        set(CLEAN_FILE "${GEN_DIR}/${FILE_NAME}")

        if(NOT EXISTS "${RAW_FILE}")
            message(FATAL_ERROR "[LiteRTLM] Source file not found: ${RAW_FILE}")
        endif()
        file(READ "${RAW_FILE}" FILE_CONTENT)

        string(REPLACE "odml/litert_lm/" "" FILE_CONTENT "${FILE_CONTENT}")
        string(REPLACE "odml/litert/" "" FILE_CONTENT "${FILE_CONTENT}")

        file(WRITE "${CLEAN_FILE}" "${FILE_CONTENT}")

        list(APPEND CLEANED_PATHS_OUT "${CLEAN_FILE}")
    endforeach()

    set(${OUTPUT_CLEAN_PATHS} "${CLEANED_PATHS_OUT}" PARENT_SCOPE)
endfunction()

add_litertlm_library(litertlm_protobuf STATIC)
add_library(LiteRTLM::Generated::Protobuf ALIAS litertlm_protobuf)

add_dependencies(litertlm_protobuf protobuf_external)

target_include_directories(litertlm_protobuf
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LITERTLM_PROJECT_ROOT}
    ${LITERTLM_PROTOBUF_SRC_DIR}
    ${LITERTLM_PROTOBUF_INCLUDE_DIR}
    ${LITERTLM_ABSL_INCLUDE_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(litertlm_protobuf
  PUBLIC
    protobuf::libprotobuf
    LiteRTLM::absl::absl
    LITERTLM_DEPS
)

if(NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${LITERTLM_PROTOC_EXECUTABLE}"
    )
endif()

generate_protobuf(litertlm_protobuf ${LITERTLM_PROJECT_ROOT})

set(GEN_C_DIR "${LITERTLM_GENERATED_SRC_DIR}/c")
set(GEN_RUNTIME_DIR "${LITERTLM_GENERATED_SRC_DIR}/runtime")
set(GEN_SUPPORT_DIR "${LITERTLM_GENERATED_SRC_DIR}/support")
set(GEN_SCHEMA_DIR "${LITERTLM_GENERATED_SRC_DIR}/schema")
set(GEN_JS_DIR "${LITERTLM_GENERATED_SRC_DIR}/js")

set(ALL_SOURCE_FILES "")
set(ALL_HEADER_FILES "")
set(ALL_SCHEMA_FILES "")
set(ALL_RUST_FILES "")

file(GLOB_RECURSE C_SRC_FILES "${LITERTLM_PROJECT_ROOT}/c/*.cc")
file(GLOB_RECURSE C_HDR_FILES "${LITERTLM_PROJECT_ROOT}/c/*.h")

file(GLOB_RECURSE RUNTIME_SRC_FILES "${LITERTLM_PROJECT_ROOT}/runtime/*.cc")
file(GLOB_RECURSE RUNTIME_HDR_FILES "${LITERTLM_PROJECT_ROOT}/runtime/*.h")

file(GLOB_RECURSE SUPPORT_SRC_FILES "${LITERTLM_PROJECT_ROOT}/support/*.cc")
file(GLOB_RECURSE SUPPORT_HDR_FILES "${LITERTLM_PROJECT_ROOT}/support/*.h")

file(GLOB_RECURSE SCHEMA_SRC_FILES "${LITERTLM_PROJECT_ROOT}/schema/*.cc")
file(GLOB_RECURSE SCHEMA_HDR_FILES "${LITERTLM_PROJECT_ROOT}/schema/*.h")
file(GLOB_RECURSE SCHEMA_FBS_FILES "${LITERTLM_PROJECT_ROOT}/schema/*.fbs")

file(GLOB_RECURSE JS_SRC_FILES "${LITERTLM_PROJECT_ROOT}/js/*.cc")
file(GLOB_RECURSE JS_HDR_FILES "${LITERTLM_PROJECT_ROOT}/js/*.h")

file(GLOB_RECURSE RUST_SRC_FILES "${LITERTLM_PROJECT_ROOT}/src/*.rs")
file(GLOB_RECURSE RUST_RUNTIME_SRC_FILES "${LITERTLM_PROJECT_ROOT}/runtime/*.rs")
file(GLOB_RECURSE RUST_TOML_FILES "${LITERTLM_PROJECT_ROOT}/cmake/rust/*.toml")

list(APPEND ALL_SOURCE_FILES 
    ${C_SRC_FILES}
    ${RUNTIME_SRC_FILES}
    ${SUPPORT_SRC_FILES}
    ${SCHEMA_SRC_FILES}
    ${JS_SRC_FILES})

list(APPEND ALL_HEADER_FILES
    ${C_HDR_FILES}
    ${RUNTIME_HDR_FILES}
    ${SUPPORT_HDR_FILES}
    ${SCHEMA_HDR_FILES}
    ${JS_HDR_FILES})
list(APPEND ALL_SCHEMA_FILES ${SCHEMA_FBS_FILES})

list(APPEND ALL_RUST_FILES 
    ${RUST_SRC_FILES} 
    ${RUST_RUNTIME_SRC_FILES} 
    ${RUST_TOML_FILES}
)

set(ALL_GENERATED_OUTPUTS 
    ${C_SRC_FILES}
    ${RUNTIME_SRC_FILES}
    ${SUPPORT_SRC_FILES}
    ${SCHEMA_SRC_FILES}
    ${JS_SRC_FILES}
    ${RUST_SRC_FILES}
)
