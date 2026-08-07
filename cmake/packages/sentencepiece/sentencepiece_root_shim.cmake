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

message(STATUS "[LiteRTLM] Redirecting SentencePiece dependencies...")

set(LITERTLM_SENTENCEPIECE_PACKAGE_DIR "${LITERTLM_SENTENCEPIECE_PACKAGE_DIR}" CACHE PATH "" FORCE)
set(LITERTLM_SENTENCEPIECE_CONFIG_PATH "${LITERTLM_SENTENCEPIECE_CONFIG_PATH}" CACHE PATH "" FORCE)
include("${LITERTLM_SENTENCEPIECE_CONFIG_PATH}")

set(LITERTLM_ABSL_PACKAGE_DIR "${LITERTLM_ABSL_PACKAGE_DIR}" CACHE PATH "" FORCE)
set(LITERTLM_ABSL_CONFIG_PATH "${LITERTLM_ABSL_CONFIG_PATH}" CACHE PATH "" FORCE)
include("${LITERTLM_ABSL_CONFIG_PATH}")

set(LITERTLM_PROTOBUF_PACKAGE_DIR "${LITERTLM_PROTOBUF_PACKAGE_DIR}" CACHE PATH "" FORCE)
set(LITERTLM_PROTOBUF_CONFIG_PATH "${LITERTLM_PROTOBUF_CONFIG_PATH}" CACHE PATH "" FORCE)
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")

set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "package" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
set(Protobuf_INCLUDE_DIRS ${LITERTLM_PROTOBUF_INCLUDE_DIR} CACHE PATH "" FORCE)
set(PROTOBUF_INCLUDE_DIR ${LITERTLM_PROTOBUF_INCLUDE_DIR} CACHE PATH "" FORCE)
set(ABSL_SRC_FILE_PATH "${LITERTLM_ABSL_SRC_DIR}/absl" CACHE PATH "" FORCE)
set(ABSL_INLUDE_FILE_PATH "${LITERTLM_ABSL_INCLUDE_DIR}" CACHE PATH "" FORCE)

set(PROTOC_EXECUTABLE "${LITERTLM_PROTOBUF_BIN_DIR}/protoc" CACHE PATH "" FORCE)
set(PROTOC_EXE "${LITERTLM_PROTOBUF_BIN_DIR}/protoc" CACHE PATH "" FORCE)
set(protobuf_generate_PROTOC_EXE "${LITERTLM_PROTOBUF_BIN_DIR}/protoc" CACHE PATH "" FORCE)

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_ABSL_PACKAGE_DIR}/absl_aggregate.cmake")
include("${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_aggregate.cmake")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

generate_absl_aggregate()
generate_protobuf_aggregate()

include_directories(${LITERTLM_ABSL_INCLUDE_DIR} ${LITERTLM_PROTOBUF_INCLUDE_DIR})

# [TODO] Refactor into macro for DRY principle.
# --- Toolchain-Specific Linker Flags ---
set(_LITERTLM_LINK_MULTIDEF "")
set(_LITERTLM_LINK_GROUP_START "")
set(_LITERTLM_LINK_GROUP_END "")
set(_LITERTLM_SYSLIBS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(APPLE)
        # AppleClang / Mach-O Linker
        set(_LITERTLM_LINK_MULTIDEF "-Wl,-multiply_defined,suppress")
        set(_LITERTLM_SYSLIBS "-lz -lpthread -ldl")
    elseif(ANDROID)
        # Android / Bionic (NO standalone rt or pthread)
        set(_LITERTLM_LINK_MULTIDEF "-Wl,--allow-multiple-definition")
        set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
        set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
        set(_LITERTLM_SYSLIBS "-lz -ldl -llog")
    else()
        # Linux / ELF Linker (GNU ld or LLD)
        set(_LITERTLM_LINK_MULTIDEF "-Wl,--allow-multiple-definition")
        set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
        set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
        set(_LITERTLM_SYSLIBS "-lz -lrt -lpthread -ldl")
    endif()
elseif(MSVC)
    # MSVC Linker
    set(_LITERTLM_LINK_MULTIDEF "/FORCE:MULTIPLE")
    set(_LITERTLM_SYSLIBS "") 
endif()

set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} ${_LITERTLM_LINK_MULTIDEF} ${_LITERTLM_LINK_GROUP_START} ${_PROTOBUF_PAYLOAD} ${_ABSL_PAYLOAD} ${_LITERTLM_SYSLIBS} ${_LITERTLM_LINK_GROUP_END}"
    CACHE STRING "" FORCE
)
