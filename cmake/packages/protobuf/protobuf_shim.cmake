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
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_ABSL_AGGREGATE_PATH}")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
add_definitions(-DABSL_LTS_GROUP_EXPORT)

generate_absl_aggregate()

set(protobuf_ABSL_PROVIDER "package" CACHE INTERNAL "" FORCE)
set(protobuf_ABSL_USED_TARGETS "LiteRTLM::absl::absl" CACHE INTERNAL "" FORCE)
set(protobuf_ABSL_USED_TEST_TARGETS "LiteRTLM::absl::absl" CACHE INTERNAL "" FORCE)

include_directories(SYSTEM ${LITERTLM_ABSL_INCLUDE_DIR})
link_libraries(LiteRTLM::absl::shim)

# [TODO] Refactor into macro for DRY principle.
# --- Toolchain-Specific Linker Flags ---
set(_LITERTLM_LINK_MULTIDEF "")
set(_LITERTLM_LINK_GROUP_START "")
set(_LITERTLM_LINK_GROUP_END "")
set(_LITERTLM_SYSLIBS "")

if(LITERTLM_TOOLCHAIN_ARGS)
    set(CMAKE_CXX_FLAGS_RELEASE "${LITERTLM_PROTOBUF_CXX_FLAGS_RELEASE}" CACHE STRING "Override NDK O3" FORCE)
    set(CMAKE_C_FLAGS_RELEASE "${LITERTLM_PROTOBUF_CXX_FLAGS_RELEASE}" CACHE STRING "Override NDK O3" FORCE)
    set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard" FORCE)
endif()

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

set(CMAKE_CXX_STANDARD_LIBRARIES
    ${CMAKE_CXX_STANDARD_LIBRARIES} 
    ${_LITERTLM_LINK_MULTIDEF}
    ${_LITERTLM_LINK_GROUP_START}
    ${_ABSL_LINK_FLAGS}
    ${_LITERTLM_SYSLIBS}
    ${_LITERTLM_LINK_GROUP_END}
    CACHE STRING "" FORCE
)
string(REPLACE ";" " " CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES}")
