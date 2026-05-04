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


include(ExternalProject)

set(FLATBUFFERS_EXT_PREFIX ${EXTERNAL_PROJECT_BINARY_DIR}/flatbuffers CACHE INTERNAL "")
set(FLATBUFFERS_INSTALL_PREFIX ${FLATBUFFERS_EXT_PREFIX}/install CACHE INTERNAL "")
set(FLATBUFFERS_INCLUDE_DIR ${FLATBUFFERS_INSTALL_PREFIX}/include CACHE INTERNAL "")
set(FLATBUFFERS_SRC_DIR ${FLATBUFFERS_EXT_PREFIX}/src CACHE INTERNAL "")
set(FLATBUFFERS_BIN_DIR ${FLATBUFFERS_INSTALL_PREFIX}/bin CACHE INTERNAL "")
set(FLATBUFFERS_LIB_DIR ${FLATBUFFERS_INSTALL_PREFIX}/lib CACHE INTERNAL "")
set(FLATBUFFERS_DIR ${FLATBUFFERS_LIB_DIR}/cmake/flatbuffers CACHE INTERNAL "")
set(FLATBUFFERS_CMAKE_CONFIG_FILE ${FLATBUFFERS_LIB_DIR}/cmake/flatbuffers/flatbuffers-config.cmake CACHE INTERNAL "")
set(FLATBUFFERS_FLATC_EXECUTABLE "${FLATBUFFERS_BIN_DIR}/flatc" CACHE INTERNAL "")
set(FLATC_EXECUTABLE "${FLATBUFFERS_BIN_DIR}/flatc" CACHE INTERNAL "")

if(LITERTLM_HOST_FLATC AND EXISTS "${LITERTLM_HOST_FLATC}")
    message(STATUS "[LiteRTLM] FlatBuffers: Using host flatc at ${LITERTLM_HOST_FLATC_BIN_DIR}")
    set(FLATBUFFERS_BIN_DIR "${LITERTLM_HOST_BIN_DIR}" CACHE INTERNAL "Host Flatbuffers binary path")
    set(FLATBUFFERS_FLATC_EXECUTABLE "${LITERTLM_HOST_FLATC}" CACHE INTERNAL "Host flatc")
    set(FLATC_EXECUTABLE "${LITERTLM_HOST_FLATC}" CACHE INTERNAL "Host flatc")

endif()

setup_external_install_structure("${FLATBUFFERS_INSTALL_PREFIX}")

if(NOT EXISTS "${FLATBUFFERS_CMAKE_CONFIG_FILE}")
  message(STATUS "Flatbuffers not found. Configuring external build...")

  ExternalProject_Add(
    flatbuffers_external
    DEPENDS
      absl_external
      gtest_external
    GIT_REPOSITORY
      https://github.com/google/flatbuffers.git
    GIT_TAG
      v25.9.23
    PREFIX
      ${FLATBUFFERS_EXT_PREFIX}
    PATCH_COMMAND
      git checkout -- . && git clean -df
    CMAKE_ARGS
        ${LITERTLM_TOOLCHAIN_FILE}
        ${LITERTLM_TOOLCHAIN_ARGS}
        -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_PREFIX}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DFLATBUFFERS_BUILD_TESTS=OFF
        -DFLATBUFFERS_BUILD_GRPCTEST=OFF
        -DFLATBUFFERS_INSTALL=ON
        -DFLATBUFFERS_BUILD_FLATC=ON
        -DFLATBUFFERS_BUILD_FLATHASH=OFF
        -DFLATBUFFERS_CPP_STD=20
  )
  ExternalProject_Add_Step(flatbuffers_external compile_schemas
    COMMAND ${CMAKE_COMMAND}
        -D FLATC_BIN=${FLATC_EXECUTABLE}
        -D SCHEMA_DIR=${GENERATED_SRC_DIR}/schema
        -P ${LITERTLM_SCRIPTS_DIR}/compile_flatbuffers.cmake

    DEPENDEES install

    COMMENT "[LiteRTLM] Batch compiling all Flatbuffer schemas..."
    ALWAYS 1 # Force check on every build in case schemas changed
)
else()
    message(STATUS "[LiteRTLM] Flatbuffers already installed at: ${FLATBUFFERS_INSTALL_PREFIX}")
    if(NOT TARGET flatbuffers_external)
        add_custom_target(flatbuffers_external)
    endif()
endif()

include(${FLATBUFFERS_PACKAGE_DIR}/flatbuffers_aggregate.cmake)
generate_flatbuffers_aggregate()
generate_flatc_aggregate()