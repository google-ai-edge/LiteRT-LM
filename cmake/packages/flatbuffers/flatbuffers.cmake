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
set(LITERTLM_FLATBUFFERS_CONFIG_PATH "${LITERTLM_FLATBUFFERS_PACKAGE_DIR}/flatbuffers_config.cmake" CACHE INTERNAL "")
include("${LITERTLM_FLATBUFFERS_CONFIG_PATH}")

set(LITERTLM_FLATBUFFERS_EXTERNAL_DONE ${LITERTLM_FLATBUFFERS_STAMP_DIR}/flatbuffers_external-done CACHE INTERNAL "")

setup_external_install_structure("${LITERTLM_FLATBUFFERS_INSTALL_PREFIX}")
set(LITERTLM_FLATBUFFERS_TAG "v25.9.23" CACHE STRING "Flatbuffers git tag")

include(ExternalProject)
if(NOT EXISTS "${LITERTLM_FLATBUFFERS_EXTERNAL_DONE}")
  message(STATUS "Flatbuffers not found. Configuring external build...")
  ExternalProject_Add(
    flatbuffers_external
    DEPENDS
      absl_external
      gtest_external
    GIT_REPOSITORY
      https://github.com/google/flatbuffers.git
    GIT_TAG
      ${LITERTLM_FLATBUFFERS_TAG}
    PREFIX
      ${LITERTLM_FLATBUFFERS_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_FLATBUFFERS_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DLITERTLM_ORCHESTRATION_PHASE=${LITERTLM_ORCHESTRATION_PHASE}
      "-DLITERTLM_BIN_EXT=${LITERTLM_BIN_EXT}"
      "-DLITERTLM_STATIC_LIB_EXT=${LITERTLM_STATIC_LIB_EXT}"
      "-DLITERTLM_DYN_LIB_EXT=${LITERTLM_DYN_LIB_EXT}"
      "-DLITERTLM_LIB_PREFIX=${LITERTLM_LIB_PREFIX}"
      "-DLITERTLM_TARGET_MAP_DIR=${LITERTLM_TARGET_MAP_DIR}"
      "-DCMAKE_INSTALL_PREFIX=${LITERTLM_FLATBUFFERS_INSTALL_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DFLATBUFFERS_BUILD_TESTS=OFF
      -DFLATBUFFERS_BUILD_GRPCTEST=OFF
      -DFLATBUFFERS_INSTALL=ON
      -DFLATBUFFERS_BUILD_FLATC=ON
      -DFLATBUFFERS_BUILD_FLATHASH=OFF
      -DFLATBUFFERS_CPP_STD=20
  )
  if(LITERTLM_GENERATE_TARGET_MAP)
    ExternalProject_Add_Step(flatbuffers_external generate_target_map
      COMMAND ${CMAKE_COMMAND}
          "-DDEP_NAME=flatbuffers"
          "-DSEARCH_DIR=${LITERTLM_FLATBUFFERS_LIB_DIR}"
          "-DSEARCH_STR=LITERTLM_FLATBUFFERS_LIB_DIR"
          "-DOUTPUT_FILE=${LITERTLM_FLATBUFFERS_TARGET_MAP_PATH}"
          -P ${LITERTLM_SCRIPTS_DIR}/generate_target_map.cmake
      DEPENDEES install
      BYPRODUCTS ${LITERTLM_FLATBUFFERS_TARGET_MAP_PATH}
    )
  endif()
else()
    message(STATUS "[LiteRTLM] Flatbuffers already installed at: ${LITERTLM_FLATBUFFERS_INSTALL_PREFIX}")
    if(NOT TARGET flatbuffers_external)
        add_custom_target(flatbuffers_external)
    endif()
endif()

include(${LITERTLM_FLATBUFFERS_PACKAGE_DIR}/flatbuffers_aggregate.cmake)
generate_flatbuffers_aggregate()
generate_flatc_aggregate()
