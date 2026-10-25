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

# Builds Oniguruma (https://github.com/kkos/oniguruma) as a static library with
# its upstream CMakeLists.txt. Oniguruma is a plain C library without further
# dependencies; it backs tokenizers::SysRegex in
# support/tokenizer/tokenizers/utils/regex.h (the HuggingFace tokenizers port).
set(LITERTLM_ONIGURUMA_CONFIG_PATH "${LITERTLM_ONIGURUMA_PACKAGE_DIR}/oniguruma_config.cmake" CACHE PATH "")
include("${LITERTLM_ONIGURUMA_CONFIG_PATH}")

setup_external_install_structure("${LITERTLM_ONIGURUMA_INSTALL_PREFIX}")
set(LITERTLM_ONIGURUMA_TAG "v6.9.10" CACHE STRING "Oniguruma git tag")

include(ExternalProject)
if(NOT EXISTS "${LITERTLM_ONIGURUMA_CONFIG_CMAKE_FILE}")
  message(STATUS "Oniguruma not found. Configuring external build...")
  ExternalProject_Add(
    oniguruma_external
    GIT_REPOSITORY
      https://github.com/kkos/oniguruma.git
    GIT_TAG
      ${LITERTLM_ONIGURUMA_TAG}
    PREFIX
      ${LITERTLM_ONIGURUMA_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_ONIGURUMA_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    CONFIGURE_COMMAND
      ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
      ${CMAKE_COMMAND} -S ${LITERTLM_ONIGURUMA_SRC_DIR} -B ${LITERTLM_ONIGURUMA_BUILD_DIR}
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      "-DCMAKE_INSTALL_PREFIX=${LITERTLM_ONIGURUMA_INSTALL_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
      "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      # Static library; the (binary compatible) POSIX regex API, the docs and
      # the sample programs are not needed by tokenizers::SysRegex.
      "-DBUILD_SHARED_LIBS=OFF"
      "-DBUILD_TEST=OFF"
      "-DENABLE_POSIX_API=OFF"
      "-DENABLE_BINARY_COMPATIBLE_POSIX_API=OFF"
      "-DINSTALL_DOCUMENTATION=OFF"
      "-DINSTALL_EXAMPLES=OFF"
  )
  if(LITERTLM_GENERATE_TARGET_MAP)
    ExternalProject_Add_Step(oniguruma_external generate_target_map
      COMMAND ${CMAKE_COMMAND}
          "-DDEP_NAME=oniguruma"
          "-DSEARCH_DIR=${LITERTLM_ONIGURUMA_LIB_DIR}"
          "-DSEARCH_STR=LITERTLM_ONIGURUMA_LIB_DIR"
          "-DOUTPUT_FILE=${LITERTLM_ONIGURUMA_TARGET_MAP_PATH}"
          -P ${LITERTLM_SCRIPTS_DIR}/generate_target_map.cmake
      DEPENDEES install
      BYPRODUCTS ${LITERTLM_ONIGURUMA_TARGET_MAP_PATH}
    )
  endif()
else()
  message(STATUS "Oniguruma already installed at: ${LITERTLM_ONIGURUMA_INSTALL_PREFIX}")
  if(NOT TARGET oniguruma_external)
    add_custom_target(oniguruma_external)
  endif()
endif()

include(${LITERTLM_ONIGURUMA_AGGREGATE_PATH})
generate_oniguruma_aggregate()
