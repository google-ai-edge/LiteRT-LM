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

set(LITERTLM_RE2_CONFIG_PATH "${LITERTLM_RE2_PACKAGE_DIR}/re2_config.cmake" CACHE PATH "")
include("${LITERTLM_RE2_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")

setup_external_install_structure("${LITERTLM_RE2_INSTALL_PREFIX}")
set(LITERTLM_RE2_TAG "main" CACHE STRING "RE2 git tag")

include(ExternalProject)
if(NOT EXISTS "${LITERTLM_RE2_CONFIG_CMAKE_FILE}")
  message(STATUS "RE2 not found. Configuring external build...")
  ExternalProject_Add(
    re2_external
    DEPENDS
      absl_external
    GIT_REPOSITORY
      https://github.com/google/re2/
    GIT_TAG
      ${LITERTLM_RE2_TAG}
    PREFIX
      ${LITERTLM_RE2_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_RE2_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    PATCH_COMMAND
      ${CMAKE_COMMAND}
      "-DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}"
      "-DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}"
      "-DLITERTLM_CMAKE_PACKAGES_DIR=${LITERTLM_CMAKE_PACKAGES_DIR}"
      "-DLITERTLM_RE2_PACKAGE_DIR=${LITERTLM_RE2_PACKAGE_DIR}"
      "-DLITERTLM_RE2_CONFIG_PATH=${LITERTLM_RE2_CONFIG_PATH}"
      "-DLITERTLM_ABSL_PACKAGE_DIR=${LITERTLM_ABSL_PACKAGE_DIR}"
      "-DLITERTLM_ABSL_CONFIG_PATH=${LITERTLM_ABSL_CONFIG_PATH}"
      "-P ${LITERTLM_RE2_PACKAGE_DIR}/re2_patcher.cmake"
    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      "-DLITERTLM_ORCHESTRATION_PHASE=${LITERTLM_ORCHESTRATION_PHASE}"
      "-DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}"
      "-DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}"
      "-DLITERTLM_CMAKE_PACKAGES_DIR=${LITERTLM_CMAKE_PACKAGES_DIR}"
      "-DLITERTLM_RE2_PACKAGE_DIR=${LITERTLM_RE2_PACKAGE_DIR}"
      "-DLITERTLM_RE2_CONFIG_PATH=${LITERTLM_RE2_CONFIG_PATH}"
      "-DLITERTLM_ABSL_PACKAGE_DIR=${LITERTLM_ABSL_PACKAGE_DIR}"
      "-DLITERTLM_ABSL_CONFIG_PATH=${LITERTLM_ABSL_CONFIG_PATH}"
      "-Dabsl_DIR=${LITERTLM_ABSL_INSTALL_PREFIX}/lib/cmake/absl"
      "-DCMAKE_INSTALL_PREFIX=${LITERTLM_RE2_INSTALL_PREFIX}"
  )
else()
  message(STATUS "RE2 already installed at: ${LITERTLM_RE2_INSTALL_PREFIX}")
  if(NOT TARGET re2_external)
    add_custom_target(re2_external)
  endif()
endif()

include(${LITERTLM_RE2_PACKAGE_DIR}/re2_aggregate.cmake)
generate_re2_aggregate()
