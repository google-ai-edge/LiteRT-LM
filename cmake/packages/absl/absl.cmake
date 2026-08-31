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
set(LITERTLM_ABSL_CONFIG_PATH "${LITERTLM_ABSL_PACKAGE_DIR}/absl_config.cmake" CACHE PATH "")
include("${LITERTLM_ABSL_CONFIG_PATH}")

set(LITERTLM_ABSL_EXTERNAL_DONE ${LITERTLM_ABSL_STAMP_DIR}/absl_external-done CACHE INTERNAL "")

setup_external_install_structure("${LITERTLM_ABSL_INSTALL_PREFIX}")

set(LITERTLM_ABSL_TAG "20260526.0" CACHE STRING "Abseil git tag")

include(ExternalProject)
if(NOT EXISTS "${LITERTLM_ABSL_EXTERNAL_DONE}")
  message(STATUS "Abseil not found. Configuring external build...")
  ExternalProject_Add(
    absl_external
    GIT_REPOSITORY
      https://github.com/abseil/abseil-cpp
    GIT_TAG
      ${LITERTLM_ABSL_TAG}
    PREFIX
      ${LITERTLM_ABSL_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_ABSL_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    PATCH_COMMAND
      ${CMAKE_COMMAND}
      -DLITERTLM_ORCHESTRATION_PHASE=${LITERTLM_ORCHESTRATION_PHASE}
      -DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}
      -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
      -DLITERTLM_CMAKE_PACKAGES_DIR=${LITERTLM_CMAKE_PACKAGES_DIR}
      -DLITERTLM_ABSL_PACKAGE_DIR=${LITERTLM_ABSL_PACKAGE_DIR}
      -DLITERTLM_ABSL_CONFIG_PATH=${LITERTLM_ABSL_CONFIG_PATH}
      -P "${LITERTLM_ABSL_PACKAGE_DIR}/absl_patcher.cmake"

    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DLITERTLM_ORCHESTRATION_PHASE=${LITERTLM_ORCHESTRATION_PHASE}
      -DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}
      -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
      -DLITERTLM_CMAKE_PACKAGES_DIR=${LITERTLM_CMAKE_PACKAGES_DIR}
      -DLITERTLM_ABSL_PACKAGE_DIR=${LITERTLM_ABSL_PACKAGE_DIR}
      -DLITERTLM_ABSL_CONFIG_PATH=${LITERTLM_ABSL_CONFIG_PATH}
      -DCMAKE_INSTALL_PREFIX=${LITERTLM_ABSL_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POLICY_DEFAULT_CMP0169=OLD
      -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DABSL_BUILD_TESTING=OFF
      -DABSL_USE_GOOGLETEST_HEAD=OFF
      -DABSL_ENABLE_INSTALL=ON
      -DABSL_PROPAGATE_CXX_STD=ON
      -DBUILD_SHARED_LIBS=OFF

  )
else()
  message(STATUS "Abseil already installed at: ${LITERTLM_ABSL_INSTALL_PREFIX}")
  if(NOT TARGET absl_external)
    add_custom_target(absl_external)
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${LITERTLM_ABSL_INSTALL_PREFIX}")
list(APPEND CMAKE_SYSTEM_PREFIX_PATH "${LITERTLM_ABSL_INSTALL_PREFIX}")

include(${LITERTLM_ABSL_AGGREGATE_PATH})
generate_absl_aggregate()