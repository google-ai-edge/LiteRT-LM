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
include("${LITERTLM_GTEST_PACKAGE_DIR}/gtest_config.cmake")

set(LITERTLM_GTEST_EXTERNAL_DONE ${LITERTLM_GTEST_STAMP_DIR}/gtest_external-done CACHE INTERNAL "")


setup_external_install_structure("${LITERTLM_GTEST_INSTALL_PREFIX}")
set(LITERTLM_GTEST_TAG "v1.17.0" CACHE STRING "GoogleTest git tag")

include(ExternalProject)
if(NOT EXISTS "${LITERTLM_GTEST_EXTERNAL_DONE}")
  message(STATUS "GoogleTest not found. Configuring external build...")

  ExternalProject_Add(
    gtest_external
    DEPENDS
      absl_external
    GIT_REPOSITORY
      https://github.com/google/googletest
    GIT_TAG
      ${LITERTLM_GTEST_TAG}
    PREFIX
      ${LITERTLM_GTEST_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_GTEST_TAG}
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
      -DCMAKE_PREFIX_PATH=${ABSL_INSTALL_PREFIX}
      -DCMAKE_INSTALL_PREFIX=${LITERTLM_GTEST_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POLICY_DEFAULT_CMP0169=OLD
      -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DLITERTLM_ABSL_CONFIG_PATH=${ABSL_CONFIG_PATH}
      -Dabsl_DIR=${absl_DIR}
      -DABSL_DIR=${ABSL_DIR}
      -Dabsl_ROOT=${absl_ROOT}
      -DABSL_ROOT=${ABSL_ROOT}
      -DABSL_INCLUDE_DIR=${ABSL_INCLUDE_DIR}
      -DABSL_INCLUDE_DIRS=${ABSL_INCLUDE_DIRS}
      -Dabsl_INCLUDE_DIR=${absl_INCLUDE_DIR}
      -Dabsl_INCLUDE_DIRS=${absl_INCLUDE_DIRS}
      -DABSL_LIBRARY_DIR=${ABSL_LIBRARY_DIR}
      -DABSL_LIB_DIR=${ABSL_LIB_DIR}
      -Dabsl_LIBRARY_DIR=${absl_LIBRARY_DIR}
  )
  if(LITERTLM_GENERATE_TARGET_MAP)
    ExternalProject_Add_Step(gtest_external generate_target_map
      COMMAND ${CMAKE_COMMAND}
          "-DDEP_NAME=gtest"
          "-DSEARCH_DIR=${LITERTLM_GTEST_LIB_DIR}"
          "-DSEARCH_STR=LITERTLM_GTEST_LIB_DIR"
          "-DOUTPUT_FILE=${LITERTLM_GTEST_TARGET_MAP_PATH}"
          -P ${LITERTLM_SCRIPTS_DIR}/generate_target_map.cmake
      DEPENDEES install
      BYPRODUCTS ${LITERTLM_GTEST_TARGET_MAP_PATH}
    )
  endif()
else()
    message(STATUS "GoogleTest already installed at: ${LITERTLM_GTEST_INSTALL_PREFIX}")
    if(NOT TARGET gtest_external)
        add_custom_target(gtest_external)
    endif()
endif()


#--- Legacy target map generation for gtest. This is not recommended and may be removed in future versions.
import_static_lib(imp_gmock "${LITERTLM_GTEST_LIB_DIR}/libgmock.a")
import_static_lib(imp_gmock_main "${LITERTLM_GTEST_LIB_DIR}/libgmock_main.a")
import_static_lib(imp_gtest "${LITERTLM_GTEST_LIB_DIR}/libgtest.a")
import_static_lib(imp_gtest_main "${LITERTLM_GTEST_LIB_DIR}/libgtest_main.a")

add_library(gtest_libs INTERFACE)
target_link_libraries(gtest_libs INTERFACE
    imp_gmock
    imp_gmock_main
    imp_gtest
    imp_gtest_main
)
