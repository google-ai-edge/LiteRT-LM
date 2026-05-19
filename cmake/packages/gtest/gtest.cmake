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


# [TODO] Update to follow the same pattern as the other packages.
include(ExternalProject)

set(GTEST_EXT_PREFIX ${EXTERNAL_PROJECT_BINARY_DIR}/googletest)
set(GTEST_INSTALL_PREFIX ${GTEST_EXT_PREFIX}/install)
set(GTEST_INCLUDE_DIR ${GTEST_INSTALL_PREFIX}/include)
set(GTEST_CONFIG_CMAKE_FILE "${GTEST_INSTALL_PREFIX}/lib/cmake/GTest/GTestConfig.cmake")

setup_external_install_structure("${GTEST_INSTALL_PREFIX}")

if(NOT EXISTS "${GTEST_CONFIG_CMAKE_FILE}")
  message(STATUS "GoogleTest not found. Configuring external build...")

  ExternalProject_Add(
    gtest_external
    DEPENDS
      absl_external
    GIT_REPOSITORY
      https://github.com/google/googletest
    GIT_TAG
      v1.17.0
    PREFIX
      ${GTEST_EXT_PREFIX}
    PATCH_COMMAND
      git checkout -- . && git clean -df
    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DCMAKE_PREFIX_PATH=${ABSL_INSTALL_PREFIX}
      -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POLICY_DEFAULT_CMP0169=OLD
      -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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

else()
    message(STATUS "GoogleTest already installed at: ${GTEST_INSTALL_PREFIX}")
    if(NOT TARGET gtest_external)
        add_custom_target(gtest_external)
    endif()
endif()

import_static_lib(imp_gmock                      "${GTEST_LIB_DIR}/libgmock.a")
import_static_lib(imp_gmock_main                 "${GTEST_LIB_DIR}/libgmock_main.a")
import_static_lib(imp_gtest                      "${GTEST_LIB_DIR}/libgtest.a")
import_static_lib(imp_gtest_main                 "${GTEST_LIB_DIR}/libgtest_main.a")

add_library(gtest_libs INTERFACE)
target_link_libraries(gtest_libs INTERFACE
    imp_gmock
    imp_gmock_main
    imp_gtest
    imp_gtest_main
)