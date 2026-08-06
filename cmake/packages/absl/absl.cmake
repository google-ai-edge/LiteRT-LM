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

set(ABSL_EXT_PREFIX ${EXTERNAL_PROJECT_BINARY_DIR}/abseil-cpp CACHE INTERNAL "")
set(ABSL_INSTALL_PREFIX ${ABSL_EXT_PREFIX}/install CACHE INTERNAL "")
set(ABSL_INCLUDE_DIR ${ABSL_INSTALL_PREFIX}/include CACHE INTERNAL "")
set(ABSL_LIB_DIR ${ABSL_INSTALL_PREFIX}/lib CACHE INTERNAL "")
set(ABSL_SRC_DIR ${ABSL_EXT_PREFIX}/src/absl_external CACHE INTERNAL "")
set(ABSL_BUILD_DIR ${ABSL_EXT_PREFIX}/src/absl_external-build CACHE INTERNAL "")
set(ABSL_CONFIG_CMAKE_FILE "${ABSL_INSTALL_PREFIX}/lib/cmake/absl/abslConfig.cmake")

set(absl_DIR "${ABSL_INSTALL_PREFIX}/lib/cmake/absl" CACHE PATH "Path to absl config")
set(ABSL_DIR "${ABSL_INSTALL_PREFIX}/lib/cmake/absl" CACHE PATH "Path to absl config")
set(absl_ROOT "${ABSL_INSTALL_PREFIX}" CACHE PATH "absl root dir")
set(ABSL_ROOT "${ABSL_INSTALL_PREFIX}" CACHE PATH "absl root dir")

set(ABSL_INCLUDE_DIR "${ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dir")
set(ABSL_INCLUDE_DIRS "${ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dirs")
set(absl_INCLUDE_DIR "${ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dir")
set(absl_INCLUDE_DIRS "${ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dirs")
set(ABSL_LIBRARY_DIR "${ABSL_INSTALL_PREFIX}/lib" CACHE PATH "absl lib dir")
set(ABSL_LIB_DIR "${ABSL_INSTALL_PREFIX}/lib" CACHE PATH "absl lib dir")
set(absl_LIBRARY_DIR "${ABSL_INSTALL_PREFIX}/lib" CACHE PATH "absl lib dir")


setup_external_install_structure("${ABSL_INSTALL_PREFIX}")

if(NOT EXISTS "${ABSL_CONFIG_CMAKE_FILE}")
  message(STATUS "Abseil not found. Configuring external build...")
  ExternalProject_Add(
    absl_external
    GIT_REPOSITORY
      https://github.com/abseil/abseil-cpp
    GIT_TAG
      20260107.1
    PREFIX
      ${ABSL_EXT_PREFIX}
    PATCH_COMMAND
      git checkout -- . && git clean -df
      COMMAND ${CMAKE_COMMAND}
      -DABSL_PACKAGE_DIR=${ABSL_PACKAGE_DIR}
      -DABSL_SRC_DIR=${ABSL_SRC_DIR}
      -P "${ABSL_PACKAGE_DIR}/absl_patcher.cmake"

    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DCMAKE_INSTALL_PREFIX=${ABSL_INSTALL_PREFIX}
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
  message(STATUS "Abseil already installed at: ${ABSL_INSTALL_PREFIX}")
  if(NOT TARGET absl_external)
    add_custom_target(absl_external)
  endif()
endif()

include(${ABSL_PACKAGE_DIR}/absl_aggregate.cmake)
generate_absl_aggregate()