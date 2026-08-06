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

set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(RE2_EXT_PREFIX ${EXTERNAL_PROJECT_BINARY_DIR}/re2 CACHE INTERNAL "")
set(RE2_INSTALL_PREFIX ${RE2_EXT_PREFIX}/install CACHE INTERNAL "")
set(RE2_LIB_DIR ${RE2_INSTALL_PREFIX}/lib CACHE INTERNAL "")
set(RE2_INCLUDE_DIR ${RE2_INSTALL_PREFIX}/include CACHE INTERNAL "")
set(RE2_SRC_DIR ${RE2_EXT_PREFIX}/src/re2_external)
set(RE2_CONFIG_CMAKE_FILE "${RE2_LIB_DIR}/cmake/re2/re2Config.cmake" CACHE INTERNAL "")


setup_external_install_structure("${RE2_INSTALL_PREFIX}")

if(NOT EXISTS "${RE2_CONFIG_CMAKE_FILE}")
  message(STATUS "RE2 not found. Configuring external build...")
  ExternalProject_Add(
    re2_external
    DEPENDS
      absl_external
    GIT_REPOSITORY
      https://github.com/google/re2/
    GIT_TAG
      main
    PREFIX
      ${RE2_EXT_PREFIX}
    PATCH_COMMAND
      git checkout -- . && git clean -df
      COMMAND ${CMAKE_COMMAND}
        -DABSL_CONFIG_CMAKE_FILE=${ABSL_CONFIG_CMAKE_FILE}
        -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
        -DLITERTLM_RE2_SRC_DIR=${RE2_SRC_DIR} 
        -DLITERTLM_RE2_SHIM_PATH="${RE2_PACKAGE_DIR}/re2_shim.cmake"
        -P "${RE2_PACKAGE_DIR}/re2_patcher.cmake"

    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DCMAKE_PREFIX_PATH=${ABSL_INSTALL_PREFIX}
      -DCMAKE_INSTALL_PREFIX=${RE2_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=lib
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
      -DABSL_PACKAGE_DIR=${ABSL_PACKAGE_DIR}
      -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
      -DLITERTLM_RE2_SHIM_PATH="${RE2_PACKAGE_DIR}"
  )

else()
  message(STATUS "RE2 already installed at: ${RE2_INSTALL_PREFIX}")
  if(NOT TARGET re2_external)
    add_custom_target(re2_external)
  endif()
endif()

include(${RE2_PACKAGE_DIR}/re2_aggregate.cmake)
generate_re2_aggregate()