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
set(LITERTLM_TOKENIZERS_EXT_PREFIX ${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/tokenizers-cpp CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_SRC_DIR ${LITERTLM_TOKENIZERS_EXT_PREFIX}/src/tokenizers-cpp_external CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_SRC_SENTENCEPIECE_DIR ${LITERTLM_TOKENIZERS_SRC_DIR}/sentencepiece CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_SRC_INCLUDE_DIR ${LITERTLM_TOKENIZERS_SRC_DIR}/include CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_SRC_WEB_DIR ${LITERTLM_TOKENIZERS_SRC_DIR}/web/src CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_BUILD_DIR ${LITERTLM_TOKENIZERS_EXT_PREFIX}/src/tokenizers-cpp_external-build CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_INSTALL_PREFIX ${LITERTLM_TOKENIZERS_EXT_PREFIX}/install CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_INCLUDE_DIR
  "${LITERTLM_TOKENIZERS_INSTALL_PREFIX}/include"
  "${LITERTLM_TOKENIZERS_SRC_DIR}"
  "${LITERTLM_TOKENIZERS_SRC_INCLUDE_DIR}"
  "${LITERTLM_TOKENIZERS_SRC_WEB_DIR}"
  "${LITERTLM_TOKENIZERS_SRC_SENTENCEPIECE_DIR}"
  "${LITERTLM_TOKENIZERS_SRC_SENTENCEPIECE_DIR}/src"
 CACHE INTERNAL "")
set(LITERTLM_TOKENIZERS_LIB_CHECK "${LITERTLM_TOKENIZERS_BUILD_DIR}/libtokenizers_cpp.a")
set(LITERTLM_TOKENIZERSS_CMAKE_PATH "${LITERTLM_CMAKE_PACKAGES_DIR}/tokenizers/tokenizers.cmake" CACHE PATH "")

set(LITERTLM_TOKENIZERS_TAG "main" CACHE STRING "Tokenizers-cpp git tag")

if(TRUE)
  message(STATUS "tokenizers-cpp not found. Configuring external build...")
  ExternalProject_Add(
    tokenizers-cpp_external
    DEPENDS
      absl_external
      protobuf_external
      gtest_external
      sentencepiece_external
    GIT_REPOSITORY
      https://github.com/mlc-ai/tokenizers-cpp
    GIT_TAG
      ${LITERTLM_TOKENIZERS_TAG}
    PREFIX
      ${LITERTLM_TOKENIZERS_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_TOKENIZERS_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
      -DCMAKE_INSTALL_PREFIX=${LITERTLM_TOKENIZERS_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POLICY_DEFAULT_CMP0169=OLD
      -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -include cstdint"
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      # "-DCMAKE_PREFIX_PATH=${LITERTLM_ABSL_INSTALL_PREFIX};${LITERTLM_PROTOBUF_INSTALL_PREFIX};${LITERTLM_SENTENCEPIECE_INSTALL_PREFIX}"
  )

else()
    message(STATUS "tokenizers-cpp already installed at: ${LITERTLM_TOKENIZERS_INSTALL_PREFIX}")
    if(NOT TARGET tokenizers-cpp_external)
        add_custom_target(tokenizers-cpp_external)
    endif()
endif()

import_static_lib(imp_tokenizers_c
  "${LITERTLM_TOKENIZERS_BUILD_DIR}/libtokenizers_c.a")
import_static_lib(imp_tokenizers_cpp
  "${LITERTLM_TOKENIZERS_BUILD_DIR}/libtokenizers_cpp.a")

add_library(tokenizers_libs INTERFACE)
target_include_directories(tokenizers_libs INTERFACE ${LITERTLM_TOKENIZERS_INCLUDE_DIR})

target_link_libraries(tokenizers_libs INTERFACE
  imp_tokenizers_c
  imp_tokenizers_cpp
)

if(NOT TARGET LiteRTLM::tokenizers::tokenizers)
    add_library(LiteRTLM::tokenizers::tokenizers INTERFACE IMPORTED GLOBAL)
    target_link_libraries(LiteRTLM::tokenizers::tokenizers INTERFACE tokenizers_libs)
endif()

set(_tokenizers_lib_paths
  "${LITERTLM_TOKENIZERS_BUILD_DIR}/libtokenizers_c.a"
  "${LITERTLM_TOKENIZERS_BUILD_DIR}/libtokenizers_cpp.a"
)