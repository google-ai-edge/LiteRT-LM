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

set(LITERTLM_PROTOBUF_CONFIG_PATH "${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_config.cmake" CACHE PATH "")
include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_ABSL_CONFIG_PATH}")

setup_external_install_structure("${LITERTLM_PROTOBUF_INSTALL_PREFIX}")

set(LITERTLM_PROTOBUF_TAG "v35.1" CACHE STRING "Protobuf git tag")

include(ExternalProject)
if(NOT EXISTS "${LITERTLM_PROTOBUF_CONFIG_CMAKE_FILE}")
  message(STATUS "Protobuf not found. Configuring external build...")
  ExternalProject_Add(
    protobuf_external
    DEPENDS
      absl_external
      gtest_external
    GIT_REPOSITORY
      https://github.com/protocolbuffers/protobuf
    GIT_TAG
      ${LITERTLM_PROTOBUF_TAG}
    PREFIX
      ${LITERTLM_PROTOBUF_EXT_PREFIX}
    UPDATE_COMMAND
      git fetch origin ${LITERTLM_PROTOBUF_TAG}
      COMMAND git reset --hard FETCH_HEAD
      COMMAND git clean -dfx
    PATCH_COMMAND
      ${CMAKE_COMMAND}
      "-DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}"
      "-DLITERTLM_CMAKE_PACKAGES_DIR=${LITERTLM_CMAKE_PACKAGES_DIR}"
      "-DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}"
      "-DLITERTLM_PROTOBUF_PACKAGE_DIR=${LITERTLM_PROTOBUF_PACKAGE_DIR}"
      "-DLITERTLM_PROTOBUF_CONFIG_PATH=${LITERTLM_PROTOBUF_CONFIG_PATH}"
      "-DLITERTLM_ABSL_PACKAGE_DIR=${LITERTLM_ABSL_PACKAGE_DIR}"
      "-DLITERTLM_ABSL_CONFIG_PATH=${LITERTLM_ABSL_CONFIG_PATH}"
      "-P ${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_patcher.cmake"
    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      "-DLITERTLM_ORCHESTRATION_PHASE=${LITERTLM_ORCHESTRATION_PHASE}"
      "-DLITERTLM_EXTERNAL_PROJECT_BIN_DIR=${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}"
      "-DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}"
      "-DLITERTLM_CMAKE_PACKAGES_DIR=${LITERTLM_CMAKE_PACKAGES_DIR}"
      "-DLITERTLM_PROTOBUF_PACKAGE_DIR=${LITERTLM_PROTOBUF_PACKAGE_DIR}"
      "-DLITERTLM_PROTOBUF_CONFIG_PATH=${LITERTLM_PROTOBUF_CONFIG_PATH}"
      "-DLITERTLM_ABSL_CONFIG_PATH=${LITERTLM_ABSL_CONFIG_PATH}"
      "-DLITERTLM_ABSL_PACKAGE_DIR=${LITERTLM_ABSL_PACKAGE_DIR}"
      "-DCMAKE_PREFIX_PATH=${LITERTLM_GTEST_INSTALL_PREFIX};${LITERTLM_ABSL_INSTALL_PREFIX}"
      "-DCMAKE_INSTALL_PREFIX=${LITERTLM_PROTOBUF_INSTALL_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_POLICY_DEFAULT_CMP0169=OLD"
      "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -I${LITERTLM_ABSL_INCLUDE_DIR}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
      "-Dprotobuf_BUILD_TESTS=OFF"
      "-Dprotobuf_BUILD_LIBPROTOC=ON"
      "-Dprotobuf_BUILD_PROTOBUF_BINARIES=ON"
      "-Dprotobuf_LOCAL_DEPENDENCIES_ONLY=ON"
      "-Dabsl_DIR=${LITERTLM_ABSL_INSTALL_PREFIX}/lib/cmake/absl"
      "-DGTest_DIR=${LITERTLM_GTEST_INSTALL_PREFIX}/lib/cmake/GTest"
      "-DProtobuf_DIR=${LITERTLM_PROTOBUF_INSTALL_PREFIX}/lib/cmake/Protobuf"
  )

else()
  message(STATUS "[LiteRTLM] Protobuf already installed at: ${LITERTLM_PROTOBUF_INSTALL_PREFIX}")
  if(NOT TARGET protobuf_external)
    add_custom_target(protobuf_external)
  endif()
endif()

include(${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_aggregate.cmake)
generate_protobuf_aggregate()