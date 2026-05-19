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

set(PROTO_EXT_PREFIX ${EXTERNAL_PROJECT_BINARY_DIR}/protobuf CACHE INTERNAL "")
set(PROTO_INSTALL_PREFIX ${PROTO_EXT_PREFIX}/install CACHE INTERNAL "")
set(PROTO_CONFIG_CMAKE_FILE "${PROTO_INSTALL_PREFIX}/lib/cmake/protobuf/protobuf-config.cmake" CACHE INTERNAL "")
set(PROTO_SRC_DIR ${PROTO_EXT_PREFIX}/src/protobuf_external CACHE INTERNAL "")
set(PROTO_INCLUDE_DIR ${PROTO_INSTALL_PREFIX}/include CACHE INTERNAL "")
set(PROTO_LIB_DIR ${PROTO_INSTALL_PREFIX}/lib CACHE INTERNAL "")
set(PROTO_LITE_LIBRARY ${PROTO_INSTALL_PREFIX}/lib/libprotobuf-lite.a CACHE INTERNAL "")
set(PROTO_BIN_DIR ${PROTO_INSTALL_PREFIX}/bin CACHE INTERNAL "")
set(PROTO_PROTOC_EXECUTABLE ${PROTO_BIN_DIR}/protoc CACHE INTERNAL "")
set(protobuf_generate_PROTOC_EXE ${PROTO_BIN_DIR}/protoc CACHE INTERNAL "")

if(LITERTLM_HOST_PROTOC AND EXISTS "${LITERTLM_HOST_PROTOC}")
    message(STATUS "[LiteRTLM] Protobuf: Using host protoc at ${LITERTLM_HOST_PROTOC}")
    set(PROTO_BIN_DIR ${LITERTLM_HOST_PROTOC_BIN_DIR} CACHE INTERNAL "Host Protobuf binary path")
    set(PROTO_PROTOC_EXECUTABLE "${LITERTLM_HOST_PROTOC}" CACHE INTERNAL "Host protoc")
    set(protobuf_generate_PROTOC_EXE "${LITERTLM_HOST_PROTOC}" CACHE INTERNAL "Host protoc for generator module")
endif()


setup_external_install_structure("${PROTO_INSTALL_PREFIX}")

if(NOT EXISTS "${PROTO_CONFIG_CMAKE_FILE}")
  message(STATUS "Protobuf not found. Configuring external build...")
  ExternalProject_Add(
    protobuf_external
    DEPENDS
      absl_external
      gtest_external
    GIT_REPOSITORY
      https://github.com/protocolbuffers/protobuf
    GIT_TAG
      v6.31.1
    PREFIX
      ${PROTO_EXT_PREFIX}
    PATCH_COMMAND
        git checkout -- . && git clean -df
      COMMAND ${CMAKE_COMMAND}
      -DPROTO_SRC_DIR=${PROTO_SRC_DIR}
      -DLITERTLM_PROTO_SHIM_PATH="${PROTOBUF_PACKAGE_DIR}/protobuf_shims.cmake"
      -P "${PROTOBUF_PACKAGE_DIR}/protobuf_patcher.cmake"
    CMAKE_ARGS
      ${LITERTLM_TOOLCHAIN_FILE}
      ${LITERTLM_TOOLCHAIN_ARGS}
      -DCMAKE_PREFIX_PATH=${GTEST_INSTALL_PREFIX};${ABSL_INSTALL_PREFIX}
      -DCMAKE_INSTALL_PREFIX=${PROTO_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POLICY_DEFAULT_CMP0169=OLD
      -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -I${ABSL_INCLUDE_DIR}"
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -Dprotobuf_BUILD_TESTS=OFF
      -Dprotobuf_BUILD_LIBPROTOC=ON
      -Dprotobuf_BUILD_PROTOBUF_BINARIES=ON
      -Dprotobuf_LOCAL_DEPENDENCIES_ONLY=ON
      -Dabsl_DIR=${ABSL_INSTALL_PREFIX}/lib/cmake/absl
      -DGTest_DIR=${GTEST_INSTALL_PREFIX}/lib/cmake/GTest
      -DProtobuf_DIR=${PROTO_INSTALL_PREFIX}/lib/cmake/Protobuf

      -DABSL_PACKAGE_DIR=${ABSL_PACKAGE_DIR}
      -DABSL_INCLUDE_DIR=${ABSL_INCLUDE_DIR}
      -DABSL_LIB_DIR=${ABSL_LIB_DIR}
      -DLITERTLM_MODULES_DIR=${LITERTLM_MODULES_DIR}
      -DLITERTLM_PROTO_SHIM_PATH="${PROTOBUF_PACKAGE_DIR}/protobuf_shims.cmake"

  )


else()
  message(STATUS "[LiteRTLM] Protobuf already installed at: ${PROTO_INSTALL_PREFIX}")
  if(NOT TARGET protobuf_external)
    add_custom_target(protobuf_external)
  endif()
endif()

include(${PROTOBUF_PACKAGE_DIR}/protobuf_aggregate.cmake)
generate_protobuf_aggregate()