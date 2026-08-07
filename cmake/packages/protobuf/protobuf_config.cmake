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

set(LITERTLM_PROTOBUF_EXT_PREFIX "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/protobuf" CACHE PATH "")
set(LITERTLM_PROTOBUF_INSTALL_PREFIX "${LITERTLM_PROTOBUF_EXT_PREFIX}/install" CACHE PATH "")
set(LITERTLM_PROTOBUF_CONFIG_PATH "${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_config.cmake" CACHE PATH "")
set(LITERTLM_PROTOBUF_SRC_DIR "${LITERTLM_PROTOBUF_EXT_PREFIX}/src/protobuf_external" CACHE PATH "")
set(LITERTLM_PROTOBUF_STAMP_DIR "${LITERTLM_PROTOBUF_EXT_PREFIX}/src/protobuf_external-stamp" CACHE INTERNAL "")
set(LITERTLM_PROTOBUF_INCLUDE_DIR
  "${LITERTLM_PROTOBUF_INSTALL_PREFIX}/include"
  "${LITERTLM_PROTOBUF_INSTALL_PREFIX}"
  CACHE PATH "")
set(LITERTLM_PROTOBUF_LIB_DIR "${LITERTLM_PROTOBUF_INSTALL_PREFIX}/lib" CACHE PATH "")
set(LITERTLM_PROTOBUF_LITE_LIBRARY "${LITERTLM_PROTOBUF_INSTALL_PREFIX}/lib/libprotobuf-lite.a" CACHE PATH "")
set(LITERTLM_PROTOBUF_SHIM_PATH "${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_shim.cmake" CACHE PATH "")
set(LITERTLM_PROTOBUF_BIN_DIR "${LITERTLM_PROTOBUF_INSTALL_PREFIX}/bin" CACHE PATH "")
set(LITERTLM_PROTOBUF_TARGET_MAP_PATH "${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_target_map.cmake" CACHE PATH "")
set(LITERTLM_PROTOBUF_AGGREGATE_PATH "${LITERTLM_PROTOBUF_PACKAGE_DIR}/protobuf_aggregate.cmake" CACHE PATH "")
set(LITERTLM_PROTOC_EXECUTABLE "${LITERTLM_PROTOBUF_BIN_DIR}/protoc" CACHE PATH "")
set(LITERTLM_PROTOC_EXE "${LITERTLM_PROTOBUF_BIN_DIR}/protoc" CACHE PATH "")
set(protobuf_generate_PROTOC_EXE "${LITERTLM_PROTOBUF_BIN_DIR}/protoc" CACHE PATH "")

if("${LITERTLM_ORCHESTRATION_PHASE}" STREQUAL "litert_lm")
    message(STATUS "[LiteRTLM] Protobuf: Using host protoc at ${LITERTLM_HOST_PROTOC}")
    set(LITERTLM_PROTOBUF_BIN_DIR "${LITERTLM_HOST_PROTOC_BIN_DIR}" CACHE PATH "Host Protobuf binary path")
    set(LITERTLM_PROTOC_EXECUTABLE "${LITERTLM_HOST_PROTOC}" CACHE PATH "Host protoc")
    set(protobuf_generate_PROTOC_EXE "${LITERTLM_HOST_PROTOC}" CACHE PATH "Host protoc for generator module")
endif()
set(LITERTLM_PROTOBUF_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/protobuf"
    CACHE PATH "Path to Protobuf related build scripts")