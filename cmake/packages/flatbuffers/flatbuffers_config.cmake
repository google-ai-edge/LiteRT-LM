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

set(LITERTLM_FLATBUFFERS_EXT_PREFIX "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/flatbuffers" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_INSTALL_PREFIX "${LITERTLM_FLATBUFFERS_EXT_PREFIX}/install" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_INCLUDE_DIR "${LITERTLM_FLATBUFFERS_INSTALL_PREFIX}/include" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_SRC_DIR "${LITERTLM_FLATBUFFERS_EXT_PREFIX}/src" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_STAMP_DIR "${LITERTLM_FLATBUFFERS_EXT_PREFIX}/src/flatbuffers_external-stamp" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_BIN_DIR "${LITERTLM_FLATBUFFERS_INSTALL_PREFIX}/bin" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_LIB_DIR "${LITERTLM_FLATBUFFERS_INSTALL_PREFIX}/lib" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_DIR "${LITERTLM_FLATBUFFERS_LIB_DIR}/cmake/flatbuffers" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_CMAKE_CONFIG_FILE "${LITERTLM_FLATBUFFERS_LIB_DIR}/cmake/flatbuffers/flatbuffers-config.cmake" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_FLATC_EXECUTABLE "${LITERTLM_FLATBUFFERS_BIN_DIR}/flatc" CACHE INTERNAL "")
set(LITERTLM_FLATC_EXECUTABLE "${LITERTLM_FLATBUFFERS_BIN_DIR}/flatc" CACHE INTERNAL "")
set(LITERTLM_FLATBUFFERS_AGGREGATE_PATH "${LITERTLM_FLATBUFFERS_PACKAGE_DIR}/flatbuffers_aggregate.cmake" CACHE PATH "")

if(DEFINED LITERTLM_HOST_FLATC)
    message(STATUS "[LiteRTLM] FlatBuffers: Using host flatc at ${LITERTLM_HOST_FLATC_BIN_DIR}")
    set(LITERTLM_FLATBUFFERS_BIN_DIR "${LITERTLM_HOST_BIN_DIR}" CACHE INTERNAL "Host Flatbuffers binary path")
    set(LITERTLM_FLATBUFFERS_FLATC_EXECUTABLE "${LITERTLM_HOST_FLATC}" CACHE INTERNAL "Host flatc")
    set(LITERTLM_FLATC_EXECUTABLE "${LITERTLM_HOST_FLATC}" CACHE INTERNAL "Host flatc")

endif()

set(LITERTLM_FLATBUFFERS_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/flatbuffers"
    CACHE PATH "Path to Flatbuffers related build scripts")