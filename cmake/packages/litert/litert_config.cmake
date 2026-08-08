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

set(LITERTLM_LITERT_EXT_PREFIX "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/litert" CACHE PATH "")
set(LITERTLM_LITERT_BUILD_DIR "${LITERTLM_LITERT_EXT_PREFIX}/src/litert_external-build" CACHE PATH "")
set(LITERTLM_LITERT_SRC_DIR "${LITERTLM_LITERT_EXT_PREFIX}/src/litert_external/litert" CACHE PATH "")
set(LITERTLM_LITERT_STAMP_DIR "${LITERTLM_LITERT_EXT_PREFIX}/src/litert_external-stamp" CACHE INTERNAL "")
set(LITERTLM_LITERT_SHIM_PATH "${LITERTLM_LITERT_PACKAGE_DIR}/litert_shim.cmake" CACHE PATH "")
set(LITERTLM_LITERT_VENDOR_SHIM_PATH "${LITERTLM_LITERT_PACKAGE_DIR}/shims/vendor_shim.cmake" CACHE PATH "")
set(LITERT_BUILD_CONFIG_DISABLE_GPU_VAL "${LITERTLM_LITERT_BUILD_CONFIG_DISABLE_GPU_VAL}" CACHE PATH "")
set(LITERT_BUILD_CONFIG_DISABLE_NPU_VAL "${LITERTLM_LITERT_BUILD_CONFIG_DISABLE_NPU_VAL}" CACHE PATH "")
set(LITERTLM_LITERT_CONFIG_PATH "${LITERTLM_LITERT_PACKAGE_DIR}/litert_config.cmake" CACHE PATH "")
set(LITERTLM_LITERT_TARGET_MAP_PATH "${LITERTLM_LITERT_PACKAGE_DIR}/litert_target_map.cmake" CACHE PATH "")


set(LITERTLM_LITERT_INCLUDE_PATHS
  "${LITERTLM_LITERT_SRC_DIR}"
  "${LITERTLM_LITERT_SRC_DIR}/.."
  "${LITERTLM_LITERT_SRC_DIR}/cc/include"
  "${LITERTLM_LITERT_SRC_DIR}/compiler/include"
  "${LITERTLM_LITERT_SRC_DIR}/core/include"
  "${LITERTLM_LITERT_BUILD_DIR}/runtime/include"
  "${LITERTLM_LITERT_BUILD_DIR}/include"
  "${LITERTLM_LITERT_SRC_DIR}/../support"
  "${LITERTLM_LITERT_BUILD_DIR}/c/include"
  
  
  CACHE PATH "")

set(LITERTLM_LITERT_CXX_FLAGS_Construct "${CMAKE_CXX_FLAGS} -fpermissive")

if(LITERTLM_OPENCL_INCLUDE_DIR)
  string(APPEND LITERTLM_LITERT_CXX_FLAGS_Construct " -isystem ${LITERTLM_OPENCL_INCLUDE_DIR}")
endif()

set(LITERTLM_LITERT_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/litert"
    CACHE PATH "Path to LiteRT related build scripts")


set(LITERTLM_LITERT_SUPPORT_DIR "${LITERTLM_SRC_DIR}/../support" CACHE PATH "")