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

set(LITERTLM_RE2_EXT_PREFIX "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/re2" CACHE PATH "")
set(LITERTLM_RE2_INSTALL_PREFIX "${LITERTLM_RE2_EXT_PREFIX}/install" CACHE PATH "")
set(LITERTLM_RE2_CONFIG_PATH "${LITERTLM_RE2_PACKAGE_DIR}/re2_config.cmake" CACHE PATH "")
set(LITERTLM_RE2_SRC_DIR "${LITERTLM_RE2_EXT_PREFIX}/src/re2_external" CACHE PATH "")
set(LITERTLM_RE2_LIB_DIR "${LITERTLM_RE2_INSTALL_PREFIX}/lib" CACHE PATH "")
set(LITERTLM_RE2_BIN_DIR "${LITERTLM_RE2_INSTALL_PREFIX}/bin" CACHE PATH "")
set(LITERTLM_RE2_INCLUDE_DIR "${LITERTLM_RE2_INSTALL_PREFIX}/include" CACHE PATH "")
set(LITERTLM_RE2_SHIM_PATH "${LITERTLM_RE2_PACKAGE_DIR}/re2_shim.cmake" CACHE PATH "")
set(LITERTLM_RE2_AGGREGATE_PATH "${LITERTLM_RE2_PACKAGE_DIR}/re2_aggregate.cmake" CACHE PATH "")
set(LITERTLM_RE2_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/re2"
    CACHE PATH "Path to RE2 related build scripts")
