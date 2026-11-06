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

# Oniguruma (https://github.com/kkos/oniguruma) is the regular expression
# engine behind the HuggingFace tokenizers port in support/tokenizer/tokenizers.
set(LITERTLM_ONIGURUMA_EXT_PREFIX "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/oniguruma" CACHE PATH "")
set(LITERTLM_ONIGURUMA_INSTALL_PREFIX "${LITERTLM_ONIGURUMA_EXT_PREFIX}/install" CACHE PATH "")
set(LITERTLM_ONIGURUMA_CONFIG_PATH "${LITERTLM_ONIGURUMA_PACKAGE_DIR}/oniguruma_config.cmake" CACHE PATH "")
set(LITERTLM_ONIGURUMA_SRC_DIR "${LITERTLM_ONIGURUMA_EXT_PREFIX}/src/oniguruma_external" CACHE PATH "")
set(LITERTLM_ONIGURUMA_BUILD_DIR "${LITERTLM_ONIGURUMA_EXT_PREFIX}/src/oniguruma_external-build" CACHE PATH "")
# The install layout is pinned with -DCMAKE_INSTALL_LIBDIR=lib (see
# oniguruma.cmake), so the library is never installed under lib64/.
set(LITERTLM_ONIGURUMA_LIB_DIR "${LITERTLM_ONIGURUMA_INSTALL_PREFIX}/lib" CACHE PATH "")
set(LITERTLM_ONIGURUMA_BIN_DIR "${LITERTLM_ONIGURUMA_INSTALL_PREFIX}/bin" CACHE PATH "")
set(LITERTLM_ONIGURUMA_INCLUDE_DIR "${LITERTLM_ONIGURUMA_INSTALL_PREFIX}/include" CACHE PATH "")
set(LITERTLM_ONIGURUMA_CONFIG_CMAKE_FILE "${LITERTLM_ONIGURUMA_LIB_DIR}/cmake/oniguruma/onigurumaConfig.cmake" CACHE PATH "")
set(LITERTLM_ONIGURUMA_AGGREGATE_PATH "${LITERTLM_ONIGURUMA_PACKAGE_DIR}/oniguruma_aggregate.cmake" CACHE PATH "")
set(LITERTLM_ONIGURUMA_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/oniguruma"
    CACHE PATH "Path to Oniguruma related build scripts")
set(LITERTLM_ONIGURUMA_TARGET_MAP_PATH "${LITERTLM_ONIGURUMA_PACKAGE_DIR}/oniguruma_${LITERTLM_ORCHESTRATION_PHASE}_target_map.cmake" CACHE PATH "")
