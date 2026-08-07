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

set(LITERTLM_GTEST_EXT_PREFIX "${LITERTLM_EXTERNAL_PROJECT_BIN_DIR}/googletest" CACHE PATH "")
set(LITERTLM_GTEST_SRC_DIR "${LITERTLM_GTEST_EXT_PREFIX}/src" CACHE INTERNAL "")
set(LITERTLM_GTEST_STAMP_DIR "${LITERTLM_GTEST_EXT_PREFIX}/src/gtest_external-stamp" CACHE INTERNAL "")
set(LITERTLM_GTEST_INSTALL_PREFIX "${LITERTLM_GTEST_EXT_PREFIX}/install" CACHE PATH "")
set(LITERTLM_GTEST_INCLUDE_DIR "${LITERTLM_GTEST_INSTALL_PREFIX}/include" CACHE PATH "")
set(LITERTLM_GTEST_CONFIG_CMAKE_FILE "${LITERTLM_GTEST_INSTALL_PREFIX}/lib/cmake/GTest/GTestConfig.cmake"  CACHE PATH "")
set(LITERTLM_GTEST_PACKAGE_DIR
    "${LITERTLM_CMAKE_PACKAGES_DIR}/gtest"
    CACHE PATH "Path to gtest related build scripts")