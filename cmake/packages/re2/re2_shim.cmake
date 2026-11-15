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

include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_RE2_CONFIG_PATH}")

set(absl_DIR "${LITERTLM_ABSL_BUILD_DIR}" CACHE PATH "Path to absl config")

set(ABSL_INCLUDE_DIR "${LITERTLM_ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dir")
set(ABSL_INCLUDE_DIRS "${LITERTLM_ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dirs")
set(absl_INCLUDE_DIR "${LITERTLM_ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dir")
set(absl_INCLUDE_DIRS "${LITERTLM_ABSL_INSTALL_PREFIX}/include" CACHE PATH "absl include dirs")
set(ABSL_LIBRARY_DIR "${LITERTLM_ABSL_INSTALL_PREFIX}/lib" CACHE PATH "absl lib dir")
set(ABSL_LIB_DIR "${LITERTLM_ABSL_INSTALL_PREFIX}/lib" CACHE PATH "absl lib dir")
set(absl_LIBRARY_DIR "${LITERTLM_ABSL_INSTALL_PREFIX}/lib" CACHE PATH "absl lib dir")


list(APPEND CMAKE_PREFIX_PATH "${LITERTLM_ABSL_INSTALL_PREFIX}")
list(APPEND CMAKE_SYSTEM_PREFIX_PATH "${LITERTLM_ABSL_INSTALL_PREFIX}")