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

set(LITERTLM_FLATBUFFERS_TARGET_MAP
  "flatbuffers::flatbuffers=${LITERTLM_FLATBUFFERS_LIB_DIR}/libflatbuffers.a"
  "flatbuffers=${LITERTLM_FLATBUFFERS_LIB_DIR}/libflatbuffers.a"
  "flatbuffers::libflatbuffers=${LITERTLM_FLATBUFFERS_LIB_DIR}/libflatbuffers.a"
  "libflatbuffers=${LITERTLM_FLATBUFFERS_LIB_DIR}/libflatbuffers.a"
  CACHE INTERNAL "Mapping of Flatbuffers targets to their corresponding library paths"
)

set(LITERTLM_FLATC_TARGET_MAP
  "flatbuffers::flatc=${LITERTLM_FLATBUFFERS_BIN_DIR}/flatc"
  "flatc=${LITERTLM_FLATBUFFERS_BIN_DIR}/flatc"
  CACHE INTERNAL "Mapping of Flatbuffers targets to their corresponding binary paths"
)