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

set(LITERTLM_PROTOBUF_TARGET_MAP
  "protobuf::libprotobuf-lite=${LITERTLM_PROTOBUF_LIB_DIR}/libprotobuf-lite.a"
  "protobuf::libprotobuf=${LITERTLM_PROTOBUF_LIB_DIR}/libprotobuf.a"
  "protobuf::libprotoc=${LITERTLM_PROTOBUF_LIB_DIR}/libprotoc.a"
  "protobuf::libupb=${LITERTLM_PROTOBUF_LIB_DIR}/libupb.a"
  "protobuf::libutf8_validity=${LITERTLM_PROTOBUF_LIB_DIR}/libutf8_validity.a"
  "protobuf::libutf8_range=${LITERTLM_PROTOBUF_LIB_DIR}/libutf8_range.a"
)