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

# --- Dependency Orchestration ---
# Warning: Order matters to satisfy inter-dependency requirements
set(LITERTLM_DEPENDENCY_ORDER
    opencl
    absl
    gtest
    protobuf
    flatbuffers
    sentencepiece
    tokenizers
    re2
    tflite
    litert
)

foreach(package ${LITERTLM_DEPENDENCY_ORDER})
    load_package(${package}) # macros.cmake
endforeach()
