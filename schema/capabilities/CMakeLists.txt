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

# ==============================================================================
# Speculative Decoding
# ==============================================================================
add_litertlm_library(runtime_core_speculative_decoding STATIC
  speculative_decoding.cc
)
add_library(LiteRTLM::Schema::Speculative::SpeculativeDecoding ALIAS runtime_core_speculative_decoding)

target_include_directories(runtime_core_speculative_decoding
  PRIVATE
    "${LITERTLM_GENERATED_SRC_DIR}"
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_core_speculative_decoding
  PUBLIC
    LiteRTLM::Schema::Core::HeaderSchema
    LiteRTLM::Schema::Core::Read
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(schema_capabilities_libs INTERFACE)
add_library(LiteRTLM::Schema::Capabilities ALIAS schema_capabilities_libs)

target_link_libraries(schema_capabilities_libs INTERFACE
  LiteRTLM::Schema::Speculative::SpeculativeDecoding
)