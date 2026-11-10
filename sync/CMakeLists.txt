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
# Sha256 Hasher
# ==============================================================================
add_litertlm_library(sync_sha256 STATIC
  sha256.cc
)
add_library(LiteRTLM::Sync::Sha256 ALIAS sync_sha256)

target_include_directories(sync_sha256
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(sync_sha256
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Model Sync
# ==============================================================================
add_litertlm_library(sync_model_sync STATIC
  model_sync.cc
)
add_library(LiteRTLM::Sync::ModelSync ALIAS sync_model_sync)

target_include_directories(sync_model_sync
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
)

target_link_libraries(sync_model_sync
  PUBLIC
    LiteRTLM::Sync::Sha256
    LiteRTLM::Schema::Core::Read
    LiteRTLM::Schema::Core::HeaderSchema
    LITERTLM_DEPS
)

