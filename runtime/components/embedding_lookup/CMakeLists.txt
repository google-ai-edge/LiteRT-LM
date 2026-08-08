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
# Embedding Lookup Interface
# ==============================================================================
add_litertlm_library(runtime_components_embedding_lookup_interface INTERFACE)
add_library(LiteRTLM::Runtime::Components::EmbeddingLookup::Interface 
  ALIAS runtime_components_embedding_lookup_interface)

target_include_directories(runtime_components_embedding_lookup_interface
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

# ==============================================================================
# Embedding Lookup End of Multi Modal
# ==============================================================================
add_litertlm_library(runtime_components_embedding_lookup_embedding_lookup_end_of_multi_modal STATIC
  embedding_lookup_end_of_multi_modal.cc
)
add_library(LiteRTLM::Runtime::Components::EmbeddingLookup::EndOfMultiModal 
  ALIAS runtime_components_embedding_lookup_embedding_lookup_end_of_multi_modal)

target_include_directories(runtime_components_embedding_lookup_embedding_lookup_end_of_multi_modal
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_embedding_lookup_embedding_lookup_end_of_multi_modal
  PUBLIC
    LiteRTLM::Runtime::Components::EmbeddingLookup::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# Embedding Lookup Manager
# ==============================================================================
add_litertlm_library(runtime_components_embedding_lookup_embedding_lookup_manager STATIC
  embedding_lookup_manager.cc
)
add_library(LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
  ALIAS runtime_components_embedding_lookup_embedding_lookup_manager)

target_include_directories(runtime_components_embedding_lookup_embedding_lookup_manager
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
  )

target_link_libraries(runtime_components_embedding_lookup_embedding_lookup_manager
  PUBLIC
    LiteRTLM::Runtime::Executor::LlmExecutorIoTypes
    LiteRTLM::Runtime::Components::EmbeddingLookup::EndOfMultiModal
    LiteRTLM::Runtime::Components::EmbeddingLookup::MultiModal
    LiteRTLM::Runtime::Components::EmbeddingLookup::Text
    LITERTLM_DEPS
)

# ==============================================================================
# Embedding Lookup Multi Modal
# ==============================================================================
add_litertlm_library(runtime_components_embedding_lookup_embedding_lookup_multi_modal STATIC
  embedding_lookup_multi_modal.cc
)
add_library(LiteRTLM::Runtime::Components::EmbeddingLookup::MultiModal 
  ALIAS runtime_components_embedding_lookup_embedding_lookup_multi_modal)

target_include_directories(runtime_components_embedding_lookup_embedding_lookup_multi_modal
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(runtime_components_embedding_lookup_embedding_lookup_multi_modal
  PUBLIC
    LiteRTLM::Runtime::Components::EmbeddingLookup::Interface
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LITERTLM_DEPS
)

# ==============================================================================
# Embedding Lookup Text
# ==============================================================================
add_litertlm_library(runtime_components_embedding_lookup_embedding_lookup_text STATIC
  embedding_lookup_text.cc
)
add_library(LiteRTLM::Runtime::Components::EmbeddingLookup::Text
  ALIAS runtime_components_embedding_lookup_embedding_lookup_text)

target_include_directories(runtime_components_embedding_lookup_embedding_lookup_text
  PUBLIC
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
    ${CMAKE_BINARY_DIR}
  )

target_link_libraries(runtime_components_embedding_lookup_embedding_lookup_text
  PUBLIC
    LiteRTLM::Runtime::Components::EmbeddingLookup::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_components_embedding_lookup_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::EmbeddingLookup 
  ALIAS runtime_components_embedding_lookup_libs)

target_link_libraries(runtime_components_embedding_lookup_libs INTERFACE
  LiteRTLM::Runtime::Components::EmbeddingLookup::Interface
  LiteRTLM::Runtime::Components::EmbeddingLookup::EndOfMultiModal
  LiteRTLM::Runtime::Components::EmbeddingLookup::Manager
  LiteRTLM::Runtime::Components::EmbeddingLookup::MultiModal
  LiteRTLM::Runtime::Components::EmbeddingLookup::Text
)