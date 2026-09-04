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
# Conversation
# ==============================================================================
add_litertlm_library(c_conversation STATIC
  conversation.cc
)
add_library(LiteRTLM::C::Conversation ALIAS c_conversation)

target_include_directories(c_conversation
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(c_conversation
  PUBLIC
    LITERTLM_DEPS
    LiteRTLM::Runtime::Conversation
    LiteRTLM::Runtime::Core
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Executor
    LiteRTLM::Support::Tokenizer
)

# ==============================================================================
# Engine
# ==============================================================================
add_litertlm_library(c_engine STATIC
  engine.cc
)
add_library(LiteRTLM::C::Engine ALIAS c_engine)

target_include_directories(c_engine
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(c_engine
  PUBLIC
    LITERTLM_DEPS
    LiteRTLM::C::EmbeddingEngine
    LiteRTLM::C::Conversation
    LiteRTLM::Runtime::Conversation
    LiteRTLM::Runtime::Core
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Executor
    LiteRTLM::Support::Tokenizer
)

# ==============================================================================
# Embedding Engine
# ==============================================================================
add_litertlm_library(c_embedding_engine STATIC
  embedding_engine.cc
)
add_library(LiteRTLM::C::EmbeddingEngine ALIAS c_embedding_engine)

target_include_directories(c_embedding_engine
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(c_embedding_engine
  PUBLIC
    LITERTLM_DEPS
    LiteRTLM::C::Conversation
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Executor
)

# ==============================================================================
# Capabilities
# ==============================================================================
add_litertlm_library(c_capabilities STATIC
  capabilities.cc
)
add_library(LiteRTLM::C::Capabilities ALIAS c_capabilities)

target_include_directories(c_capabilities
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(c_capabilities
  PUBLIC
    LITERTLM_DEPS
    LiteRTLM::Schema::Capabilities
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(c_libs INTERFACE)
add_library(LiteRTLM::C ALIAS c_libs)

target_link_libraries(c_libs INTERFACE
  LiteRTLM::C::Capabilities
  LiteRTLM::C::Conversation
  LiteRTLM::C::Engine
  LiteRTLM::C::EmbeddingEngine
)
