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


set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(model_data_processor)

add_litertlm_library(runtime_conversation_conversation STATIC
  conversation.cc
)
add_library(LiteRTLM::Runtime::Conversation::Conversation ALIAS runtime_conversation_conversation)

target_include_directories(runtime_conversation_conversation
  PUBLIC
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_conversation
  PUBLIC
    LiteRTLM::Runtime::Conversation::InternalCallbackUtil
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::Processor::Factory
    LiteRTLM::Runtime::Conversation::Processor::Interface

    LiteRTLM::Runtime::Components::PromptTemplate
    LiteRTLM::Runtime::Components::ConstrainedDecoding
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

add_litertlm_library(runtime_conversation_internal_callback_util STATIC
  internal_callback_util.cc
)
add_library(LiteRTLM::Runtime::Conversation::InternalCallbackUtil ALIAS runtime_conversation_internal_callback_util)

target_include_directories(runtime_conversation_internal_callback_util
  PUBLIC
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_internal_callback_util
  PUBLIC
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LITERTLM_DEPS
)

add_litertlm_library(runtime_conversation_io_types STATIC
  io_types.cc
)
add_library(LiteRTLM::Runtime::Conversation::IoTypes ALIAS runtime_conversation_io_types)

target_include_directories(runtime_conversation_io_types
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_io_types
  PUBLIC
    LITERTLM_DEPS
)


add_litertlm_library(runtime_conversation_prompt_utils STATIC
  prompt_utils.cc
)
add_library(LiteRTLM::Runtime::Conversation::PromptUtils ALIAS runtime_conversation_prompt_utils)

target_include_directories(runtime_conversation_prompt_utils
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_prompt_utils
  PUBLIC
    LITERTLM_DEPS
)


add_litertlm_library(runtime_conversation_channel_util STATIC
  channel_util.cc
)
add_library(LiteRTLM::Runtime::Conversation::ChannelUtil ALIAS runtime_conversation_channel_util)

target_include_directories(runtime_conversation_channel_util
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_channel_util
  PUBLIC
    LITERTLM_DEPS
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Engine::IoTypes
)




add_library(runtime_conversation_libs INTERFACE)
add_library(LiteRTLM::Runtime::Conversation ALIAS runtime_conversation_libs)

target_link_libraries(runtime_conversation_libs INTERFACE
  LiteRTLM::Runtime::Conversation::Conversation
  LiteRTLM::Runtime::Conversation::InternalCallbackUtil
  LiteRTLM::Runtime::Conversation::IoTypes
  LiteRTLM::Runtime::Conversation::PromptUtils
  LiteRTLM::Runtime::Conversation::Processor
)