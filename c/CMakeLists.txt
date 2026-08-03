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

# ==============================================================================
# 1. Engine Library
#    Bazel: cc_library(name = "engine", ...)
# ==============================================================================
add_litertlm_library(c_engine STATIC
  engine.cc
)
add_library(LiteRTLM::C::Engine ALIAS c_engine)

target_include_directories(c_engine
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}        # Project Root & Generated Headers
    ${LITERT_INCLUDE_PATHS}          # LiteRT External Headers
    ${THIRD_PARTY_DIR}/json/include # @nlohmann_json//:json
)

target_link_libraries(c_engine
  PUBLIC
    LITERTLM_DEPS

    LiteRTLM::Runtime::Conversation
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Core::EngineImpl
    LiteRTLM::Runtime::Engine::Interface
    LiteRTLM::Runtime::Engine::Settings
    LiteRTLM::Runtime::Engine::IoTypes
    LiteRTLM::Runtime::Executor::ExecutorSettingsBase
)

# ==============================================================================
# 2. Folder Facade
# ==============================================================================
add_library(c_libs INTERFACE)
add_library(LiteRTLM::C ALIAS c_libs)

target_link_libraries(c_libs INTERFACE
  LiteRTLM::C::Engine
)