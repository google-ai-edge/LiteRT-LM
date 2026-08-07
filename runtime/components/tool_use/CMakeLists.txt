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

add_subdirectory(antlr)

# ==============================================================================
# FC Parser Utils
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_fc_parser_utils STATIC
  fc_parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::FcParserUtils
  ALIAS runtime_components_tool_use_fc_parser_utils)

target_include_directories(runtime_components_tool_use_fc_parser_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_fc_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::ParserCommon
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Fc
    LITERTLM_DEPS
)

# ==============================================================================
# FC Tool Format Utils
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_fc_tool_format_utils STATIC
  fc_tool_format_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::FcToolFormatUtils
  ALIAS runtime_components_tool_use_fc_tool_format_utils)

target_include_directories(runtime_components_tool_use_fc_tool_format_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_fc_tool_format_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)

# ==============================================================================
# Json Parser Utils
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_json_parser_utils STATIC
  json_parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::JsonParserUtils
  ALIAS runtime_components_tool_use_json_parser_utils)

target_include_directories(runtime_components_tool_use_json_parser_utils
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
)

target_link_libraries(runtime_components_tool_use_json_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::ParserCommon
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)

# ==============================================================================
# Parser Common
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_parser_common STATIC
  parser_common.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::ParserCommon
  ALIAS runtime_components_tool_use_parser_common)

target_include_directories(runtime_components_tool_use_parser_common
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_GENERATED_SRC_DIR}
)

target_link_libraries(runtime_components_tool_use_parser_common
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)

# ==============================================================================
# Parser Utils
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_parser_utils STATIC
  parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::ParserUtils
  ALIAS runtime_components_tool_use_parser_utils)

target_include_directories(runtime_components_tool_use_parser_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Fc

    LiteRTLM::Runtime::Components::ToolUse::JsonParserUtils
    LiteRTLM::Runtime::Components::ToolUse::PythonParserUtils
    LiteRTLM::Runtime::Components::ToolUse::FcParserUtils
    LiteRTLM::Runtime::Components::ToolUse::FcToolFormatUtils

    LITERTLM_DEPS
)

# ==============================================================================
# Python Parser Utils
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_python_parser_utils STATIC
  python_parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::PythonParserUtils
  ALIAS runtime_components_tool_use_python_parser_utils)

target_include_directories(runtime_components_tool_use_python_parser_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_python_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::ParserCommon
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
    LITERTLM_DEPS
)

# ==============================================================================
# Python Tool Format Utils
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_python_tool_format_utils STATIC
  python_tool_format_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils
  ALIAS runtime_components_tool_use_python_tool_format_utils)

target_include_directories(runtime_components_tool_use_python_tool_format_utils
  PRIVATE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_python_tool_format_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(runtime_components_tool_use_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::ToolUse
  ALIAS runtime_components_tool_use_libs)

target_link_libraries(runtime_components_tool_use_libs INTERFACE
  LiteRTLM::Runtime::Components::ToolUse::Antlr
  LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
  LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
  LiteRTLM::Runtime::Components::ToolUse::FcParserUtils
  LiteRTLM::Runtime::Components::ToolUse::FcToolFormatUtils
  LiteRTLM::Runtime::Components::ToolUse::JsonParserUtils
  LiteRTLM::Runtime::Components::ToolUse::ParserCommon
  LiteRTLM::Runtime::Components::ToolUse::ParserUtils
  LiteRTLM::Runtime::Components::ToolUse::PythonParserUtils
  LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils

)