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
# Config Registry
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_config_registry INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry ALIAS runtime_conversation_model_data_processor_config_registry)

target_include_directories(runtime_conversation_model_data_processor_config_registry
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Data Utils
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_data_utils STATIC
  data_utils.cc
)
add_library(LiteRTLM::Runtime::Conversation::Processor::DataUtils ALIAS runtime_conversation_model_data_processor_data_utils)

target_include_directories(runtime_conversation_model_data_processor_data_utils
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_data_utils
  PUBLIC
    runtime_util_memory_mapped_file
    LITERTLM_DEPS
)

# ==============================================================================
# Gemma3 Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma3_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor::Gemma3Config ALIAS runtime_conversation_model_data_processor_gemma3_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_gemma3_data_processor_config
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)


# ==============================================================================
# Gemma4 Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma4_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor::Gemma4Config ALIAS runtime_conversation_model_data_processor_gemma4_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_gemma4_data_processor_config
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)


add_litertlm_library(runtime_conversation_model_data_processor_function_gemma_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::FuncProcessor::GemmaConfig ALIAS runtime_conversation_model_data_processor_function_gemma_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_function_gemma_data_processor_config
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)


# ==============================================================================
# Generic Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_generic_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor::GenericConfig ALIAS runtime_conversation_model_data_processor_generic_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_generic_data_processor_config
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Qwen3 Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_qwen3_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor::Qwen3Config ALIAS runtime_conversation_model_data_processor_qwen3_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_qwen3_data_processor_config
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Model Data Processor Interface
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor::Interface ALIAS runtime_conversation_model_data_processor)

target_include_directories(runtime_conversation_model_data_processor
  INTERFACE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor
  INTERFACE
    LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    runtime_conversation_io_types
    runtime_engine_io_types
    LITERTLM_DEPS
)

# ==============================================================================
# Gemma3 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma3_data_processor STATIC
  gemma3_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::Processor::Gemma3 ALIAS runtime_conversation_model_data_processor_gemma3_data_processor)

target_include_directories(runtime_conversation_model_data_processor_gemma3_data_processor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_gemma3_data_processor
  PUBLIC
    runtime_conversation_io_types
    runtime_engine_io_types
    LiteRTLM::Runtime::Conversation::Processor::Gemma3Config
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
    LiteRTLM::Runtime::Components::Preprocessor::Audio
    LiteRTLM::Runtime::Components::Preprocessor::AudioMiniAudio
    LiteRTLM::Runtime::Components::Preprocessor::Image
    LiteRTLM::Runtime::Components::Preprocessor::StbImage
    LiteRTLM::Runtime::Components::ToolUse::ParserUtils
    LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils
    runtime_util_litert_status_util
    runtime_util_memory_mapped_file
    LiteRTLM::Runtime::Conversation::Processor::DataUtils
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LITERTLM_DEPS
)


add_litertlm_library(runtime_conversation_model_data_processor_function_gemma_data_processor STATIC
  function_gemma_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::FunctionDataProcessor ALIAS runtime_conversation_model_data_processor_function_gemma_data_processor)

target_include_directories(runtime_conversation_model_data_processor_function_gemma_data_processor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_function_gemma_data_processor
  PUBLIC
    runtime_conversation_io_types
    runtime_engine_io_types
    LiteRTLM::Runtime::Conversation::Processor::Gemma3Config
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
    LiteRTLM::Runtime::Components::Preprocessor::Audio
    LiteRTLM::Runtime::Components::Preprocessor::AudioMiniAudio
    LiteRTLM::Runtime::Components::Preprocessor::Image
    LiteRTLM::Runtime::Components::Preprocessor::StbImage
    LiteRTLM::Runtime::Components::ToolUse::ParserUtils
    LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils
    runtime_util_litert_status_util
    runtime_util_memory_mapped_file
    LiteRTLM::Runtime::Conversation::Processor::DataUtils
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LITERTLM_DEPS
)


# ==============================================================================
# Generic Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_generic_data_processor STATIC
  generic_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::Processor::Generic ALIAS runtime_conversation_model_data_processor_generic_data_processor)

target_include_directories(runtime_conversation_model_data_processor_generic_data_processor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_generic_data_processor
  PUBLIC
    LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::Processor::GenericConfig
    runtime_conversation_io_types
    runtime_engine_io_types
    runtime_util_litert_status_util
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# Qwen3 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_qwen3_data_processor STATIC
  qwen3_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::Processor::Qwen3 ALIAS runtime_conversation_model_data_processor_qwen3_data_processor)

target_include_directories(runtime_conversation_model_data_processor_qwen3_data_processor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_qwen3_data_processor
  PUBLIC
    LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LiteRTLM::Runtime::Conversation::Processor::Qwen3Config
    LiteRTLM::Runtime::Components::ToolUse::ParserUtils
    runtime_conversation_io_types
    runtime_engine_io_types
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# Model Data Processor Factory
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_factory STATIC
  model_data_processor_factory.cc
)
add_library(LiteRTLM::Runtime::Conversation::Processor::Factory ALIAS runtime_conversation_model_data_processor_factory)

target_include_directories(runtime_conversation_model_data_processor_factory
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_factory
  PUBLIC
    LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::Processor::Gemma3
    LiteRTLM::Runtime::Conversation::Processor::Gemma3Config
    LiteRTLM::Runtime::Conversation::Processor::Generic
    LiteRTLM::Runtime::Conversation::Processor::GenericConfig
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LiteRTLM::Runtime::Conversation::Processor::Qwen3
    LiteRTLM::Runtime::Conversation::Processor::Qwen3Config
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Conversation::IoTypes
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

# ==============================================================================
# Gemma3 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma4_data_processor STATIC
  gemma4_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::Processor::Gemma4 ALIAS runtime_conversation_model_data_processor_gemma4_data_processor)

target_include_directories(runtime_conversation_model_data_processor_gemma4_data_processor
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_gemma4_data_processor
  PUBLIC
    runtime_conversation_io_types
    runtime_engine_io_types
    LiteRTLM::Runtime::Conversation::Processor::Gemma4Config
    LiteRTLM::Runtime::Components::Tokenizer::Interface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
    LiteRTLM::Runtime::Components::Preprocessor::Audio
    LiteRTLM::Runtime::Components::Preprocessor::AudioMiniAudio
    LiteRTLM::Runtime::Components::Preprocessor::Image
    LiteRTLM::Runtime::Components::Preprocessor::StbImage
    LiteRTLM::Runtime::Components::ToolUse::ParserUtils
    LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils
    runtime_util_litert_status_util
    runtime_util_memory_mapped_file
    LiteRTLM::Runtime::Conversation::Processor::DataUtils
    LiteRTLM::Runtime::Conversation::Processor::Interface
    LITERTLM_DEPS
)



# ==============================================================================
# Folder Facade
# ==============================================================================
add_library(runtime_conversation_model_data_processor_libs INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::Processor ALIAS runtime_conversation_model_data_processor_libs)

target_link_libraries(runtime_conversation_model_data_processor_libs INTERFACE
  LiteRTLM::Runtime::Conversation::Processor::ConfigRegistry
  LiteRTLM::Runtime::Conversation::Processor::DataUtils
  LiteRTLM::Runtime::Conversation::Processor::Gemma3
  LiteRTLM::Runtime::Conversation::Processor::Gemma3Config
  
  LiteRTLM::Runtime::Conversation::FunctionDataProcessor

  LiteRTLM::Runtime::Conversation::Processor::Generic
  LiteRTLM::Runtime::Conversation::Processor::GenericConfig
  LiteRTLM::Runtime::Conversation::Processor::Interface
  LiteRTLM::Runtime::Conversation::Processor::Factory
  LiteRTLM::Runtime::Conversation::Processor::Qwen3
  LiteRTLM::Runtime::Conversation::Processor::Qwen3Config
)