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
# Config Registry
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_config_registry INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::ConfigRegistry
  ALIAS runtime_conversation_model_data_processor_config_registry)

target_include_directories(runtime_conversation_model_data_processor_config_registry
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Data Utils
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_data_utils STATIC
  data_utils.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
  ALIAS runtime_conversation_model_data_processor_data_utils)

target_include_directories(runtime_conversation_model_data_processor_data_utils
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_data_utils
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Gemma3 Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma3_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3Config
  ALIAS runtime_conversation_model_data_processor_gemma3_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_gemma3_data_processor_config
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)


add_litertlm_library(runtime_conversation_model_data_processor_function_gemma_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::FuncProcessor::GemmaConfig
  ALIAS runtime_conversation_model_data_processor_function_gemma_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_function_gemma_data_processor_config
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Generic Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_generic_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::GenericConfig
  ALIAS runtime_conversation_model_data_processor_generic_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_generic_data_processor_config
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Qwen3 Data Processor Config
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_qwen3_data_processor_config INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3Config
  ALIAS runtime_conversation_model_data_processor_qwen3_data_processor_config)

target_include_directories(runtime_conversation_model_data_processor_qwen3_data_processor_config
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Model Data Processor Interface
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
  ALIAS runtime_conversation_model_data_processor)

target_include_directories(runtime_conversation_model_data_processor
  INTERFACE
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor
  INTERFACE
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::ConfigRegistry
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Engine::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Gemma3 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma3_data_processor STATIC
  gemma3_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3DataProcessor
  ALIAS runtime_conversation_model_data_processor_gemma3_data_processor)

target_include_directories(runtime_conversation_model_data_processor_gemma3_data_processor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_gemma3_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3Config
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::MultimodalProcessorHelper
    LITERTLM_DEPS
)

# ==============================================================================
# Gemma4 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_gemma4_data_processor STATIC
  gemma4_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma4DataProcessor 
  ALIAS runtime_conversation_model_data_processor_gemma4_data_processor)

target_include_directories(runtime_conversation_model_data_processor_gemma4_data_processor
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_conversation_model_data_processor_gemma4_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::MultimodalProcessorHelper
    LiteRTLM::Runtime::Conversation::PromptUtils
    LITERTLM_DEPS
)

# ==============================================================================
# Fastvlm Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_fastvlm_data_processor STATIC
  fastvlm_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::FastvlmDataProcessor ALIAS 
    runtime_conversation_model_data_processor_fastvlm_data_processor)

target_include_directories(runtime_conversation_model_data_processor_fastvlm_data_processor
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_conversation_model_data_processor_fastvlm_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::MultimodalProcessorHelper
    LITERTLM_DEPS
)

# ==============================================================================
# Function Gemma Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_function_gemma_data_processor STATIC
  function_gemma_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::FunctionDataProcessor
ALIAS runtime_conversation_model_data_processor_function_gemma_data_processor)

target_include_directories(runtime_conversation_model_data_processor_function_gemma_data_processor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_function_gemma_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3Config
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# Generic Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_generic_data_processor STATIC
  generic_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Generic ALIAS runtime_conversation_model_data_processor_generic_data_processor)

target_include_directories(runtime_conversation_model_data_processor_generic_data_processor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_generic_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::GenericConfig
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::MultimodalProcessorHelper
    LiteRTLM::Runtime::Conversation::IoTypes
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
    LITERTLM_DEPS
)

# ==============================================================================
# Qwen3 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_qwen3_data_processor STATIC
  qwen3_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3 ALIAS runtime_conversation_model_data_processor_qwen3_data_processor)

target_include_directories(runtime_conversation_model_data_processor_qwen3_data_processor
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_qwen3_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3Config
    LiteRTLM::Runtime::Conversation::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Model Data Processor Factory
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_factory STATIC
  model_data_processor_factory.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Factory ALIAS runtime_conversation_model_data_processor_factory)

target_include_directories(runtime_conversation_model_data_processor_factory
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_conversation_model_data_processor_factory
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::ConfigRegistry
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3DataProcessor
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3Config
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Generic
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::GenericConfig
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3Config
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma4DataProcessor
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::Lfm2DataProcessor
    LiteRTLM::Runtime::Conversation::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Multimodal Processor Helper
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_multimodal_processor_helper STATIC
  multimodal_processor_helper.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::MultimodalProcessorHelper
  ALIAS runtime_conversation_model_data_processor_multimodal_processor_helper)

target_include_directories(runtime_conversation_model_data_processor_multimodal_processor_helper
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_conversation_model_data_processor_multimodal_processor_helper
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
    LiteRTLM::Runtime::Conversation::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Minicpm5 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_minicpm5_data_processor STATIC
  minicpm5_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Minicpm5DataProcessor
    ALIAS runtime_conversation_model_data_processor_minicpm5_data_processor)

target_include_directories(runtime_conversation_model_data_processor_minicpm5_data_processor
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_conversation_model_data_processor_minicpm5_data_processor
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# LFM2 Data Processor
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_lfm2_data_processor STATIC
  lfm2_data_processor.cc
)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor::Lfm2DataProcessor
    ALIAS runtime_conversation_model_data_processor_lfm2_data_processor)

target_include_directories(runtime_conversation_model_data_processor_lfm2_data_processor
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_conversation_model_data_processor_lfm2_data_processor
  PUBLIC
    LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Components
    LiteRTLM::Runtime::Conversation::IoTypes
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(runtime_conversation_model_data_processor_libs INTERFACE)
add_library(LiteRTLM::Runtime::Conversation::ModelDataProcessor
  ALIAS runtime_conversation_model_data_processor_libs)

target_link_libraries(runtime_conversation_model_data_processor_libs INTERFACE
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::ConfigRegistry
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::DataUtils
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3DataProcessor
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma3Config
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Gemma4DataProcessor
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::FastvlmDataProcessor
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Generic
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::GenericConfig
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Interface
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Factory
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Qwen3Config
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::MultimodalProcessorHelper
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Minicpm5DataProcessor
  LiteRTLM::Runtime::Conversation::ModelDataProcessor::Lfm2DataProcessor
)