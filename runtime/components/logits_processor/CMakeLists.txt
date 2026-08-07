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

add_subdirectory(constrained_decoding)

# ==============================================================================
# Logits Processor
# ==============================================================================
add_litertlm_library(runtime_components_logits_processor INTERFACE)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::LogitsProcessorInterface
  ALIAS runtime_components_logits_processor)

target_include_directories(runtime_components_logits_processor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Mock Logits Processor
# ==============================================================================
add_litertlm_library(runtime_components_logits_mock_processor INTERFACE)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::MockLogitsProcessorInterface
  ALIAS runtime_components_logits_mock_processor)

target_include_directories(runtime_components_logits_mock_processor
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# No Repeat NGram Config
# ==============================================================================
add_litertlm_library(runtime_components_logits_no_repeat_ngram_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::NoRepeatNGramConfigInterface
  ALIAS runtime_components_logits_no_repeat_ngram_config)

target_include_directories(runtime_components_logits_no_repeat_ngram_config
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Repetition Penalty Config
# ==============================================================================
add_litertlm_library(runtime_components_logits_repetition_penalty_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::RepetitionPenaltyConfigInterface
  ALIAS runtime_components_logits_repetition_penalty_config)

target_include_directories(runtime_components_logits_repetition_penalty_config
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Suppress Tokens Config
# ==============================================================================
add_litertlm_library(runtime_components_logits_suppress_tokens_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::SuppressTokensConfigInterface
  ALIAS runtime_components_logits_suppress_tokens_config)

target_include_directories(runtime_components_logits_suppress_tokens_config
  INTERFACE
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# No Repeat NGram Processor
# ==============================================================================
add_litertlm_library(runtime_components_logits_no_repeat_ngram_processor STATIC
  no_repeat_ngram_processor.cc
)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::NoRepeatNGramProcessor
  ALIAS runtime_components_logits_no_repeat_ngram_processor)

target_include_directories(runtime_components_logits_no_repeat_ngram_processor
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)
target_link_libraries(runtime_components_logits_no_repeat_ngram_processor
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Repetition Penalty Processor
# ==============================================================================
add_litertlm_library(runtime_components_logits_repetition_penalty_processor STATIC
  repetition_penalty_processor.cc
)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::RepetitionPenaltyProcessor
  ALIAS runtime_components_logits_repetition_penalty_processor)

target_include_directories(runtime_components_logits_repetition_penalty_processor
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)
target_link_libraries(runtime_components_logits_repetition_penalty_processor
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Suppress Tokens Processor
# ==============================================================================
add_litertlm_library(runtime_components_logits_suppress_tokens_processor STATIC
  suppress_tokens_processor.cc
)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::SuppressTokensProcessor
  ALIAS runtime_components_logits_suppress_tokens_processor)

target_include_directories(runtime_components_logits_suppress_tokens_processor
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)
target_link_libraries(runtime_components_logits_suppress_tokens_processor
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(runtime_components_logits_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor ALIAS runtime_components_logits_libs)

target_link_libraries(runtime_components_logits_libs INTERFACE
  LiteRTLM::Runtime::Components::ConstrainedDecoding
  LiteRTLM::Runtime::Components::LogitsProcessor::MockLogitsProcessorInterface
  LiteRTLM::Runtime::Components::LogitsProcessor::NoRepeatNGramConfigInterface
  LiteRTLM::Runtime::Components::LogitsProcessor::RepetitionPenaltyConfigInterface
  LiteRTLM::Runtime::Components::LogitsProcessor::NoRepeatNGramProcessor
  LiteRTLM::Runtime::Components::LogitsProcessor::RepetitionPenaltyProcessor
  LiteRTLM::Runtime::Components::LogitsProcessor::SuppressTokensProcessor

)