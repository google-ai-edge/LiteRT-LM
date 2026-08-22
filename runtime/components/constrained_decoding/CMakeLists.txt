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
# Bitmap Interface
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_bitmap INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
  ALIAS runtime_components_constrained_decoding_bitmap)

target_include_directories(runtime_components_constrained_decoding_bitmap
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)


# ==============================================================================
# LogitMask Interface & Implementation
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_logit_mask STATIC
  logit_mask.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
  ALIAS runtime_components_constrained_decoding_logit_mask)

target_include_directories(runtime_components_constrained_decoding_logit_mask
  PUBLIC
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_logit_mask
  PUBLIC
    LITERTLM_DEPS
)

# ==============================================================================
# Constraint Interface
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_constraint STATIC
  constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  ALIAS runtime_components_constrained_decoding_constraint)

target_include_directories(runtime_components_constrained_decoding_constraint
  PUBLIC
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
    LITERTLM_DEPS
)

# ==============================================================================
# Constraint Provider Interface
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_constraint_provider INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
  ALIAS runtime_components_constrained_decoding_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_constraint_provider
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Constraint Provider Config Interface
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_constraint_provider_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProviderConfig
  ALIAS runtime_components_constrained_decoding_constraint_provider_config)

target_include_directories(runtime_components_constrained_decoding_constraint_provider_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Constraint Config Interface
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_constraint_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintConfig
  ALIAS runtime_components_constrained_decoding_constraint_config)

target_include_directories(runtime_components_constrained_decoding_constraint_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# FST Constraint Interface
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_fst_constraint INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::FstConstraint
  ALIAS runtime_components_constrained_decoding_fst_constraint)

target_include_directories(runtime_components_constrained_decoding_fst_constraint
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Constraint Decoder
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_constrained_decoder STATIC
  constrained_decoder.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
  ALIAS runtime_components_constrained_decoding_constrained_decoder)

target_include_directories(runtime_components_constrained_decoding_constrained_decoder
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constrained_decoder
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Constraint Provider Factory
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_constraint_provider_factory STATIC
  constraint_provider_factory.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Factory
  ALIAS runtime_components_constrained_decoding_constraint_provider_factory)

target_include_directories(runtime_components_constrained_decoding_constraint_provider_factory
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constraint_provider_factory
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LITERTLM_DEPS
)

# ==============================================================================
# External Constraint Provider
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_external_constraint_provider STATIC
  external_constraint_provider.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ExternalProvider
  ALIAS runtime_components_constrained_decoding_external_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_external_constraint_provider
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_external_constraint_provider
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Fake Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_fake_constraint STATIC
  fake_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::FakeConstraint 
  ALIAS runtime_components_constrained_decoding_fake_constraint)

target_include_directories(runtime_components_constrained_decoding_fake_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_fake_constraint
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  )

# ==============================================================================
# LLG Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_llg_constraint STATIC
  llg_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraint 
  ALIAS runtime_components_constrained_decoding_llg_constraint)

target_include_directories(runtime_components_constrained_decoding_llg_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_constraint
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
)

# ==============================================================================
# LLG Provider
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_llg_constraint_provider STATIC
  llg_constraint_provider.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraintProvider 
  ALIAS runtime_components_constrained_decoding_llg_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_llg_constraint_provider
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_constraint_provider
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraint
)

# ==============================================================================
# LLG FC Tool Calls
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_llg_fc_tool_calls STATIC
  llg_fc_tool_calls.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgFcToolCalls
  ALIAS runtime_components_constrained_decoding_llg_fc_tool_calls)

target_include_directories(runtime_components_constrained_decoding_llg_fc_tool_calls
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_fc_tool_calls
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  )

# ==============================================================================
# LLG Python Tool Calls
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_llg_python_tool_calls STATIC
  llg_python_tool_calls.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgPythonToolCalls
  ALIAS runtime_components_constrained_decoding_llg_python_tool_calls)

target_include_directories(runtime_components_constrained_decoding_llg_python_tool_calls
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_python_tool_calls
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  )

# ==============================================================================
# LLG Tool Call Utils
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_llg_tool_call_utils STATIC
  llg_tool_call_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgToolCallUtils
  ALIAS runtime_components_constrained_decoding_llg_tool_call_utils)

target_include_directories(runtime_components_constrained_decoding_llg_tool_call_utils
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_tool_call_utils
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  )

# ==============================================================================
# LLGuidance Schema Utils
# ==============================================================================
set(_llguidance_schema_utils_src
    "llg_fc_tool_calls.cc"
    "llg_python_tool_calls.cc"
    "llg_tool_call_utils.cc"
    "llguidance_schema_utils.cc"
)
add_litertlm_library(runtime_components_constrained_decoding_llguidance_schema_utils STATIC
  ${_llguidance_schema_utils_src}
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlguidanceSchemaUtils
  ALIAS runtime_components_constrained_decoding_llguidance_schema_utils)

target_include_directories(runtime_components_constrained_decoding_llguidance_schema_utils
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llguidance_schema_utils
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  )


add_litertlm_library(runtime_components_constrained_decoding_gemma_model_constraint_provider STATIC
  gemma_model_constraint_provider.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::GemmaModelConstraintProvider
  ALIAS runtime_components_constrained_decoding_gemma_model_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_gemma_model_constraint_provider
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERTLM_LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_gemma_model_constraint_provider
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Thinking Budget Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_thinking_budget_constraint STATIC
  thinking_budget_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ThinkingBudgetConstraint ALIAS runtime_components_constrained_decoding_thinking_budget_constraint)

target_include_directories(runtime_components_constrained_decoding_thinking_budget_constraint
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_components_constrained_decoding_thinking_budget_constraint
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LITERTLM_DEPS
)

# ==============================================================================
# Config Interfaces
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_no_repeat_ngram_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::NoRepeatNGramConfigInterface
  ALIAS runtime_components_constrained_decoding_no_repeat_ngram_config)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::NoRepeatNGramConfigInterface
  ALIAS runtime_components_constrained_decoding_no_repeat_ngram_config)

target_include_directories(runtime_components_constrained_decoding_no_repeat_ngram_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

add_litertlm_library(runtime_components_constrained_decoding_repetition_penalty_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::RepetitionPenaltyConfigInterface
  ALIAS runtime_components_constrained_decoding_repetition_penalty_config)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::RepetitionPenaltyConfigInterface
  ALIAS runtime_components_constrained_decoding_repetition_penalty_config)

target_include_directories(runtime_components_constrained_decoding_repetition_penalty_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

add_litertlm_library(runtime_components_constrained_decoding_suppress_tokens_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::SuppressTokensConfigInterface
  ALIAS runtime_components_constrained_decoding_suppress_tokens_config)
add_library(LiteRTLM::Runtime::Components::LogitsProcessor::SuppressTokensConfigInterface
  ALIAS runtime_components_constrained_decoding_suppress_tokens_config)

target_include_directories(runtime_components_constrained_decoding_suppress_tokens_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

# ==============================================================================
# Suppress Tokens Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_suppress_tokens_constraint STATIC
  suppress_tokens_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::SuppressTokensConstraint
  ALIAS runtime_components_constrained_decoding_suppress_tokens_constraint)

target_include_directories(runtime_components_constrained_decoding_suppress_tokens_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_suppress_tokens_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
    LiteRTLM::Runtime::Components::ConstrainedDecoding::SuppressTokensConfigInterface
    LITERTLM_DEPS
)

# ==============================================================================
# No Repeat Ngram Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_no_repeat_ngram_constraint STATIC
  no_repeat_ngram_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::NoRepeatNgramConstraint
  ALIAS runtime_components_constrained_decoding_no_repeat_ngram_constraint)

target_include_directories(runtime_components_constrained_decoding_no_repeat_ngram_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_no_repeat_ngram_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
    LiteRTLM::Runtime::Components::ConstrainedDecoding::NoRepeatNGramConfigInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Repetition Penalty Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_repetition_penalty_constraint STATIC
  repetition_penalty_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::RepetitionPenaltyConstraint
  ALIAS runtime_components_constrained_decoding_repetition_penalty_constraint)

target_include_directories(runtime_components_constrained_decoding_repetition_penalty_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_repetition_penalty_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
    LiteRTLM::Runtime::Components::ConstrainedDecoding::RepetitionPenaltyConfigInterface
    LITERTLM_DEPS
)

# ==============================================================================
# Composite Constraint
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_composite_constraint STATIC
  composite_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::CompositeConstraint ALIAS runtime_components_constrained_decoding_composite_constraint)

target_include_directories(runtime_components_constrained_decoding_composite_constraint
  PRIVATE
    "${LITERTLM_INCLUDE_PATHS}"
)

target_link_libraries(runtime_components_constrained_decoding_composite_constraint
  PUBLIC
    LiteRTLM::Runtime::Util
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
    LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
    LITERTLM_DEPS
)

# ==============================================================================
# Folder Facade
# ==============================================================================
add_litertlm_library(runtime_components_constrained_decoding_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding
  ALIAS runtime_components_constrained_decoding_libs)

target_link_libraries(runtime_components_constrained_decoding_libs INTERFACE
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LogitMask
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintInterface
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintConfig
  LiteRTLM::Runtime::Components::ConstrainedDecoding::FstConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::GemmaModelConstraintProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ExternalProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Factory
  LiteRTLM::Runtime::Components::ConstrainedDecoding::FakeConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraintProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgFcToolCalls
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgPythonToolCalls
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgToolCallUtils
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlguidanceSchemaUtils
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ThinkingBudgetConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::CompositeConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::SuppressTokensConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::NoRepeatNgramConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::RepetitionPenaltyConstraint
)


