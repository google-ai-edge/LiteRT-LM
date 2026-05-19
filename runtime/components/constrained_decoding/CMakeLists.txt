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

add_litertlm_library(runtime_components_constrained_decoding_bitmap INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
  ALIAS runtime_components_constrained_decoding_bitmap)

target_include_directories(runtime_components_constrained_decoding_bitmap
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

add_litertlm_library(runtime_components_constrained_decoding_constraint INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
  ALIAS runtime_components_constrained_decoding_constraint)

target_include_directories(runtime_components_constrained_decoding_constraint
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constraint
  INTERFACE
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LITERTLM_DEP
)

add_litertlm_library(runtime_components_constrained_decoding_constraint_provider INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
  ALIAS runtime_components_constrained_decoding_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_constraint_provider
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)


add_litertlm_library(runtime_components_constrained_decoding_constraint_provider_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProviderConfig ALIAS runtime_components_constrained_decoding_constraint_provider_config)

target_include_directories(runtime_components_constrained_decoding_constraint_provider_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

add_litertlm_library(runtime_components_constrained_decoding_constraint_config INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintConfig ALIAS runtime_components_constrained_decoding_constraint_config)

target_include_directories(runtime_components_constrained_decoding_constraint_config
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)

add_litertlm_library(runtime_components_constrained_decoding_fst_constraint INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::FstConstraint ALIAS runtime_components_constrained_decoding_fst_constraint)

target_include_directories(runtime_components_constrained_decoding_fst_constraint
  INTERFACE
    ${PKG_ROOT}
    ${LITERTLM_INCLUDE_PATHS}
)


add_litertlm_library(runtime_components_constrained_decoding_constrained_decoder STATIC
  constrained_decoder.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder ALIAS runtime_components_constrained_decoding_constrained_decoder)

target_include_directories(runtime_components_constrained_decoding_constrained_decoder
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constrained_decoder
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::LiteRtStatusUtil

    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_constraint_provider_factory STATIC
  constraint_provider_factory.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::Factory ALIAS runtime_components_constrained_decoding_constraint_provider_factory)

target_include_directories(runtime_components_constrained_decoding_constraint_provider_factory
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_constraint_provider_factory
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::LiteRtStatusUtil

    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_external_constraint_provider STATIC
  external_constraint_provider.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::ExternalProvider ALIAS runtime_components_constrained_decoding_external_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_external_constraint_provider
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_external_constraint_provider
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::LiteRtStatusUtil

    LITERTLM_DEPS
)



add_litertlm_library(runtime_components_constrained_decoding_fake_constraint STATIC
  fake_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::FakeConstraint ALIAS runtime_components_constrained_decoding_fake_constraint)

target_include_directories(runtime_components_constrained_decoding_fake_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_fake_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_llg_constraint STATIC
  llg_constraint.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraint ALIAS runtime_components_constrained_decoding_llg_constraint)

target_include_directories(runtime_components_constrained_decoding_llg_constraint
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_constraint
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_llg_constraint_provider STATIC
  llg_constraint_provider.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraintProvider ALIAS runtime_components_constrained_decoding_llg_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_llg_constraint_provider
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llg_constraint_provider
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_llguidance_schema_utils STATIC
  llguidance_schema_utils.cc
  llg_fc_tool_calls.cc
  llg_python_tool_calls.cc
  llg_tool_call_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::LlguidanceSchemaUtils
  ALIAS runtime_components_constrained_decoding_llguidance_schema_utils)

target_include_directories(runtime_components_constrained_decoding_llguidance_schema_utils
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_llguidance_schema_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_constrained_decoding_gemma_model_constraint_provider STATIC
  gemma_model_constraint_provider.cc
)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding::GemmaModelConstraintProvider
  ALIAS runtime_components_constrained_decoding_gemma_model_constraint_provider)

target_include_directories(runtime_components_constrained_decoding_gemma_model_constraint_provider
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${LITERT_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_constrained_decoding_gemma_model_constraint_provider
  PUBLIC
    LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
    LiteRTLM::Runtime::Util::ConvertTensorBuffer
    LiteRTLM::Runtime::Util::LiteRtStatusUtil

    LITERTLM_DEPS
)


add_library(runtime_components_constrained_decoding_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::ConstrainedDecoding
  ALIAS runtime_components_constrained_decoding_libs)

target_link_libraries(runtime_components_constrained_decoding_libs INTERFACE
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Bitmap
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Constraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintProviderConfig
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ConstraintConfig
  LiteRTLM::Runtime::Components::ConstrainedDecoding::FstConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::GemmaModelConstraintProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Decoder
  LiteRTLM::Runtime::Components::ConstrainedDecoding::ExternalProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::Factory
  LiteRTLM::Runtime::Components::ConstrainedDecoding::FakeConstraint
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlgConstraintProvider
  LiteRTLM::Runtime::Components::ConstrainedDecoding::LlguidanceSchemaUtils
)
