# Copyright 2025 The ODML Authors.
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

load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "@bazel_skylib//rules:common_settings.bzl",
    "bool_flag",
    "string_flag",
    "string_list_flag",
)

package(
    default_visibility = [
        "//:__subpackages__",
    ],
)

config_setting(
    name = "android_arm64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "android_x86_64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "ios_arm64",
    values = { "apple_platform_type": "ios", "cpu": "ios_arm64",},
)

config_setting(
    name = "ios_sim_arm64",
    values = { "apple_platform_type": "ios", "cpu": "ios_sim_arm64",},
)

alias(
    name = "patchelf_exe",
    actual = select({
        ":linux_x86_64": "@patchelf_linux_x86_64//:patchelf",
        ":linux_arm64": "@patchelf_linux_arm64//:patchelf",
        "//conditions:default": "@patchelf_linux_x86_64//:patchelf",
    }),
    visibility = ["//visibility:public"],
)

# ==============================================================================
# Centralized Build-Time Feature Flags
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Tokenizers & Prompt Templating
# ------------------------------------------------------------------------------

bool_flag(
    name = "enable_sentencepiece_tokenizer",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "sentencepiece_tokenizer_disabled_by_flag",
    flag_values = {":enable_sentencepiece_tokenizer": "false"},
)

config_setting(
    name = "sentencepiece_tokenizer_disabled_by_define",
    define_values = {"DISABLE_SENTENCEPIECE_TOKENIZER": "1"},
)

selects.config_setting_group(
    name = "disable_sentencepiece_tokenizer",
    match_any = [
        ":sentencepiece_tokenizer_disabled_by_flag",
        ":sentencepiece_tokenizer_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_huggingface_tokenizer",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "huggingface_tokenizer_disabled_by_flag",
    flag_values = {":enable_huggingface_tokenizer": "false"},
)

config_setting(
    name = "huggingface_tokenizer_disabled_by_define",
    define_values = {"DISABLE_HUGGINGFACE_TOKENIZER": "1"},
)

selects.config_setting_group(
    name = "disable_huggingface_tokenizer",
    match_any = [
        ":huggingface_tokenizer_disabled_by_flag",
        ":huggingface_tokenizer_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_minijinja_template",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "minijinja_template_disabled_by_flag",
    flag_values = {":enable_minijinja_template": "false"},
)

config_setting(
    name = "minijinja_template_disabled_by_define",
    define_values = {"LITERT_LM_MINIJINJA_TEMPLATE_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_minijinja_template",
    match_any = [
        ":minijinja_template_disabled_by_flag",
        ":minijinja_template_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

# ------------------------------------------------------------------------------
# 2. Constrained Decoding & Tool Use Parsing
# ------------------------------------------------------------------------------

bool_flag(
    name = "enable_constrained_decoding",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "constrained_decoding_disabled_by_flag",
    flag_values = {":enable_constrained_decoding": "false"},
)

config_setting(
    name = "constrained_decoding_disabled_by_define",
    define_values = {"LITERT_LM_CONSTRAINED_DECODING_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_constrained_decoding",
    match_any = [
        ":constrained_decoding_disabled_by_flag",
        ":constrained_decoding_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_llguidance",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "llguidance_disabled_by_flag",
    flag_values = {":enable_llguidance": "false"},
)

config_setting(
    name = "llguidance_disabled_by_define",
    define_values = {"LITERT_LM_LLGUIDANCE_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_llguidance",
    match_any = [
        ":disable_constrained_decoding",
        ":llguidance_disabled_by_flag",
        ":llguidance_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_fst_constraints",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fst_constraints_disabled_by_flag",
    flag_values = {":enable_fst_constraints": "false"},
)

config_setting(
    name = "fst_constraints_disabled_by_define",
    define_values = {"LITERT_LM_FST_CONSTRAINTS_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_fst_constraints",
    match_any = [
        ":disable_constrained_decoding",
        ":fst_constraints_disabled_by_flag",
        ":fst_constraints_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_tool_use_parser",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "tool_use_parser_disabled_by_flag",
    flag_values = {":enable_tool_use_parser": "false"},
)

config_setting(
    name = "tool_use_parser_disabled_by_define",
    define_values = {"LITERT_LM_TOOL_USE_PARSER_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_tool_use_parser",
    match_any = [
        ":tool_use_parser_disabled_by_flag",
        ":tool_use_parser_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

# ------------------------------------------------------------------------------
# 3. Hardware Backends & Legacy/Experimental Executors
# ------------------------------------------------------------------------------

config_setting(
    name = "litert_build_include_cpu_only",
    flag_values = {
        "@litert//litert/build_common:build_include": "cpu_only",
    },
)

bool_flag(
    name = "enable_gpu_backend",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "gpu_backend_disabled_by_flag",
    flag_values = {":enable_gpu_backend": "false"},
)

config_setting(
    name = "gpu_backend_disabled_by_define",
    define_values = {"LITERT_LM_GPU_BACKEND_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_gpu_backend",
    match_any = [
        ":gpu_backend_disabled_by_flag",
        ":gpu_backend_disabled_by_define",
        ":litert_build_include_cpu_only",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_npu_backend",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "npu_backend_disabled_by_flag",
    flag_values = {":enable_npu_backend": "false"},
)

config_setting(
    name = "npu_backend_disabled_by_define",
    define_values = {"LITERT_LM_NPU_BACKEND_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_npu_backend",
    match_any = [
        ":npu_backend_disabled_by_flag",
        ":npu_backend_disabled_by_define",
        "@platforms//os:ios",
        ":litert_build_include_cpu_only",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_legacy_artisan_executors",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "legacy_artisan_executors_disabled_by_flag",
    flag_values = {":enable_legacy_artisan_executors": "false"},
)

config_setting(
    name = "legacy_artisan_executors_disabled_by_define",
    define_values = {"LITERT_LM_EXCLUDE_LEGACY_EXECUTORS": "1"},
)

selects.config_setting_group(
    name = "disable_legacy_artisan_executors",
    match_any = [
        ":legacy_artisan_executors_disabled_by_flag",
        ":legacy_artisan_executors_disabled_by_define",
        ":litert_build_include_cpu_only",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "disable_engine_advanced",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "engine_advanced_impl_disabled_by_flag",
    flag_values = {":disable_engine_advanced": "true"},
)

selects.config_setting_group(
    name = "disable_engine_advanced_impl",
    match_any = [
        ":engine_advanced_impl_disabled_by_flag",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_diffusion_executor",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "diffusion_executor_enabled_by_flag",
    flag_values = {":enable_diffusion_executor": "true"},
)

config_setting(
    name = "diffusion_executor_enabled_by_define",
    define_values = {"LITERT_LM_SATURN_ENABLED": "1"},
)

selects.config_setting_group(
    name = "enable_saturn",
    match_any = [
        ":diffusion_executor_enabled_by_flag",
        ":diffusion_executor_enabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_google_tensor_backend",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "google_tensor_enabled_by_flag",
    constraint_values = ["@platforms//os:android"],
    flag_values = {":enable_google_tensor_backend": "true"},
)

config_setting(
    name = "google_tensor_enabled_by_define",
    constraint_values = ["@platforms//os:android"],
    define_values = {"GOOGLE_TENSOR_ENABLED": "1"},
)

selects.config_setting_group(
    name = "enable_google_tensor",
    match_any = [
        ":google_tensor_enabled_by_flag",
        ":google_tensor_enabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

# ------------------------------------------------------------------------------
# 4. Modalities & Observability
# ------------------------------------------------------------------------------

bool_flag(
    name = "enable_vision",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "vision_disabled_by_flag",
    flag_values = {":enable_vision": "false"},
)

config_setting(
    name = "vision_disabled_by_define",
    define_values = {"LITERT_LM_VISION_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_vision",
    match_any = [
        ":vision_disabled_by_flag",
        ":vision_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_audio",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "audio_disabled_by_flag",
    flag_values = {":enable_audio": "false"},
)

config_setting(
    name = "audio_disabled_by_define",
    define_values = {"LITERT_LM_AUDIO_DISABLED": "1"},
)

selects.config_setting_group(
    name = "disable_audio",
    match_any = [
        ":audio_disabled_by_flag",
        ":audio_disabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_debugger",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debugger_enabled_by_flag",
    flag_values = {":enable_debugger": "true"},
)

config_setting(
    name = "debugger_enabled_by_define",
    define_values = {"LITERT_LM_DEBUGGER_ENABLED": "1"},
)

selects.config_setting_group(
    name = "enable_debugger_setting",
    match_any = [
        ":debugger_enabled_by_flag",
        ":debugger_enabled_by_define",
    ],
    visibility = ["//visibility:public"],
)

# ------------------------------------------------------------------------------
# 5. Model Family Data Processors
# ------------------------------------------------------------------------------

bool_flag(
    name = "enable_all_model_processors",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_all_model_processors_true",
    flag_values = {":enable_all_model_processors": "true"},
)

string_list_flag(
    name = "enabled_model_families",
    build_setting_default = ["all"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "model_families_all",
    flag_values = {":enabled_model_families": "all"},
)

selects.config_setting_group(
    name = "all_model_processors_active",
    match_all = [
        ":enable_all_model_processors_true",
        ":model_families_all",
    ],
)

string_flag(
    name = "enable_gemma_processors",
    build_setting_default = "auto",
    values = [
        "auto",
        "true",
        "false",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_gemma_processors_true",
    flag_values = {":enable_gemma_processors": "true"},
)

config_setting(
    name = "enable_gemma_processors_auto",
    flag_values = {":enable_gemma_processors": "auto"},
)

config_setting(
    name = "model_family_gemma_in_list",
    flag_values = {":enabled_model_families": "gemma"},
)

selects.config_setting_group(
    name = "gemma_processors_auto_all",
    match_all = [
        ":enable_gemma_processors_auto",
        ":all_model_processors_active",
    ],
)

selects.config_setting_group(
    name = "gemma_processors_auto_list",
    match_all = [
        ":enable_gemma_processors_auto",
        ":model_family_gemma_in_list",
    ],
)

selects.config_setting_group(
    name = "enable_gemma_processors_setting",
    match_any = [
        ":enable_gemma_processors_true",
        ":gemma_processors_auto_all",
        ":gemma_processors_auto_list",
    ],
    visibility = ["//visibility:public"],
)

string_flag(
    name = "enable_qwen_processor",
    build_setting_default = "auto",
    values = [
        "auto",
        "true",
        "false",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_qwen_processor_true",
    flag_values = {":enable_qwen_processor": "true"},
)

config_setting(
    name = "enable_qwen_processor_auto",
    flag_values = {":enable_qwen_processor": "auto"},
)

config_setting(
    name = "model_family_qwen_in_list",
    flag_values = {":enabled_model_families": "qwen"},
)

selects.config_setting_group(
    name = "qwen_processor_auto_all",
    match_all = [
        ":enable_qwen_processor_auto",
        ":all_model_processors_active",
    ],
)

selects.config_setting_group(
    name = "qwen_processor_auto_list",
    match_all = [
        ":enable_qwen_processor_auto",
        ":model_family_qwen_in_list",
    ],
)

selects.config_setting_group(
    name = "enable_qwen_processor_setting",
    match_any = [
        ":enable_qwen_processor_true",
        ":qwen_processor_auto_all",
        ":qwen_processor_auto_list",
    ],
    visibility = ["//visibility:public"],
)

string_flag(
    name = "enable_fastvlm_processor",
    build_setting_default = "auto",
    values = [
        "auto",
        "true",
        "false",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_fastvlm_processor_true",
    flag_values = {":enable_fastvlm_processor": "true"},
)

config_setting(
    name = "enable_fastvlm_processor_auto",
    flag_values = {":enable_fastvlm_processor": "auto"},
)

config_setting(
    name = "model_family_fastvlm_in_list",
    flag_values = {":enabled_model_families": "fastvlm"},
)

selects.config_setting_group(
    name = "fastvlm_processor_auto_all",
    match_all = [
        ":enable_fastvlm_processor_auto",
        ":all_model_processors_active",
    ],
)

selects.config_setting_group(
    name = "fastvlm_processor_auto_list",
    match_all = [
        ":enable_fastvlm_processor_auto",
        ":model_family_fastvlm_in_list",
    ],
)

selects.config_setting_group(
    name = "enable_fastvlm_processor_setting",
    match_any = [
        ":enable_fastvlm_processor_true",
        ":fastvlm_processor_auto_all",
        ":fastvlm_processor_auto_list",
    ],
    visibility = ["//visibility:public"],
)

string_flag(
    name = "enable_lfm2_processor",
    build_setting_default = "auto",
    values = [
        "auto",
        "true",
        "false",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_lfm2_processor_true",
    flag_values = {":enable_lfm2_processor": "true"},
)

config_setting(
    name = "enable_lfm2_processor_auto",
    flag_values = {":enable_lfm2_processor": "auto"},
)

config_setting(
    name = "model_family_lfm2_in_list",
    flag_values = {":enabled_model_families": "lfm2"},
)

selects.config_setting_group(
    name = "lfm2_processor_auto_all",
    match_all = [
        ":enable_lfm2_processor_auto",
        ":all_model_processors_active",
    ],
)

selects.config_setting_group(
    name = "lfm2_processor_auto_list",
    match_all = [
        ":enable_lfm2_processor_auto",
        ":model_family_lfm2_in_list",
    ],
)

selects.config_setting_group(
    name = "enable_lfm2_processor_setting",
    match_any = [
        ":enable_lfm2_processor_true",
        ":lfm2_processor_auto_all",
        ":lfm2_processor_auto_list",
    ],
    visibility = ["//visibility:public"],
)

string_flag(
    name = "enable_minicpm_processor",
    build_setting_default = "auto",
    values = [
        "auto",
        "true",
        "false",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_minicpm_processor_true",
    flag_values = {":enable_minicpm_processor": "true"},
)

config_setting(
    name = "enable_minicpm_processor_auto",
    flag_values = {":enable_minicpm_processor": "auto"},
)

config_setting(
    name = "model_family_minicpm_in_list",
    flag_values = {":enabled_model_families": "minicpm"},
)

selects.config_setting_group(
    name = "minicpm_processor_auto_all",
    match_all = [
        ":enable_minicpm_processor_auto",
        ":all_model_processors_active",
    ],
)

selects.config_setting_group(
    name = "minicpm_processor_auto_list",
    match_all = [
        ":enable_minicpm_processor_auto",
        ":model_family_minicpm_in_list",
    ],
)

selects.config_setting_group(
    name = "enable_minicpm_processor_setting",
    match_any = [
        ":enable_minicpm_processor_true",
        ":minicpm_processor_auto_all",
        ":minicpm_processor_auto_list",
    ],
    visibility = ["//visibility:public"],
)
