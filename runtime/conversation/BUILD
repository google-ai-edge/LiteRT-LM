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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = ["@nlohmann_json//:json"],
)

cc_library(
    name = "thinking_config",
    hdrs = ["thinking_config.h"],
)

cc_library(
    name = "prompt_utils",
    srcs = ["prompt_utils.cc"],
    hdrs = ["prompt_utils.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation/model_data_processor",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "prompt_utils_lite",
    srcs = ["prompt_utils.cc"],
    hdrs = ["prompt_utils.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template_noop",
        "//runtime/conversation/model_data_processor",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "channel_util",
    srcs = ["channel_util.cc"],
    hdrs = ["channel_util.h"],
    deps = [
        ":io_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/engine:io_types",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "internal_callback_util",
    srcs = ["internal_callback_util.cc"],
    hdrs = ["internal_callback_util.h"],
    deps = [
        ":channel_util",
        ":io_types",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/engine:io_types",
    ],
)

cc_test(
    name = "internal_callback_util_test",
    srcs = ["internal_callback_util_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    deps = [
        ":internal_callback_util",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:gemma3_data_processor",
        "//runtime/conversation/model_data_processor:gemma3_data_processor_config",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "conversation",
    srcs = ["conversation.cc"],
    hdrs = ["conversation.h"],
    deps = [
        ":channel_util",
        ":internal_callback_util",
        ":io_types",
        ":prompt_utils",
        ":thinking_config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:constraint_provider",
        "//runtime/components/constrained_decoding:constraint_provider_config",
        "//runtime/components/constrained_decoding:constraint_provider_factory",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:gemma3_data_processor_config",
        "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
        "//runtime/conversation/model_data_processor:model_data_processor_factory",
        "//runtime/core:session_utils",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_type_utils",
    ],
)

# Minimal Conversation target ("LiteRT-LM Lite" profile) for binary-size-constrained
# clients. Together with `:engine_advanced_impl_lite`, this saves ~6 MB of
# uncompressed binary size on iOS arm64.
#
# Included features:
#   - Gemma3, Gemma4, TinyGemma, and Generic model data processors.
# Excluded heavy dependencies (linked against *_noop / *_lite targets instead):
#   - Rust crates (`minijinja` prompt templates, `llguidance` constrained
#     decoding, `tokenizers_cpp` HuggingFace tokenizer, and Rust tool-use
#     parsers) + ANTLR4 (~5 MB combined on Linux/Android; ~3.0 MB net on iOS
#     arm64 after linker `-dead_strip`)
#   - Constrained decoding / FST (:constraint_provider_factory_noop, ~0.5 MB)
#   - Non-Gemma model data processors (:model_data_processor_factory_lite)
#   - Legacy Artisan GPU/Audio executors (via :engine_advanced_impl_lite, ~2.5 MB)
#
# Usage expectations:
#   1. Pair `:conversation_lite` with `//runtime/core:engine_advanced_impl_lite`
#      (or `:engine_advanced_impl_cpu_only_lite`) so heavy dependencies are excluded
#      across both the Conversation and Engine layers.
#   2. Do NOT link both `:conversation` and `:conversation_lite` (or both standard and
#      `*_lite` variants of the same subsystem) in the same binary, as they provide
#      competing link-time definitions for the same factory/component symbols.
cc_library(
    name = "conversation_lite",
    srcs = ["conversation.cc"],
    hdrs = ["conversation.h"],
    deps = [
        ":channel_util",
        ":internal_callback_util",
        ":io_types",
        ":prompt_utils_lite",
        ":thinking_config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template_noop",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:constraint_provider",
        "//runtime/components/constrained_decoding:constraint_provider_config",
        "//runtime/components/constrained_decoding:constraint_provider_factory_noop",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/conversation/model_data_processor",
        "//runtime/conversation/model_data_processor:config_registry",
        "//runtime/conversation/model_data_processor:gemma3_data_processor_config",
        "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
        "//runtime/conversation/model_data_processor:model_data_processor_factory_lite",
        "//runtime/core:session_utils",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_type_utils",
    ],
)

cc_test(
    name = "conversation_lite_test",
    srcs = ["conversation_lite_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],
    deps = [
        ":conversation_lite",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:prompt_template_noop",
        "//runtime/components/tool_use:parser_utils_lite",
        "//runtime/conversation/model_data_processor:fastvlm_data_processor_config",
        "//runtime/conversation/model_data_processor:gemma3_data_processor_config",
        "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
        "//runtime/conversation/model_data_processor:model_data_processor_factory_lite",
        "//runtime/conversation/model_data_processor:qwen3_data_processor_config",
        "//runtime/core:engine_advanced_impl_lite",  # buildcleaner: keep
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
        "//support/tokenizer:huggingface_tokenizer_noop",
    ],
)

cc_test(
    name = "channel_util_test",
    srcs = ["channel_util_test.cc"],
    deps = [
        ":channel_util",
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "conversation_test",
    size = "medium",
    srcs = ["conversation_test.cc"],
    data = [
        "//runtime/components/testdata",
        "//runtime/testdata",
        "//support/preprocessor/testdata",
    ],
    shard_count = 4,
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":conversation",
        ":io_types",
        ":thinking_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/components/constrained_decoding:bitmap",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:external_constraint_config",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/engine:litert_lm_lib",  # buildcleaner: keep
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "prompt_utils_test",
    srcs = ["prompt_utils_test.cc"],
    deps = [
        ":io_types",
        ":prompt_utils",
        "@com_google_googletest//:gtest_main",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation/model_data_processor:gemma3_data_processor",
        "//runtime/util:test_utils",
    ],
)
