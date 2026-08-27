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

licenses(["notice"])

# In most case, pick the engine implementation with
# //runtime/core:engine_impl_selected.
#
package_group(
    name = "engine_impl_users",
    packages = [
        # for ":engine_impl_selected".
        "//runtime/engine",
        # for e2e tests.
        "//runtime/e2e_tests/...",
        "//c/...",
        "//python/...",
        "//kotlin/...",
        # for SpeechProcessor
        # for AI Edge Hub
        # for AI Edge Gallery
        # for MediaPipe
        # for Tunelab Gemma
    ],
)

ENGINE_IMPL_COMMON_DEPS = [
    ":session_advanced",
    ":cached_session",
    "@com_google_absl//absl/base:no_destructor",
    "@com_google_absl//absl/log",
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/log:check",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:status_macros",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/time",
    "@litert//litert/cc:litert_macros",
    "//runtime/components:model_resources",
    "//support/tokenizer",
    "//runtime/engine:engine_factory",
    "//runtime/engine:engine_interface",
    "//runtime/engine:engine_settings",
    "//runtime/engine:io_types",
    "//runtime/executor:audio_executor",
    "//runtime/executor:audio_executor_settings",
    "//runtime/executor:audio_executor_utils",
    "//runtime/executor:audio_litert_compiled_model_executor",
    "//runtime/executor:executor_settings_base",
    "//runtime/executor:litert_compiled_model_executor_utils",
    "//runtime/executor:llm_executor",
    "//runtime/executor:llm_executor_settings",
    "//runtime/executor:llm_litert_compiled_model_executor",
    "//runtime/executor:llm_litert_compiled_model_executor_factory",
    "//runtime/executor:magic_number_configs_helper",
    "//runtime/executor:model_signature_utils",
    "//runtime/executor:vision_executor",
    "//runtime/executor:vision_executor_settings",
    "//runtime/executor:vision_executor_utils",
    "//runtime/executor:vision_litert_compiled_model_executor",
    "//runtime/framework:threadpool",
    "//runtime/framework/resource_management:execution_manager",
    "//runtime/framework/resource_management:serial_execution_manager",
    "//runtime/framework/resource_management:threaded_execution_manager",
    "//runtime/proto:llm_metadata_cc_proto",
    "//runtime/proto:sampler_params_cc_proto",
    "//runtime/util:file_format_util",
    "//runtime/util:litert_status_util",
    "//runtime/util:litert_util",
    "//runtime/util:logging",
    "@litert//litert/cc:litert_environment",
    "@litert//litert/cc:litert_environment_options",
]

cc_library(
    name = "engine_advanced_impl",
    srcs = ["engine_advanced_impl.cc"],
    local_defines = select({
        "//conditions:default": [],
    }) + select({
        "//runtime/util:litert_lm_debugger_enabled": [
            "LITERT_LM_DEBUGGER_ENABLED=1",
        ],
        "//conditions:default": [],
    }),
    deps = ENGINE_IMPL_COMMON_DEPS + [
        "//runtime/components:default_static_gpu_samplers",
        "//runtime/executor:default_static_gpu_accelerator",
    ] + select({
        "//conditions:default": [],
    }) + select({
        "//runtime/util:litert_lm_debugger_enabled": [
            "//runtime/util:runtime_debugger",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "engine_advanced_impl_cpu_only",
    srcs = ["engine_advanced_impl.cc"],
    local_defines = select({
        "//runtime/util:litert_lm_debugger_enabled": [
            "LITERT_LM_DEBUGGER_ENABLED=1",
        ],
        "//conditions:default": [],
    }),
    deps = ENGINE_IMPL_COMMON_DEPS + select({
        "//runtime/util:litert_lm_debugger_enabled": [
            "//runtime/util:runtime_debugger",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_test(
    name = "engine_advanced_impl_test",
    srcs = ["engine_advanced_impl_test.cc"],
    data = ["//runtime/testdata"],
    defines = ["ENGINE_ADVANCED"],
    tags = ["requires-mac-inputs:hard"],
    deps = [
        ":engine_advanced_impl",  # buildcleaner: keep
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

# TODO - b/502275587: Remove these aliases once the migration is complete.
alias(
    name = "engine_impl",
    actual = ":engine_advanced_impl",
    deprecation = "Use engine_advanced_impl instead.",
)

alias(
    name = "engine_impl_cpu_only",
    actual = ":engine_advanced_impl_cpu_only",
    deprecation = "Use engine_advanced_impl_cpu_only instead.",
)

cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        ":tasks",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:sampler",
        "//runtime/components:stop_token_detector",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":pipeline",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/components:stop_token_detector",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
        "//support/tokenizer",
        "//support/tokenizer:sentencepiece_tokenizer",
    ],
)

cc_library(
    name = "session_advanced",
    srcs = ["session_advanced.cc"],
    hdrs = ["session_advanced.h"],
    local_defines = select({
        "//runtime/util:litert_lm_debugger_enabled": [
            "LITERT_LM_DEBUGGER_ENABLED=1",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":session_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/framework/resource_management:execution_manager",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
        "//support/tokenizer",
    ] + select({
        "//runtime/util:litert_lm_debugger_enabled": [
            "//runtime/util:runtime_debugger",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "session_advanced_test",
    srcs = ["session_advanced_test.cc"],
    data = [
        "//runtime/components/testdata",
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":session_advanced",
        ":session_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:fake_llm_executor",
        "//runtime/framework/resource_management:execution_manager",
        "//runtime/framework/resource_management:threaded_execution_manager",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
        "//support/tokenizer",
        "//support/tokenizer:sentencepiece_tokenizer",
    ],
)

cc_library(
    name = "session_utils",
    srcs = ["session_utils.cc"],
    hdrs = ["session_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc/internal:litert_detail",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "session_utils_test",
    srcs = ["session_utils_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    deps = [
        ":session_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:sampler_params_cc",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
        "//support/tokenizer",
        "//support/tokenizer:sentencepiece_tokenizer",
    ],
)

cc_library(
    name = "tasks",
    srcs = ["tasks.cc"],
    hdrs = ["tasks.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:sampler",
        "//runtime/components:scoring_cpu_util",
        "//runtime/components:stop_token_detector",
        "//runtime/components/constrained_decoding:composite_constraint",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:no_repeat_ngram_constraint",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:repetition_penalty_constraint",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/components/constrained_decoding:suppress_tokens_constraint",
        "//runtime/components/constrained_decoding:thinking_budget_constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor",
        "//runtime/proto:sampler_params_cc",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//support/tokenizer",
        "//support/tokenizer:buffered_streaming_detokenizer",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "tasks_test",
    srcs = ["tasks_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":tasks",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/components:stop_token_detector",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/components/constrained_decoding:no_repeat_ngram_config",
        "//runtime/components/constrained_decoding:repetition_penalty_config",
        "//runtime/components/constrained_decoding:suppress_tokens_config",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
        "//support/tokenizer",
        "//support/tokenizer:sentencepiece_tokenizer",
    ],
)

cc_library(
    name = "prefix_cache",
    srcs = ["prefix_cache.cc"],
    hdrs = ["prefix_cache.h"],
    deps = ["@com_google_absl//absl/types:span"],
)

cc_test(
    name = "prefix_cache_test",
    srcs = ["prefix_cache_test.cc"],
    deps = [
        ":prefix_cache",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cached_session",
    srcs = ["cached_session.cc"],
    hdrs = ["cached_session.h"],
    deps = [
        ":prefix_cache",
        ":session_utils",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor_io_types",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "cached_session_test",
    srcs = ["cached_session_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    deps = [
        ":cached_session",
        ":prefix_cache",
        ":session_advanced",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework/resource_management:threaded_execution_manager",
        "//runtime/proto:engine_cc_proto",
        "//support/tokenizer",
        "//support/tokenizer:sentencepiece_tokenizer",
    ],
)

cc_library(
    name = "embedding_engine_impl",
    srcs = ["embedding_engine_impl.cc"],
    hdrs = ["embedding_engine_impl.h"],
    visibility = [":engine_impl_users"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:model_resources",
        "//runtime/engine:embedding_engine",
        "//runtime/engine:embedding_engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:audio_litert_compiled_model_executor",
        "//runtime/executor:embedding_executor_base",
        "//runtime/executor:embedding_executor_settings",
        "//runtime/executor:embedding_litert_compiled_model_executor",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:model_signature_utils",
        "//runtime/executor:vision_executor",
        "//runtime/executor:vision_litert_compiled_model_executor",
        "//runtime/proto:embedding_metadata_cc_proto",
        "//runtime/proto:embedding_model_type_cc_proto",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:executor_data_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:litert_util",
        "//runtime/util:tensor_buffer_util",
        "//support/preprocessor:audio_preprocessor",
        "//support/preprocessor:audio_preprocessor_miniaudio",
        "//support/preprocessor:image_preprocessor",
        "//support/preprocessor:image_preprocessor_impl",  # buildcleaner: keep
        "//support/tokenizer",
        "//support/util:io_types",
    ],
)

cc_test(
    name = "embedding_engine_impl_test",
    srcs = ["embedding_engine_impl_test.cc"],
    data = [
        "//runtime/testdata:test_embedding.litertlm",
        "//runtime/testdata:test_lm.litertlm",
    ],
    deps = [
        ":embedding_engine_impl",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/engine:embedding_engine",
        "//runtime/engine:embedding_engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:embedding_executor_base",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:vision_executor",
        "//runtime/proto:embedding_metadata_cc_proto",
        "//runtime/proto:embedding_model_type_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
        "//support/preprocessor:image_preprocessor",
        "//support/preprocessor:image_preprocessor_impl",  # buildcleaner: keep
        "//support/tokenizer",
    ],
)
