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
    ],
)

ENGINE_IMPL_COMMON_DEPS = [
    ":session_advanced",
    "@com_google_absl//absl/base:no_destructor",
    "@com_google_absl//absl/log",
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/log:check",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/time",
    "@litert//litert/cc:litert_macros",
    "//runtime/components:model_resources",
    "@litert//support/tokenizer",
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
    }),
    deps = ENGINE_IMPL_COMMON_DEPS + [
        "//runtime/components:default_static_gpu_samplers",
        "//runtime/executor:default_static_gpu_accelerator",
    ] + select({
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "engine_advanced_impl_cpu_only",
    srcs = ["engine_advanced_impl.cc"],
    deps = ENGINE_IMPL_COMMON_DEPS + select({
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//support/tokenizer",
        "//runtime/components:sampler",
        "//runtime/components:scoring_cpu_util",
        "//runtime/components:stop_token_detector",
        "//runtime/components:token_id_util",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/logits_processor:repetition_penalty_config",
        "//runtime/components/logits_processor/constrained_decoding:constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@litert//support/tokenizer",
        "@litert//support/tokenizer:sentencepiece_tokenizer",
        "//runtime/components:stop_token_detector",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/logits_processor:repetition_penalty_config",
        "//runtime/components/logits_processor/constrained_decoding:fake_constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "session_advanced",
    srcs = ["session_advanced.cc"],
    hdrs = ["session_advanced.h"],
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@litert//support/tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/framework/resource_management:execution_manager",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
    ],
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
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "@litert//support/tokenizer",
        "@litert//support/tokenizer:sentencepiece_tokenizer",
        "//runtime/components:model_resources",
        "//runtime/components/logits_processor/constrained_decoding:fake_constraint",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:audio_litert_compiled_model_executor",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/framework/resource_management:execution_manager",
        "//runtime/framework/resource_management:threaded_execution_manager",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:tensor_buffer_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "session_utils",
    srcs = ["session_utils.cc"],
    hdrs = ["session_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc/internal:litert_detail",
        "@litert//support/tokenizer",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "@litert//support/tokenizer",
        "@litert//support/tokenizer:sentencepiece_tokenizer",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/proto:sampler_params_cc",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:tensor_buffer_util",
        "//runtime/util:test_utils",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//support/tokenizer",
        "//runtime/components:sampler",
        "//runtime/components:scoring_cpu_util",
        "//runtime/components:stop_token_detector",
        "//runtime/components/logits_processor",
        "//runtime/components/logits_processor:repetition_penalty_config",
        "//runtime/components/logits_processor:repetition_penalty_processor",
        "//runtime/components/logits_processor/constrained_decoding:constrained_decoder",
        "//runtime/components/logits_processor/constrained_decoding:constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor",
        "//runtime/proto:sampler_params_cc",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@litert//support/tokenizer",
        "@litert//support/tokenizer:sentencepiece_tokenizer",
        "//runtime/components:stop_token_detector",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/logits_processor:repetition_penalty_config",
        "//runtime/components/logits_processor/constrained_decoding:fake_constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)
