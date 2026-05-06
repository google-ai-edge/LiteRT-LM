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
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "execution_manager",
    hdrs = ["execution_manager.h"],
    deps = [
        ":resource_manager",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/components:sampler",
        "//runtime/components:stop_token_detector",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/framework/resource_management/context_handler",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
        ],
    }),
)

cc_library(
    name = "threaded_execution_manager",
    srcs = ["threaded_execution_manager.cc"],
    hdrs = ["threaded_execution_manager.h"],
    deps = [
        ":execution_manager",
        ":resource_manager",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/components:model_resources",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components:stop_token_detector",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/core:tasks",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:vision_executor_settings",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:executor_data_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:tensor_buffer_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_environment_options",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "serial_execution_manager",
    srcs = ["serial_execution_manager.cc"],
    hdrs = ["serial_execution_manager.h"],
    deps = [
        ":execution_manager",
        ":resource_manager",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//runtime/components:model_resources",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components:stop_token_detector",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/core:tasks",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:vision_executor_settings",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:executor_data_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:tensor_buffer_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_environment_options",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "execution_manager_test",
    srcs = ["execution_manager_test.cc"],
    deps = [
        ":execution_manager",
        ":serial_execution_manager",
        ":threaded_execution_manager",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:model_resources",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/proto:token_cc",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

RESOURCE_MANAGER_DEPS = [
    "@com_google_absl//absl/base:core_headers",
    "@com_google_absl//absl/base:nullability",
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/synchronization",
    "@com_google_absl//absl/types:span",
    "//runtime/components:model_resources",
    "//runtime/engine:engine_settings",
    "//runtime/engine:io_types",
    "//runtime/executor:audio_executor",
    "//runtime/executor:audio_executor_settings",
    "//runtime/executor:audio_litert_compiled_model_executor",
    "//runtime/executor:executor_settings_base",
    "//runtime/executor:litert_compiled_model_executor_utils",
    "//runtime/executor:llm_executor",
    "//runtime/executor:llm_executor_io_types",
    "//runtime/executor:llm_executor_processed_tokens",
    "//runtime/executor:llm_executor_settings",
    "//runtime/executor:vision_executor",
    "//runtime/executor:vision_executor_settings",
    "//runtime/executor:vision_litert_compiled_model_executor",
    "//runtime/framework/resource_management/context_handler",
    "//runtime/framework/resource_management/utils:movable_mutex_lock",
    "//runtime/framework/resource_management/utils:resource_manager_utils",
    "//runtime/util:convert_tensor_buffer",
    "//runtime/util:litert_status_util",
    "//runtime/util:logging",
]

cc_library(
    name = "resource_manager",
    srcs = ["resource_manager.cc"],
    hdrs = ["resource_manager.h"],
    defines = ["LITERT_LM_EXCLUDE_LEGACY_EXECUTORS=1"],
    deps = RESOURCE_MANAGER_DEPS +
           select({
               "@litert//litert:litert_link_capi_so": [
                   "@litert//litert/cc:litert_api_with_dynamic_runtime",
               ],
               "//conditions:default": [
                   "@litert//litert/cc:litert_environment",
                   "@litert//litert/cc:litert_environment_options",
                   "@litert//litert/cc:litert_macros",
                   "@litert//litert/cc:litert_tensor_buffer",
               ],
           }),
)
