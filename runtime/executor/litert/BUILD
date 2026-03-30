# Copyright 2026 The ODML Authors.
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

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "debug_utils",
    srcs = ["debug_utils.cc"],
    hdrs = ["debug_utils.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "debug_utils_test",
    srcs = ["debug_utils_test.cc"],
    deps = [
        ":debug_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "kv_cache",
    srcs = ["kv_cache.cc"],
    hdrs = ["kv_cache.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor:common_utils",
        "//runtime/executor:kv_cache_interface",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_model_types",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc:litert_tensor_buffer_types",
        ],
    }),
)

cc_test(
    name = "kv_cache_test",
    srcs = ["kv_cache_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    deps = [
        ":kv_cache",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor",
    srcs = ["llm_executor.cc"],
    hdrs = ["llm_executor.h"],
    deps = [
        ":kv_cache",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/components:model_resources",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:kv_cache_interface",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor_interface",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_cache_utils",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:file_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:lora_util",
        "//runtime/util:scoped_file",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_model_types",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc:litert_tensor_buffer_types",
            "@litert//litert/cc/options:litert_cpu_options",
            "@litert//litert/cc/options:litert_gpu_options",
            "@litert//litert/cc/options:litert_runtime_options",
        ],
    }),
)

cc_test(
    name = "llm_executor_cpu_test",
    srcs = ["llm_executor_cpu_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    deps = [
        ":llm_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "llm_executor_gpu_test",
    srcs = ["llm_executor_gpu_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":llm_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)
