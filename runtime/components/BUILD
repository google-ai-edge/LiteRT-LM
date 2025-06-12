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

load("@litert//litert/build_common:litert_build_defs.bzl", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

config_setting(
    name = "disable_huggingface_tokenizer",
    define_values = {
        "DISABLE_HUGGINGFACE_TOKENIZER": "1",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "disable_sentencepiece_tokenizer",
    define_values = {
        "DISABLE_SENTENCEPIECE_TOKENIZER": "1",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "embedding_lookup",
    hdrs = ["embedding_lookup.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "embedding_lookup_text",
    srcs = ["embedding_lookup_text.cc"],
    hdrs = ["embedding_lookup_text.h"],
    deps = [
        ":embedding_lookup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/util:litert_status_util",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_compiled_model",
            "@litert//litert/cc/internal:litert_environment",
            "@litert//litert/cc/internal:litert_options",
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "tokenizer",
    hdrs = ["tokenizer.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "//runtime/util:convert_tensor_buffer",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "tokenizer_test",
    srcs = ["tokenizer_test.cc"],
    deps = [
        ":tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/test:matchers",
        "//runtime/util:convert_tensor_buffer",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "sampling_cpu_util",
    srcs = ["sampling_cpu_util.cc"],
    hdrs = ["sampling_cpu_util.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sampler",
    hdrs = ["sampler.h"],
    deps = ["@com_google_absl//absl/status"] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "sampling_cpu_util_test",
    srcs = ["sampling_cpu_util_test.cc"],
    deps = [
        ":sampling_cpu_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer",
    srcs = ["sentencepiece_tokenizer.cc"],
    hdrs = ["sentencepiece_tokenizer.h"],
    deps = [
        ":tokenizer",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@sentencepiece//:sentencepiece_processor",
    ],
)

cc_test(
    name = "sentencepiece_tokenizer_test",
    srcs = ["sentencepiece_tokenizer_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":sentencepiece_tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "huggingface_tokenizer",
    srcs = ["huggingface_tokenizer.cc"],
    hdrs = ["huggingface_tokenizer.h"],
    deps = [
        ":tokenizer",
        "@com_google_absl//absl/debugging:leak_check",  # See b/402708346
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "@tokenizers_cpp//:huggingface_tokenizer",
    ],
)

cc_test(
    name = "huggingface_tokenizer_test",
    srcs = ["huggingface_tokenizer_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":huggingface_tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "stop_token_detector",
    srcs = ["stop_token_detector.cc"],
    hdrs = ["stop_token_detector.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "stop_token_detector_test",
    srcs = ["stop_token_detector_test.cc"],
    deps = [
        ":stop_token_detector",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "token_id_util",
    srcs = ["token_id_util.cc"],
    hdrs = ["token_id_util.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "token_id_util_test",
    srcs = ["token_id_util_test.cc"],
    deps = [
        ":token_id_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "top_p_cpu_sampler",
    srcs = ["top_p_cpu_sampler.cc"],
    hdrs = ["top_p_cpu_sampler.h"],
    deps = [
        ":sampler",
        ":sampling_cpu_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:tensor_buffer_util",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "top_p_cpu_sampler_test",
    srcs = ["top_p_cpu_sampler_test.cc"],
    deps = [
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "sampler_factory",
    srcs = ["sampler_factory.cc"],
    hdrs = ["sampler_factory.h"],
    deps = [
        ":sampler",
        ":top_p_cpu_sampler",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_shared_library",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }) + select({
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": [
        ],
    }),
)

cc_test(
    name = "sampler_factory_test",
    srcs = ["sampler_factory_test.cc"],
    deps = [
        ":model_resources_task",
        ":sampler_factory",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_expected",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc/options:litert_gpu_options",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_compiled_model",
            "@litert//litert/cc/internal:litert_environment",
            "@litert//litert/cc/internal:litert_gpu_options",
            "@litert//litert/cc/internal:litert_options",
        ],
    }) + select({
        "//conditions:default": [],
    }),
)

cc_test(
    name = "sampler_factory_failed_dlopen_test",
    srcs = ["sampler_factory_failed_dlopen_test.cc"],
    deps = [
        ":sampler_factory",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/test:matchers",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:test_utils",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_environment",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_environment",
        ],
    }),
)

cc_library(
    name = "model_resources_litert_lm",
    srcs = ["model_resources_litert_lm.cc"],
    hdrs = ["model_resources_litert_lm.h"],
    deps = [
        ":model_resources_task",
        ":tokenizer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
    ] + select({
        ":disable_huggingface_tokenizer": [],
        "//conditions:default": [":huggingface_tokenizer"],
    }) + select({
        ":disable_sentencepiece_tokenizer": [],
        "//conditions:default": [":sentencepiece_tokenizer"],
    }),
)

cc_library(
    name = "model_resources_task",
    srcs = ["model_resources_task.cc"],
    hdrs = [
        "model_resources.h",
        "model_resources_task.h",
    ],
    deps = [
        ":sentencepiece_tokenizer",
        ":tokenizer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:metadata_util",
        "//runtime/util:model_asset_bundle_resources",
    ],
)
