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

# [Google-internal load of `cc_binary`]
# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_shared_library`]
# [Google-internal load of `cc_test`]
load(
    "@litert//litert/build_common:litert_build_defs.bzl",
    "export_lrt_runtime_only_linkopt",
    "export_lrt_runtime_only_script",
)

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
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
    name = "android_webgpu",
    constraint_values = [
        "@platforms//os:android",
    ],
    values = {
        "copt": "-DLITERT_USE_WEBGPU_ACCELERATOR",
    },
)

config_setting(
    name = "disable_sentencepiece_tokenizer",
    define_values = {
        "DISABLE_SENTENCEPIECE_TOKENIZER": "1",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tokenizer",
    hdrs = ["tokenizer.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "tokenizer_test",
    srcs = ["tokenizer_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "sampling_cpu_util",
    srcs = ["sampling_cpu_util.cc"],
    hdrs = ["sampling_cpu_util.h"],
    deps = [
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "scoring_cpu_util",
    srcs = ["scoring_cpu_util.cc"],
    hdrs = ["scoring_cpu_util.h"],
    deps = [
        ":sampling_cpu_util",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "sampler",
    hdrs = ["sampler.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//runtime/proto:sampler_params_cc_proto",
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
    name = "sampling_cpu_util_test",
    srcs = ["sampling_cpu_util_test.cc"],
    deps = [
        ":sampling_cpu_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "scoring_cpu_util_test",
    srcs = ["scoring_cpu_util_test.cc"],
    deps = [
        ":sampling_cpu_util",
        ":scoring_cpu_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer",
    srcs = ["sentencepiece_tokenizer.cc"],
    hdrs = ["sentencepiece_tokenizer.h"],
    defines = ["ENABLE_SENTENCEPIECE_TOKENIZER"],
    deps = [
        ":tokenizer",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@sentencepiece//:sentencepiece_model_cc_proto",
        "@sentencepiece//:sentencepiece_processor",
    ],
)

cc_test(
    name = "sentencepiece_tokenizer_test",
    srcs = ["sentencepiece_tokenizer_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":sentencepiece_tokenizer",
        ":tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "huggingface_tokenizer",
    srcs = ["huggingface_tokenizer.cc"],
    hdrs = ["huggingface_tokenizer.h"],
    defines = ["ENABLE_HUGGINGFACE_TOKENIZER"],
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
    ] + select({
        "@platforms//os:windows": [
            "//rust:alloc_defs",
            "//rust:global_allocator",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "huggingface_tokenizer_test",
    srcs = ["huggingface_tokenizer_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":huggingface_tokenizer",
        ":tokenizer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "stop_token_detector",
    srcs = ["stop_token_detector.cc"],
    hdrs = ["stop_token_detector.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:tensor_buffer_util",
        "@litert//tflite/types:half",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "top_p_cpu_sampler_test",
    srcs = ["top_p_cpu_sampler_test.cc"],
    deps = [
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "@litert//tflite/types:half",
    ],
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
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc/internal:litert_shared_library",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

# This is the target to link all the static GPU samplers. If you plan to enable GPU sampling, this
# target should be linked.
cc_library(
    name = "default_static_gpu_samplers",
    deps = select({
        # If CAPI is linked dynamically, need to link the sampler dynamically as well.
        "@litert//litert:litert_link_capi_so": [],
        "@platforms//os:ios": [],
        "//conditions:default": [
        ],
    }) + select({
        "//conditions:default": [],
    }),
)

cc_test(
    name = "sampler_factory_test",
    srcs = ["sampler_factory_test.cc"],
    deps = [
        ":model_resources",
        ":model_resources_task",
        ":sampler_factory",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "sampler_factory_failed_dlopen_test",
    srcs = ["sampler_factory_failed_dlopen_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":sampler_factory",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/test:matchers",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:test_utils",
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
    name = "model_resources_litert_lm",
    srcs = ["model_resources_litert_lm.cc"],
    hdrs = ["model_resources_litert_lm.h"],
    deps = [
        ":model_resources",
        ":tokenizer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//schema/core:litertlm_header_schema",
    ] + select({
        ":disable_huggingface_tokenizer": [],
        "//conditions:default": [":huggingface_tokenizer"],
    }) + select({
        ":disable_sentencepiece_tokenizer": [],
        "//conditions:default": [":sentencepiece_tokenizer"],
    }) + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_buffer_ref",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
        ],
    }),
)

cc_library(
    name = "model_resources",
    hdrs = ["model_resources.h"],
    deps = [
        ":tokenizer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:scoped_file",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_model",
        ],
    }),
)

cc_library(
    name = "model_resources_task",
    srcs = ["model_resources_task.cc"],
    hdrs = [
        "model_resources_task.h",
    ],
    deps = [
        ":model_resources",
        ":sentencepiece_tokenizer",
        ":tokenizer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:metadata_util",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_buffer_ref",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
        ],
    }),
)

cc_library(
    name = "prompt_template",
    srcs = ["prompt_template.cc"],
    hdrs = ["prompt_template.h"],
    linkopts = select({
        "@platforms//os:windows": ["ntdll.lib"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/components/rust:minijinja_template_cpp",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "prompt_template_test",
    srcs = ["prompt_template_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":prompt_template",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "lora",
    srcs = ["lora.cc"],
    hdrs = ["lora.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # TODO: b/467362164 Move tflite_lora_utils to an OSS directory to support open sourcing LoRA.
        "//runtime/util:litert_status_util",
        "//runtime/util:lora_data",
        "//runtime/util:lora_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "lora_test",
    srcs = ["lora_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":lora",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/util:lora_data",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "lora_manager",
    srcs = ["lora_manager.cc"],
    hdrs = ["lora_manager.h"],
    deps = [
        ":lora",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor:executor_settings_base",
        "//runtime/util:litert_status_util",
        "//runtime/util:lora_data",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "lora_manager_test",
    srcs = ["lora_manager_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":lora_manager",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/executor:executor_settings_base",
        "//runtime/util:test_utils",
    ],
)
