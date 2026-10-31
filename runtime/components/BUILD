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

load("@rules_cc//cc:cc_import.bzl", "cc_import")
# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_shared_library`]
# [Google-internal load of `cc_test`]
# [Google-internal load of `objc_library`]
load(
    "@bazel_skylib//rules:copy_file.bzl",
    "copy_file",
)
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework", "ios_unit_test")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib", "macos_unit_test")

LITERT_MIN_IOS_VERSION = "15.0"

LITERT_MIN_MACOS_VERSION = "12.0"

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

alias(
    name = "disable_huggingface_tokenizer",
    actual = "//build_config:disable_huggingface_tokenizer",
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_webgpu",
    constraint_values = ["@platforms//os:android"],
    values = {"copt": "-DLITERT_USE_WEBGPU_ACCELERATOR"},
)

alias(
    name = "disable_sentencepiece_tokenizer",
    actual = "//build_config:disable_sentencepiece_tokenizer",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sampling_cpu_util",
    srcs = ["sampling_cpu_util.cc"],
    hdrs = ["sampling_cpu_util.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "sampler",
    hdrs = ["sampler.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/proto:sampler_params_cc_proto",
    ],
)

cc_test(
    name = "sampling_cpu_util_test",
    srcs = ["sampling_cpu_util_test.cc"],
    deps = [
        ":sampling_cpu_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "scoring_cpu_util_test",
    srcs = ["scoring_cpu_util_test.cc"],
    deps = [
        ":scoring_cpu_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:tensor_buffer_util",
        "@litert//tflite/types:half",
    ],
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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_shared_library",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

# This is the target to link all the static GPU samplers. If you plan to enable GPU sampling, this
# target should be linked.
#
# The sampler backend has to match the inference backend linked by
# //runtime/executor:default_static_gpu_accelerator: the sampler is
# created from the device published by the accelerator, so linking a sampler for a different
# backend makes GPU sampler creation fail and silently degrades to CPU sampling. That is why the
# Apple conditions below follow the same
# //runtime/executor:gpu_accelerator flag.
cc_library(
    name = "default_static_gpu_samplers",
    deps = select({
        ":android_webgpu": [":top_k_webgpu_sampler_static"],
        "@platforms//os:android": [":top_k_opencl_sampler_static"],
        "//runtime/executor:ios_webgpu": [":top_k_webgpu_sampler_static"],
        "//runtime/executor:macos_metal": [":top_k_metal_sampler_static"],
        "@platforms//os:ios": [":top_k_metal_sampler_static"],
        "@platforms//os:macos": [
            ":top_k_metal_sampler_static",
            ":top_k_webgpu_sampler_static",
        ],
        "//conditions:default": [":top_k_webgpu_sampler_static"],
    }),
)

cc_test(
    name = "sampler_factory_test",
    srcs = ["sampler_factory_test.cc"],
    deps = [
        ":sampler",
        ":sampler_factory",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
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
        "@litert//litert/cc:litert_environment",
        "@litert//litert/test:matchers",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "model_resources_litert_lm",
    srcs = ["model_resources_litert_lm.cc"],
    hdrs = ["model_resources_litert_lm.h"],
    deps = [
        ":model_resources",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//runtime/proto:embedding_metadata_cc_proto",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//schema/core:litertlm_header_schema",
        "//support/tokenizer",
        "//support/tokenizer:huggingface_tokenizer_noop",
    ],
)

cc_test(
    name = "model_resources_litert_lm_test",
    srcs = ["model_resources_litert_lm_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":model_resources_litert_lm",
        "@com_google_googletest//:gtest_main",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "model_resources",
    srcs = ["model_resources.cc"],
    hdrs = ["model_resources.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_model",
        "//runtime/proto:any2any_metadata_cc_proto",
        "//runtime/proto:embedding_metadata_cc_proto",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:speech2text_metadata_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2speech_metadata_cc_proto",
        "//runtime/util:scoped_file",
        "//support/tokenizer",
    ] + select({
        ":disable_sentencepiece_tokenizer": [],
        "//conditions:default": ["//support/tokenizer:sentencepiece_tokenizer"],
    }),
)

cc_library(
    name = "model_resources_task",
    srcs = ["model_resources_task.cc"],
    hdrs = ["model_resources_task.h"],
    deps = [
        ":model_resources",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
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
        "//support/tokenizer:sentencepiece_tokenizer",
    ],
)

cc_library(
    name = "top_k_gpu_sampler",
    srcs = ["top_k_gpu_sampler.cc"],
    hdrs = ["top_k_gpu_sampler.h"],
    deps = [
        ":sampler",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

METAL_SAMPLER_COMMON_DEPS = [
    ":top_k_gpu_sampler",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "@ml_drift//ml_drift/common:data_type",
    "@ml_drift//ml_drift/common:gpu_info",
    "@ml_drift//ml_drift/common:gpu_model",
    "@ml_drift//ml_drift/common:gpu_model_builder",
    "@ml_drift//ml_drift/common:model",
    "@ml_drift//ml_drift/common:model_hints",
    "@ml_drift//ml_drift/common:precision",
    "@ml_drift//ml_drift/common:shape",
    "@ml_drift//ml_drift/common:util",
    "@ml_drift//ml_drift/common/task:gpu_operation",
    "@ml_drift//ml_drift/common/task:tensor_desc",
    "@ml_drift//ml_drift/metal:common",
    "@ml_drift//ml_drift/metal:compute_task",
    "@ml_drift//ml_drift/metal:environment",
    "@ml_drift//ml_drift/metal:inference_context",
    "@ml_drift//ml_drift/metal:metal_spatial_tensor",
    "@litert//ml_drift_delegate/delegate:custom_event_metal",
    "//runtime/executor:executor_settings_base",
    "//runtime/proto:sampler_params_cc_proto",
    "//runtime/util:litert_status_util",
]

objc_library(
    name = "top_k_metal_sampler_static",
    srcs = ["top_k_metal_sampler.mm"],
    copts = ["-ObjC++"],
    defines = ["LITERT_LM_USE_STATIC_LINKED_GPU_SAMPLER"],
    features = ["-layering_check"],
    target_compatible_with = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    textual_hdrs = ["top_k_metal_sampler.h"],
    deps = METAL_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/status:status_macros",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_event",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_handle",
    ],
    alwayslink = 1,
)

# This module requires the shared library of LiteRT runtime `libLiteRt.so`.
# Need extra flag to build this target: --define=litert_runtime_link_mode=dynamic.
objc_library(
    name = "top_k_metal_sampler_runtimecapi",
    srcs = ["top_k_metal_sampler.mm"],
    copts = ["-ObjC++"],
    features = ["-layering_check"],
    target_compatible_with = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    textual_hdrs = ["top_k_metal_sampler.h"],
    visibility = ["//visibility:private"],
    deps = METAL_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/status:status_macros",
        "@litert//litert/c:litert_common",
        "@litert//litert/c/internal:litert_runtime_builtin_none",
        "@litert//litert/cc:litert_environment_api",
        "@litert//litert/cc:litert_event",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_handle",
    ],
)

OPENCL_SAMPLER_COMMON_DEPS = [
    ":top_k_gpu_sampler",
    "@com_google_absl//absl/base:nullability",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "@ml_drift//ml_drift/cl:cl_command_queue",
    "@ml_drift//ml_drift/cl:cl_context",
    "@ml_drift//ml_drift/cl:cl_device",
    "@ml_drift//ml_drift/cl:cl_event",
    "@ml_drift//ml_drift/cl:cl_operation",
    "@ml_drift//ml_drift/cl:environment",
    "@ml_drift//ml_drift/cl:inference_context",
    "@ml_drift//ml_drift/cl:opencl_wrapper",
    "@ml_drift//ml_drift/cl:tensor",
    "@ml_drift//ml_drift/cl:util",
    "@ml_drift//ml_drift/common:data_type",
    "@ml_drift//ml_drift/common:gpu_info",
    "@ml_drift//ml_drift/common:gpu_model",
    "@ml_drift//ml_drift/common:gpu_model_builder",
    "@ml_drift//ml_drift/common:model",
    "@ml_drift//ml_drift/common:model_hints",
    "@ml_drift//ml_drift/common:precision",
    "@ml_drift//ml_drift/common:shape",
    "@ml_drift//ml_drift/common:util",
    "@ml_drift//ml_drift/common/task:gpu_operation",
    "@ml_drift//ml_drift/common/task:tensor_desc",
    "//runtime/executor:executor_settings_base",
    "//runtime/proto:sampler_params_cc_proto",
    "//runtime/util:litert_status_util",
    "@litert//tflite/delegates/gpu/cl:opencl_wrapper",
]

# Basic version of TopKOpenClSampler that can be directly used with #include.
cc_library(
    name = "top_k_opencl_sampler",
    srcs = ["top_k_opencl_sampler.cc"],
    hdrs = ["top_k_opencl_sampler.h"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = OPENCL_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/status:status_macros",
        "@litert//litert/cc:litert_any",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_handle",
    ],
)

# Static version of the OpenCL Sampler module for LiteRT LM.
cc_library(
    name = "top_k_opencl_sampler_static",
    srcs = ["top_k_opencl_sampler.cc"],
    hdrs = ["top_k_opencl_sampler.h"],
    defines = ["LITERT_LM_USE_STATIC_LINKED_GPU_SAMPLER"],
    linkopts = select({
        "@platforms//os:android": ["-landroid"],
        "//conditions:default": [],
    }),
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = OPENCL_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/status:status_macros",
        "@litert//litert/cc:litert_any",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_handle",
    ],
    alwayslink = 1,
)

# This module requires the shared library of LiteRT runtime `libLiteRt.so`.
# Need extra flag to build this target: --define=litert_runtime_link_mode=dynamic.
cc_library(
    name = "top_k_opencl_sampler_runtimecapi",
    srcs = ["top_k_opencl_sampler.cc"],
    hdrs = ["top_k_opencl_sampler.h"],
    features = ["-layering_check"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:private"],
    deps = OPENCL_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/status:status_macros",
        "@litert//litert/c/internal:litert_runtime_builtin_none",
        "@litert//litert/cc:litert_any",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment_api",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_handle",
    ],
)

cc_test(
    name = "top_k_opencl_sampler_test",
    srcs = ["top_k_opencl_sampler_test.cc"],
    data = ["//runtime/testdata"],
    tags = [
        "manual",
        "nopresubmit",
        "requires-gpu-sm90",
    ],
    deps = [
        ":model_resources",
        ":model_resources_task",
        ":top_k_opencl_sampler",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "@litert//litert/test:matchers",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "top_k_opencl_sampler_c_api_test",
    srcs = ["top_k_opencl_sampler_c_api_test.cc"],
    copts = ["-Wreturn-type-c-linkage"],
    data = ["//runtime/testdata"],
    tags = [
        "manual",
        "nopresubmit",
        "requires-gpu-sm90",
    ],
    deps = [
        ":model_resources",
        ":model_resources_task",
        ":top_k_opencl_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "@litert//litert/test:matchers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ],
)

cc_shared_library(
    name = "top_k_opencl_sampler_so",
    shared_lib_name = "libLiteRtTopKOpenClSampler.so",
    tags = ["manual"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    user_link_flags = select({
        "@platforms//os:android": [
            "-landroid",
            "-Wl,-soname=libLiteRtTopKOpenClSampler.so",
            # Align PT_LOAD segments and pad GNU_RELRO to a 16 KB boundary.
            # See https://developer.android.com/guide/practices/page-sizes#relro-flag.
            "-Wl,-z,max-page-size=16384",
            "-Wl,-z,common-page-size=16384",
        ],
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["-Wl,-soname=libLiteRtTopKOpenClSampler.so"],
    }),
    deps = [":top_k_opencl_sampler_runtimecapi"],
)

cc_library(
    name = "top_k_opencl_sampler_shared_lib",
    srcs = [":top_k_opencl_sampler_so"],
    linkstatic = 1,
    tags = ["manual"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    alwayslink = 1,
)

cc_test(
    name = "top_k_opencl_sampler_c_api_dlopen_test",
    srcs = ["top_k_opencl_sampler_c_api_dlopen_test.cc"],
    copts = ["-Wreturn-type-c-linkage"],
    data = [
        ":top_k_opencl_sampler_so",
        "//runtime/testdata",
    ],
    tags = [
        "manual",
        # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
        "noasan",
        "nomsan",
        "nopresubmit",
        "notsan",
        "requires-gpu-sm90",
    ],
    deps = [
        ":model_resources",
        ":model_resources_task",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/internal:litert_shared_library",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "@litert//litert/test:matchers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ],
)

genrule(
    name = "generate_webgpu_dawn_stub",
    srcs = ["@dawn//:include/dawn/webgpu.h"],
    outs = ["webgpu_dawn_stub.cc"],
    cmd = "sed -n -E 's/.*WGPU_EXPORT .* (wgpu[A-Za-z0-9_]+)\\(.*/extern \"C\" __declspec(dllexport) void \\1() {}/p' $(location @dawn//:include/dawn/webgpu.h) > $@",
    target_compatible_with = ["@platforms//os:windows"],
)

cc_binary(
    name = "stub/webgpu_dawn.dll",
    srcs = [":webgpu_dawn_stub.cc"],
    linkshared = 1,
    target_compatible_with = ["@platforms//os:windows"],
)

filegroup(
    name = "webgpu_dawn_interface_lib",
    srcs = [":stub/webgpu_dawn.dll"],
    output_group = "interface_library",
    target_compatible_with = ["@platforms//os:windows"],
)

cc_import(
    name = "webgpu_dawn_dll_import",
    interface_library = ":webgpu_dawn_interface_lib",
    system_provided = 1,
    target_compatible_with = ["@platforms//os:windows"],
)

cc_library(
    name = "webgpu_dawn_shared_lib",
    srcs = select({
        "//build_config:android_arm64": ["//prebuilt/android_arm64:libwebgpu_dawn.so"],
        "//build_config:android_x86_64": ["//prebuilt/android_x86_64:libwebgpu_dawn.so"],
        "//build_config:linux_x86_64": ["//prebuilt/linux_x86_64:libwebgpu_dawn.so"],
        "//build_config:linux_arm64": ["//prebuilt/linux_arm64:libwebgpu_dawn.so"],
        "@platforms//os:macos": ["//prebuilt/macos_arm64:libwebgpu_dawn.dylib"],
        "//conditions:default": [],
    }),
    data = select({
        "//build_config:linux_x86_64": ["//prebuilt/linux_x86_64:libwebgpu_dawn.so"],
        "//build_config:linux_arm64": ["//prebuilt/linux_arm64:libwebgpu_dawn.so"],
        "@platforms//os:macos": ["//prebuilt/macos_arm64:libwebgpu_dawn.dylib"],
        "@platforms//os:windows": ["//prebuilt/windows_x86_64:webgpu_dawn.dll"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "@platforms//os:ios": ["-Wl,-rpath,@loader_path"],
        "@platforms//os:windows": [
            "/DELAYLOAD:webgpu_dawn.dll",
            "delayimp.lib",
        ],
        "//conditions:default": [],
    }),
    deps = select({
        "@platforms//os:windows": [":webgpu_dawn_dll_import"],
        "//conditions:default": [],
    }),
)

TOP_K_WEBGPU_SAMPLER_COMMON_DEPS = [
    ":top_k_gpu_sampler",
    ":webgpu_dawn_shared_lib",
    "@com_google_absl//absl/base:nullability",
    "@com_google_absl//absl/cleanup",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/time",
    "@ml_drift//ml_drift/common:data_type",
    "@ml_drift//ml_drift/common:gpu_info",
    "@ml_drift//ml_drift/common:gpu_model",
    "@ml_drift//ml_drift/common:gpu_model_builder",
    "@ml_drift//ml_drift/common:model",
    "@ml_drift//ml_drift/common:model_hints",
    "@ml_drift//ml_drift/common:precision",
    "@ml_drift//ml_drift/common:shape",
    "@ml_drift//ml_drift/common:tensor",
    "@ml_drift//ml_drift/common:util",
    "@ml_drift//ml_drift/common/task:gpu_operation",
    "@ml_drift//ml_drift/common/task:tensor_desc",
    "@ml_drift//ml_drift/webgpu:compute_task",
    "@ml_drift//ml_drift/webgpu:environment",
    "@ml_drift//ml_drift/webgpu:execution_environment",
    "@ml_drift//ml_drift/webgpu:inference_context",
    "@ml_drift//ml_drift/webgpu:instance",
    "@ml_drift//ml_drift/webgpu:spatial_tensor",
    "@ml_drift//ml_drift/webgpu:webgpu_headers",
    "//runtime/executor:executor_settings_base",
    "//runtime/proto:sampler_params_cc_proto",
    "//runtime/util:litert_status_util",
    "@litert//litert/cc:litert_environment_api",
    "@litert//litert/cc:litert_environment_options",
    "@litert//litert/cc:litert_macros",
    "@litert//litert/cc:litert_tensor_buffer",
    "@litert//litert/cc/internal:litert_handle",
]

cc_library(
    name = "top_k_webgpu_sampler",
    srcs = ["top_k_webgpu_sampler.cc"],
    hdrs = ["top_k_webgpu_sampler.h"],
    deps = TOP_K_WEBGPU_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/status:status_macros",
        "@dawn//:webgpu_dawn",
    ],
)

# Static version of the WebGPU Sampler module for LiteRT LM.
cc_library(
    name = "top_k_webgpu_sampler_static",
    srcs = ["top_k_webgpu_sampler.cc"],
    hdrs = ["top_k_webgpu_sampler.h"],
    defines = ["LITERT_LM_USE_STATIC_LINKED_GPU_SAMPLER"],
    local_defines = select({
        "@litert//litert:litert_runtime_link_mode_dynamic": ["LITERT_SAMPLER_LINKED_DYNAMICALLY"],
        "//conditions:default": [],
    }),
    deps = TOP_K_WEBGPU_SAMPLER_COMMON_DEPS + ["@com_google_absl//absl/status:status_macros"],
    alwayslink = 1,
)

# This module requires the shared library of LiteRT runtime `libLiteRt.so`.
cc_library(
    name = "top_k_webgpu_sampler_runtimecapi",
    srcs = ["top_k_webgpu_sampler.cc"],
    hdrs = ["top_k_webgpu_sampler.h"],
    features = ["-layering_check"],
    local_defines = ["LITERT_SAMPLER_LINKED_DYNAMICALLY"],
    visibility = ["//visibility:private"],
    deps = TOP_K_WEBGPU_SAMPLER_COMMON_DEPS + [
        "@com_google_absl//absl/status:status_macros",
        "@litert//litert/c/internal:litert_runtime_builtin_none",
    ],
)

cc_test(
    name = "top_k_webgpu_sampler_test",
    srcs = ["top_k_webgpu_sampler_test.cc"],
    data = ["//runtime/testdata"],
    tags = [
        "manual",
        "nopresubmit",
        "requires-gpu-sm90-full",
    ],
    deps = [
        ":model_resources",
        ":model_resources_task",
        ":top_k_webgpu_sampler",
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/test:matchers",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ] + select({
        "@platforms//os:linux": [
            "@litert//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator",
        ],
        "//conditions:default": [
            "//runtime/executor:default_static_gpu_accelerator",
        ],
    }),
)

cc_test(
    name = "top_k_webgpu_sampler_c_api_test",
    srcs = ["top_k_webgpu_sampler_c_api_test.cc"],
    copts = ["-Wreturn-type-c-linkage"],
    data = ["//runtime/testdata"],
    tags = [
        "manual",
        "nopresubmit",
        "requires-gpu-sm90-full",
    ],
    deps = [
        ":model_resources",
        ":model_resources_task",
        ":top_k_webgpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/test:matchers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ] + select({
        "@platforms//os:linux": [
            "@litert//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator",
        ],
        "//conditions:default": [
            "//runtime/executor:default_static_gpu_accelerator",
        ],
    }),
)

cc_shared_library(
    name = "top_k_webgpu_sampler_so",
    shared_lib_name = "libLiteRtTopKWebGpuSampler.so",
    tags = ["manual"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    user_link_flags = select({
        "@platforms//os:android": [
            "-landroid",
            "-Wl,-soname=libLiteRtTopKWebGpuSampler.so",
            # Align PT_LOAD segments and pad GNU_RELRO to a 16 KB boundary.
            # See https://developer.android.com/guide/practices/page-sizes#relro-flag.
            "-Wl,-z,max-page-size=16384",
            "-Wl,-z,common-page-size=16384",
        ],
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["-Wl,-soname=libLiteRtTopKWebGpuSampler.so"],
    }),
    deps = [":top_k_webgpu_sampler_runtimecapi"],
)

macos_dylib(
    name = "build_top_k_webgpu_sampler_dylib",
    exported_symbols_lists = ["webgpu_exported_symbols.lds"],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libLiteRtTopKWebGpuSampler.dylib",
    ],
    minimum_os_version = LITERT_MIN_MACOS_VERSION,
    tags = ["manual"],
    visibility = ["//visibility:private"],
    deps = [":top_k_webgpu_sampler_runtimecapi"],
)

copy_file(
    name = "top_k_webgpu_sampler_dylib",
    src = ":build_top_k_webgpu_sampler_dylib",
    out = "libLiteRtTopKWebGpuSampler.dylib",
    tags = ["manual"],
)

macos_dylib(
    name = "build_top_k_metal_sampler_dylib",
    exported_symbols_lists = ["metal_exported_symbols.lds"],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libLiteRtTopKMetalSampler.dylib",
    ],
    minimum_os_version = LITERT_MIN_MACOS_VERSION,
    tags = ["manual"],
    visibility = ["//visibility:private"],
    deps = [":top_k_metal_sampler_runtimecapi"],
)

copy_file(
    name = "top_k_metal_sampler_dylib",
    src = ":build_top_k_metal_sampler_dylib",
    out = "libLiteRtTopKMetalSampler.dylib",
    tags = ["manual"],
)

ios_framework(
    name = "build_top_k_metal_sampler_framework",
    bundle_id = "com.google.litert_lm.runtime",
    bundle_name = "libLiteRtTopKMetalSampler.dylib",
    exported_symbols_lists = ["metal_exported_symbols.lds"],
    families = [
        "iphone",
        "ipad",
    ],
    # copybara:uncomment google_create_srl_bindings_module = False,
    infoplists = ["LiteRtTopKMetalSamplerInfo.plist"],
    linkopts = [
        "-undefined dynamic_lookup",
        "-install_name @rpath/libLiteRtTopKMetalSampler.dylib",
    ],
    minimum_os_version = LITERT_MIN_IOS_VERSION,
    tags = [
        "manual",
        "nofastbuild",
    ],
    visibility = ["//visibility:private"],
    deps = [":top_k_metal_sampler_runtimecapi"],
)

# Target for windows libLiteRtTopKWebGpuSampler.dll.
cc_binary(
    name = "libLiteRtTopKWebGpuSampler",
    linkshared = 1,
    tags = [
        "nobuilder",
        "notap",
        "windows",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    visibility = ["//visibility:private"],
    win_def_file = ":windows_exported_symbols.def",
    deps = [":top_k_webgpu_sampler_runtimecapi"] + select({
        "@platforms//os:windows": [
            "@litert//litert/c:litert_runtime_c_api_import",
        ],
        "//conditions:default": [],
    }),
)

alias(
    name = "top_k_webgpu_sampler_dll",
    actual = ":libLiteRtTopKWebGpuSampler",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "top_k_webgpu_sampler_shared_lib",
    srcs = [":top_k_webgpu_sampler_so"],
    linkstatic = 1,
    tags = ["manual"],
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    alwayslink = 1,
)

# Need extra flag to run this test: --define=litert_runtime_link_mode=dynamic.
cc_test(
    name = "top_k_webgpu_sampler_c_api_dlopen_test",
    srcs = ["top_k_webgpu_sampler_c_api_dlopen_test.cc"],
    copts = ["-Wreturn-type-c-linkage"],
    data = [
        ":top_k_webgpu_sampler_so",
        "//runtime/testdata",
    ] + select({
        "@litert//litert:litert_runtime_link_mode_dynamic": [
            "@litert//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_so",
        ],
        "//conditions:default": [],
    }),
    defines = ["litert_runtime_link_mode=dynamic"],
    tags = [
        "manual",
        "norapid",
        # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-sm90-full",
        "notap",  # remove when dynamic linking is enabled
    ],
    deps = [
        ":model_resources",
        ":model_resources_task",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/internal:litert_shared_library",
        "@litert//litert/cc/options:litert_google_tensor_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_intel_openvino_options",
        "@litert//litert/cc/options:litert_mediatek_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "@litert//litert/cc/options:litert_samsung_options",
        "@litert//litert/test:matchers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ] + select({
        "@litert//litert:litert_runtime_link_mode_dynamic": [
            "@litert//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_runtimecapi",
        ],
        "//conditions:default": [
            "//runtime/executor:default_static_gpu_accelerator",
        ],
    }),
)

cc_library(
    name = "prompt_template",
    srcs = select({
        "//build_config:disable_minijinja_template": [
            "prompt_template_noop.cc",
        ],
        "//conditions:default": ["prompt_template.cc"],
    }),
    hdrs = ["prompt_template.h"],
    linkopts = select({
        "@platforms//os:windows": ["ntdll.lib"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
    ] + select({
        "//build_config:disable_minijinja_template": [],
        "//conditions:default": [
            "@com_google_absl//absl/strings",
            "//runtime/components/rust:minijinja_template_cpp",
            "@com_googlesource_code_re2//:re2",
        ],
    }),
)

cc_library(
    name = "prompt_template_noop",
    srcs = ["prompt_template_noop.cc"],
    hdrs = ["prompt_template.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "prompt_template_test",
    srcs = ["prompt_template_test.cc"],
    data = [
        "//models/qwen3:chat_template.jinja",
        "//runtime/components/testdata",
    ],
    deps = [
        ":prompt_template",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "prompt_template_noop_test",
    srcs = ["prompt_template_noop_test.cc"],
    deps = [
        ":prompt_template_noop",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "lora",
    srcs = ["lora.cc"],
    hdrs = ["lora.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # TODO: b/467362164 Move tflite_lora_utils to an OSS directory to support open sourcing LoRA.
        "//runtime/util:litert_status_util",
        "//runtime/util:lora_data",
        "//runtime/util:lora_util",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "lora_test",
    srcs = ["lora_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":lora",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
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
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor:executor_settings_base",
        "//runtime/util:litert_status_util",
        "//runtime/util:lora_data",
    ],
)

cc_library(
    name = "model_resources_streaming",
    srcs = ["model_resources_streaming.cc"],
    hdrs = ["model_resources_streaming.h"],
    deps = [
        ":model_resources",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@litert//litert/c:litert_tensor_buffer_types",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_model",
        "//runtime/proto:embedding_metadata_cc_proto",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:data_stream",
        "//runtime/util:scoped_file",
        "//support/tokenizer",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "model_resources_streaming_test",
    srcs = ["model_resources_streaming_test.cc"],
    deps = [
        ":model_resources_streaming",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@flatbuffers",
        "//runtime/proto:embedding_metadata_cc_proto",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:data_stream",
        "//runtime/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
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

alias(
    name = "tokenizer",
    actual = "//support/tokenizer",
)

alias(
    name = "sentencepiece_tokenizer",
    actual = "//support/tokenizer:sentencepiece_tokenizer",
)

alias(
    name = "huggingface_tokenizer",
    actual = "//support/tokenizer:huggingface_tokenizer",
)
