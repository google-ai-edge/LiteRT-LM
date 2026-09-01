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
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "audio_executor_settings",
    srcs = ["audio/audio_executor_settings.cc"],
    hdrs = ["audio/audio_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "audio_executor_settings_test",
    srcs = ["audio/audio_executor_settings_test.cc"],
    deps = [
        ":audio_executor_settings",
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "vision_executor_settings",
    srcs = ["vision/vision_executor_settings.cc"],
    hdrs = ["vision/vision_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "vision_executor_settings_test",
    srcs = ["vision/vision_executor_settings_test.cc"],
    deps = [
        ":executor_settings_base",
        ":vision_executor_settings",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "executor_settings_base",
    srcs = ["executor_settings_base.cc"],
    hdrs = ["executor_settings_base.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:data_stream",
        "//runtime/util:file_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "executor_settings_base_test",
    srcs = ["executor_settings_base_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:file_data_stream",
        "//runtime/util:file_util",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "litert_compiled_model_executor_utils",
    srcs = ["litert_compiled_model_executor_utils.cc"],
    hdrs = ["litert_compiled_model_executor_utils.h"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_io_types",
        ":llm_executor_settings",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:model_resources_task",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/components/embedding_lookup:embedding_lookup_text",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:file_format_util",
        "//runtime/util:file_util",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:tensor_buffer_util",
        "//schema/core:litertlm_header_schema",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "@litert//tflite/types:half",
    ],
)

# Flags to configure the static GPU accelerator backend for Apple targets (iOS and macOS).
# By default, iOS uses Metal ("metal") and macOS uses WebGPU ("webgpu").
# Pass --//runtime/executor:gpu_accelerator=webgpu or =metal to override.
string_flag(
    name = "gpu_accelerator",
    build_setting_default = "default",
    values = [
        "default",
        "metal",
        "webgpu",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "gpu_accelerator_webgpu_setting",
    flag_values = {
        ":gpu_accelerator": "webgpu",
    },
)

config_setting(
    name = "gpu_accelerator_webgpu_define",
    define_values = {"LITERT_GPU_ACCELERATOR": "webgpu"},
)

selects.config_setting_group(
    name = "gpu_accelerator_webgpu_requested",
    match_any = [
        ":gpu_accelerator_webgpu_setting",
        ":gpu_accelerator_webgpu_define",
    ],
)

config_setting(
    name = "gpu_accelerator_metal_setting",
    flag_values = {
        ":gpu_accelerator": "metal",
    },
)

config_setting(
    name = "gpu_accelerator_metal_define",
    define_values = {"LITERT_GPU_ACCELERATOR": "metal"},
)

selects.config_setting_group(
    name = "gpu_accelerator_metal_requested",
    match_any = [
        ":gpu_accelerator_metal_setting",
        ":gpu_accelerator_metal_define",
    ],
)

selects.config_setting_group(
    name = "ios_webgpu",
    match_all = [
        "@platforms//os:ios",
        ":gpu_accelerator_webgpu_requested",
    ],
)

selects.config_setting_group(
    name = "macos_metal",
    match_all = [
        "@platforms//os:macos",
        ":gpu_accelerator_metal_requested",
    ],
)

cc_library(
    name = "default_static_gpu_accelerator",
    deps = select({
        # If CAPI is linked dynamically, need to link the accelerator dynamically as well.
        "@litert//litert:litert_runtime_link_mode_dynamic": [],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "llm_litert_compiled_model_cache_utils",
    srcs = ["llm_litert_compiled_model_cache_utils.cc"],
    hdrs = ["llm_litert_compiled_model_cache_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_litert_compiled_model_cache_utils_test",
    srcs = ["llm_litert_compiled_model_cache_utils_test.cc"],
    deps = [
        ":llm_litert_compiled_model_cache_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_litert_compiled_model_executor",
    srcs = ["llm_litert_compiled_model_executor.cc"],
    hdrs = ["llm_litert_compiled_model_executor.h"],
    deps = [
        ":common_utils",
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        ":llm_executor_settings_utils",
        ":llm_litert_compiled_model_cache_utils",
        ":llm_litert_mtp_drafter",
        ":llm_processed_context",
        ":state_interface",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",  # buildcleaner: keep
        "@litert//litert/c/internal:litert_external_litert_buffer_context",  # buildcleaner: keep
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_model_types",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_profiler",  # buildcleaner: keep
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/internal:litert_handle",  # buildcleaner: keep
        "//runtime/components:lora_manager",
        "//runtime/components:model_resources",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/executor/litert:state",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:log_tensor_buffer",
        "//runtime/util:lora_util",
        "//runtime/util:tensor_buffer_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "llm_litert_compiled_model_executor_test",
    size = "large",
    srcs = ["llm_litert_compiled_model_executor_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notap",
        "notsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":default_static_gpu_accelerator",
        ":executor_settings_base",
        ":llm_executor_io_types",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:sampler",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_processed_context",
    hdrs = ["llm_processed_context.h"],
    deps = [
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":state_interface",
    ],
)

cc_library(
    name = "llm_litert_compiled_model_executor_factory",
    srcs = ["llm_litert_compiled_model_executor_factory.cc"],
    hdrs = ["llm_litert_compiled_model_executor_factory.h"],
    defines = select({
        "@platforms//os:ios": ["LITERT_DISABLE_NPU"],
        "//conditions:default": [],
    }),
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_model_types",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "//runtime/components:model_resources",
        "//runtime/proto:executor_metadata_cc_proto",
        "//runtime/util:litert_status_util",
    ] + select({
        "@platforms//os:ios": [],
        "//conditions:default": [":llm_litert_npu_compiled_model_executor"],
    }),
)

cc_test(
    name = "llm_litert_compiled_model_executor_factory_test",
    srcs = ["llm_litert_compiled_model_executor_factory_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor_factory",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "fake_llm_executor",
    srcs = ["fake_llm_executor.cc"],
    hdrs = ["fake_llm_executor.h"],
    deps = [
        ":executor_settings_base",
        ":llm_executor",
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "fake_llm_executor_test",
    srcs = ["fake_llm_executor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":fake_llm_executor",
        ":llm_executor_io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor",
    hdrs = ["llm_executor.h"],
    deps = [
        ":llm_executor_base",
    ],
)

cc_library(
    name = "llm_executor_base",
    hdrs = ["llm_executor_base.h"],
    deps = [
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "llm_litert_npu_compiled_model_executor",
    srcs = ["llm_litert_npu_compiled_model_executor.cc"],
    hdrs = ["llm_litert_npu_compiled_model_executor.h"],
    deps = [
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        ":llm_processed_context",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/c:litert_model_types",
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
        "@litert//litert/cc/options:litert_google_tensor_options",
        "@litert//litert/cc/options:litert_mediatek_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "//runtime/components:model_resources",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/executor/litert:legacy_map_state",
        "//runtime/executor/npu:llm_litert_npu_compiled_model_executor_utils",
        "//runtime/executor/npu:llm_litert_npu_embedder",
        "//runtime/executor/npu:llm_litert_npu_kv_cache",
        "//runtime/executor/npu:llm_litert_npu_mask",
        "//runtime/executor/npu:llm_litert_npu_rope",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:tensor_buffer_util",
    ],
)

cc_library(
    name = "llm_executor_processed_tokens",
    srcs = ["llm_executor_processed_tokens.cc"],
    hdrs = ["llm_executor_processed_tokens.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "llm_executor_processed_tokens_test",
    srcs = ["llm_executor_processed_tokens_test.cc"],
    deps = [
        ":llm_executor_processed_tokens",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor_settings",
    srcs = ["llm_executor_settings.cc"],
    hdrs = ["llm_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:logging",
    ],
)

cc_test(
    name = "llm_executor_settings_test",
    srcs = ["llm_executor_settings_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor_settings_utils",
    srcs = ["llm_executor_settings_utils.cc"],
    hdrs = ["llm_executor_settings_utils.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_settings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/util:file_util",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_executor_settings_utils_test",
    srcs = ["llm_executor_settings_utils_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        ":llm_executor_settings_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@litert//litert/c/options:litert_gpu_options",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_options",
        "@litert//litert/test:matchers",
        "//runtime/util:file_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor_io_types",
    srcs = ["llm_executor_io_types.cc"],
    hdrs = ["llm_executor_io_types.h"],
    deps = [
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/util:logging_tensor_buffer",
    ],
)

cc_test(
    name = "llm_executor_io_types_test",
    srcs = ["llm_executor_io_types_test.cc"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_tensor_buffer_types",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "vision_executor",
    hdrs = ["vision/vision_executor.h"],
    deps = [":vision_executor_base"],
)

cc_library(
    name = "vision_executor_base",
    hdrs = ["vision/vision_executor_base.h"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/engine:io_types",
    ],
)

cc_library(
    name = "vision_litert_compiled_model_executor",
    srcs = ["vision_litert_compiled_model_executor.cc"],
    hdrs = ["vision_litert_compiled_model_executor.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_io_types",
        ":vision_executor",
        ":vision_executor_settings",
        ":vision_executor_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_common",
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
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_google_tensor_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "//runtime/components:model_resources",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "vision_litert_compiled_model_executor_test",
    srcs = ["vision_litert_compiled_model_executor_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":vision_executor_settings",
        ":vision_litert_compiled_model_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "audio_executor",
    hdrs = ["audio/audio_executor.h"],
    deps = [":audio_executor_base"],
)

cc_library(
    name = "audio_executor_base",
    hdrs = ["audio/audio_executor_base.h"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_io_types",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/engine:io_types",
    ],
)

cc_library(
    name = "embedding_executor_settings",
    srcs = ["embedding/embedding_executor_settings.cc"],
    hdrs = ["embedding/embedding_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "embedding_executor_settings_test",
    srcs = ["embedding/embedding_executor_settings_test.cc"],
    deps = [
        ":embedding_executor_settings",
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "embedding_executor_base",
    hdrs = ["embedding/embedding_executor_base.h"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
    ],
)

cc_library(
    name = "embedding_litert_compiled_model_executor",
    srcs = ["embedding_litert_compiled_model_executor.cc"],
    hdrs = ["embedding_litert_compiled_model_executor.h"],
    deps = [
        ":embedding_executor_base",
        ":embedding_executor_settings",
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_io_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/options:litert_google_tensor_options",
        "@litert//litert/cc/options:litert_qualcomm_options",
        "//runtime/components:model_resources",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "embedding_litert_compiled_model_executor_test",
    srcs = ["embedding_litert_compiled_model_executor_test.cc"],
    deps = [
        ":embedding_executor_base",
        ":embedding_executor_settings",
        ":embedding_litert_compiled_model_executor",
        ":executor_settings_base",
        ":llm_executor_io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
        "@litert//litert/c:litert_tensor_buffer_types",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "audio_litert_compiled_model_executor",
    srcs = ["audio_litert_compiled_model_executor.cc"],
    hdrs = ["audio_litert_compiled_model_executor.h"],
    deps = [
        ":audio_executor",
        ":audio_executor_settings",
        ":audio_executor_utils",
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_io_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/options:litert_cpu_options",
        "@litert//litert/cc/options:litert_google_tensor_options",
        "@litert//litert/cc/options:litert_gpu_options",
        "//runtime/components:lora_manager",
        "//runtime/components:model_resources",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:tensor_buffer_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "audio_litert_compiled_model_executor_test",
    srcs = ["audio_litert_compiled_model_executor_test.cc"],
    data = ["//runtime/testdata"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":audio_executor_settings",
        ":audio_litert_compiled_model_executor",
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "audio_executor_utils",
    srcs = ["audio/audio_executor_utils.cc"],
    hdrs = ["audio/audio_executor_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "audio_executor_utils_test",
    srcs = ["audio/audio_executor_utils_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":audio_executor_utils",
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "vision_executor_utils",
    srcs = ["vision/vision_executor_utils.cc"],
    hdrs = ["vision/vision_executor_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "//runtime/components:model_resources",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "vision_executor_utils_test",
    srcs = ["vision/vision_executor_utils_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":vision_executor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "model_signature_utils",
    srcs = ["model_signature_utils.cc"],
    hdrs = ["model_signature_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "//runtime/components:model_resources",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "model_signature_utils_test",
    srcs = ["model_signature_utils_test.cc"],
    deps = [
        ":model_signature_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",  # buildcleaner: keep
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_library(
    name = "magic_number_configs_helper",
    srcs = ["magic_number_configs_helper.cc"],
    hdrs = ["magic_number_configs_helper.h"],
    deps = [
        ":llm_executor_settings",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_environment_options",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc/options:litert_magic_number_options",
        "//runtime/components:model_resources",
    ],
)

cc_test(
    name = "magic_number_configs_helper_test",
    srcs = ["magic_number_configs_helper_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        ":magic_number_configs_helper",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",  # buildcleaner: keep
    ],
)

cc_library(
    name = "llm_litert_mtp_drafter",
    srcs = ["llm_litert_mtp_drafter.cc"],
    hdrs = ["llm_litert_mtp_drafter.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_settings",
        ":llm_executor_settings_utils",
        ":state_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model_types",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/components:model_resources",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:logit_mask",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/executor/litert:state",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "llm_litert_mtp_drafter_test",
    srcs = ["llm_litert_mtp_drafter_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        ":llm_litert_mtp_drafter",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

# ==================================================================================================
# Interfaces
# ==================================================================================================
cc_library(
    name = "state_interface",
    hdrs = ["state_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "llm_executor_interface",
    hdrs = ["llm_executor_interface.h"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_io_types",
        ":state_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

# ==================================================================================================
# Common Utils shared across all implementations
# ==================================================================================================

cc_library(
    name = "common_utils",
    srcs = ["common_utils.cc"],
    hdrs = ["common_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "common_utils_test",
    srcs = ["common_utils_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":common_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
        "//runtime/util:test_utils",
    ],
)
