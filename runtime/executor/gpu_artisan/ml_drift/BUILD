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
# [Google-internal load of `cc_test`]
# [Google-internal load of `objc_library`]
load("@bazel_skylib//lib:selects.bzl", "selects")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# Workaround bug in Bazel before 8.0 where --cxxopt did not apply to objc++ compilations.
CXX20_BAZEL_ONLY_COPTS = [
    "-std=c++20",
]

DEFAULT_COPTS = CXX20_BAZEL_ONLY_COPTS

cc_library(
    name = "name_check",
    hdrs = ["name_check.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "runtime_params",
    hdrs = ["runtime_params.h"],
)

cc_library(
    name = "group_size",
    srcs = ["group_size.cc"],
    hdrs = ["group_size.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:util",
    ],
)

cc_test(
    name = "group_size_test",
    srcs = ["group_size_test.cc"],
    deps = [
        ":group_size",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "speculative_utils",
    srcs = ["speculative_utils.cc"],
    hdrs = ["speculative_utils.h"],
    deps = [
        ":runtime_params",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
    ],
)

cc_test(
    name = "speculative_utils_test",
    srcs = ["speculative_utils_test.cc"],
    deps = [
        ":speculative_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "speculative_shared_state",
    hdrs = ["speculative_shared_state.h"],
    deps = [
        "@com_google_absl//absl/time",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
    ],
)

cc_library(
    name = "mtp_binding",
    srcs = ["mtp_binding.cc"],
    hdrs = ["mtp_binding.h"],
    deps = [
        ":speculative_shared_state",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:tensor_handle",
    ],
)

cc_test(
    name = "mtp_binding_test",
    srcs = ["mtp_binding_test.cc"],
    deps = [
        ":mtp_binding",
        ":speculative_shared_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:tensor_handle",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "transformer",
    srcs = [
        "transformer.cc",
        "transformer_transform_qkv.cc",
    ],
    hdrs = ["transformer.h"],
    deps = [
        ":name_check",
        ":runtime_params",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:gpu_model_builder_moe_util",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/kernels:elementwise",
        "@ml_drift//ml_drift/common/kernels:padding",
        "@ml_drift//ml_drift/common/kernels:reduce",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_object_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:add_with_overflow",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:apply_attention_mask",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:apply_fire_bias",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:flash_decoding_attention",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:fused_rope",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:mul_with_overflow",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:rotqat_rotate",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:embedding_updater",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:placeholder_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:transformer_params_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "audio_encoder",
    srcs = ["audio_encoder.cc"],
    hdrs = ["audio_encoder.h"],
    deps = [
        ":runtime_params",
        ":transformer",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/kernels:elementwise",
        "@ml_drift//ml_drift/common/kernels:reduce",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
    ],
)

cc_library(
    name = "llm",
    srcs = ["llm.cc"],
    hdrs = ["llm.h"],
    deps = [
        ":audio_encoder",
        ":transformer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_object_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:llm_inference_engine_cc_proto",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:transformer_params_cc_proto",
    ],
)

cc_library(
    name = "gpu_config_converters",
    srcs = ["gpu_config_converters.cc"],
    hdrs = ["gpu_config_converters.h"],
    deps = [
        ":audio_encoder",
        ":llm",
        ":transformer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:shape",
        "//runtime/executor/gpu_artisan/proto:audio_encoder_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:transformer_params_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "gpu_config_converters_test",
    srcs = ["gpu_config_converters_test.cc"],
    deps = [
        ":audio_encoder",
        ":gpu_config_converters",
        ":transformer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/executor/gpu_artisan/proto:audio_encoder_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:transformer_params_cc_proto",
        "//runtime/executor/gpu_artisan/proto:vision_encoder_params_cc_proto",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "gpu_model_info_generator",
    srcs = ["gpu_model_info_generator.cc"],
    hdrs = ["gpu_model_info_generator.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:precision",
        "//runtime/executor/gpu_artisan/proto:llm_inference_engine_cc_proto",
    ],
)

cc_test(
    name = "gpu_model_info_generator_test",
    srcs = ["gpu_model_info_generator_test.cc"],
    deps = [
        ":gpu_model_info_generator",
        "@com_google_googletest//:gtest_main",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:precision",
        "//runtime/executor/gpu_artisan/proto:llm_inference_engine_cc_proto",
    ],
)

cc_library(
    name = "executor_factory",
    hdrs = ["executor_factory.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:executor",
    ],
)

cc_library(
    name = "litert_thread_pool_executor",
    srcs = ["litert_thread_pool_executor.cc"],
    hdrs = ["litert_thread_pool_executor.h"],
    deps = [
        ":executor_factory",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:executor",
        "//runtime/framework:thread_options",
        "//runtime/framework:threadpool",
    ],
)

cc_test(
    name = "litert_thread_pool_executor_test",
    srcs = ["litert_thread_pool_executor_test.cc"],
    deps = [
        ":litert_thread_pool_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@ml_drift//ml_drift/common:executor",
        "//runtime/util:test_utils",
    ],
)

config_setting(
    name = "force_webgpu_runner",
    define_values = {"enable_webgpu_runner": "true"},
)

config_setting(
    name = "force_mldrift_profiling",
    define_values = {"enable_mldrift_profiling": "true"},
)

config_setting(
    name = "web_llm_webgpu_debug_mldrift_tensors",
    define_values = {"web_llm_webgpu_debug_mldrift_tensors": "true"},
)

config_setting(
    name = "llm_metal_debug_mldrift_tensors",
    define_values = {"llm_metal_debug_mldrift_tensors": "true"},
)

cc_library(
    name = "drafter",
    hdrs = ["drafter.h"],
    deps = [
        ":llm",
        ":speculative_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "llm_gpu_runner",
    hdrs = ["llm_gpu_runner.h"],
    defines = select({
        ":force_mldrift_profiling": ["ML_DRIFT_PROFILING"],
        "//conditions:default": [],
    }),
    deps = [
        ":executor_factory",
        ":llm",
        ":speculative_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
    ] + select({
        ":force_mldrift_profiling": [
            "@ml_drift//ml_drift/cl:environment",
            "@ml_drift//ml_drift/cl:inference_context",
        ],
        "//conditions:default": [],
    }) + select({
        "@platforms//os:emscripten": [
            "@ml_drift//ml_drift/webgpu:webgpu_headers",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "llm_gpu_drafter",
    srcs = ["llm_gpu_drafter.cc"],
    hdrs = ["llm_gpu_drafter.h"],
    deps = [
        ":drafter",
        ":llm",
        ":llm_gpu_runner",
        ":speculative_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "heuristic_drafter",
    srcs = ["heuristic_drafter.cc"],
    hdrs = ["heuristic_drafter.h"],
    deps = [
        ":drafter",
        ":llm",
        ":speculative_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "heuristic_drafter_test",
    srcs = ["heuristic_drafter_test.cc"],
    deps = [
        ":heuristic_drafter",
        ":speculative_utils",
        "@com_google_googletest//:gtest_main",
        "//runtime/util:test_utils",
    ],
)

# buildifier: disable=no-hdrs-no-alwayslink
cc_library(
    name = "llm_speculative_runner",
    srcs = ["llm_speculative_runner.cc"],
    deps = [
        ":drafter",
        ":llm",
        ":llm_gpu_runner",
        ":speculative_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/util:litert_status_util",
    ] + select({
        ":force_mldrift_profiling": [
            "@ml_drift//ml_drift/cl:environment",
            "@ml_drift//ml_drift/cl:inference_context",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "llm_speculative_runner_test",
    srcs = ["llm_speculative_runner_test.cc"],
    deps = [
        ":llm",
        ":llm_gpu_drafter",
        ":llm_gpu_runner",
        ":llm_speculative_runner",  # buildcleaner: keep
        ":speculative_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_webgpu_debug",
    srcs = ["llm_webgpu_debug.cc"],
    hdrs = ["llm_webgpu_debug.h"],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:inference_context",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
    ],
)

cc_library(
    name = "llm_webgpu",
    srcs = ["llm_webgpu.cc"],
    hdrs = ["llm_webgpu.h"],
    local_defines = select({
        "//conditions:default": [],
        ":web_llm_webgpu_debug_mldrift_tensors": [
            "WEB_LLM_WEBGPU_DEBUG_MLDRIFT_TENSORS=1",
        ],
    }),
    deps = [
        ":audio_encoder",
        ":executor_factory",
        ":group_size",
        ":llm",
        ":name_check",
        ":runtime_params",
        ":speculative_utils",
        ":transformer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:executor",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:buffer",
        "@ml_drift//ml_drift/webgpu:compute_task",
        "@ml_drift//ml_drift/webgpu:environment",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:inference_context",
        "@ml_drift//ml_drift/webgpu:memory_manager",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
        "@ml_drift//ml_drift/webgpu:webgpu_api_util",
        "@ml_drift//ml_drift/webgpu:webgpu_headers",
        "//runtime/executor/gpu_artisan/ml_drift/byom:byom_factory",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:embedding_updater",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:gpu_optimized_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_file_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:placeholder_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//runtime/util:litert_status_util",
    ] + select({
        "//conditions:default": [],
        ":web_llm_webgpu_debug_mldrift_tensors": [":llm_webgpu_debug"],
    }),
)

# buildifier: disable=no-hdrs-no-alwayslink
cc_library(
    name = "llm_webgpu_runner",
    srcs = ["llm_webgpu_runner.cc"],
    deps = [
        ":executor_factory",
        ":llm",
        ":llm_gpu_runner",
        ":llm_webgpu",
        ":speculative_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:executor",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/webgpu:execution_environment",
        "@ml_drift//ml_drift/webgpu:spatial_tensor",
        "@ml_drift//ml_drift/webgpu:webgpu_api_util",
        "@ml_drift//ml_drift/webgpu:webgpu_headers",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "llm_opencl",
    srcs = ["llm_opencl.cc"],
    hdrs = ["llm_opencl.h"],
    deps = [
        ":audio_encoder",
        ":group_size",
        ":llm",
        ":name_check",
        ":runtime_params",
        ":transformer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/cl:buffer",
        "@ml_drift//ml_drift/cl:cl_command_queue",
        "@ml_drift//ml_drift/cl:cl_context",
        "@ml_drift//ml_drift/cl:cl_errors",
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
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift/byom:byom_factory",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:embedding_updater",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_file_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//runtime/util:litert_status_util",
        "//support/util:memory_mapped_file",
    ],
)

# buildifier: disable=no-hdrs-no-alwayslink
cc_library(
    name = "llm_opencl_runner",
    srcs = ["llm_opencl_runner.cc"],
    defines = select({
        ":force_mldrift_profiling": ["ML_DRIFT_PROFILING"],
        "//conditions:default": [],
    }),
    deps = [
        ":drafter",
        ":executor_factory",
        ":llm",
        ":llm_gpu_runner",
        ":llm_opencl",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
        "@ml_drift//ml_drift/cl:tensor",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "Metal",
    linkopts = ["-framework Metal"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

objc_library(
    name = "llm_metal_debug",
    srcs = ["llm_metal_debug.cc"],
    hdrs = ["llm_metal_debug.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    tags = ["manual"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":Metal",
        ":transformer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/metal:inference_context",
        "@ml_drift//ml_drift/metal:metal_spatial_tensor",
        "//runtime/util:litert_status_util",
    ],
)

objc_library(
    name = "llm_metal",
    srcs = ["llm_metal.cc"],
    hdrs = ["llm_metal.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ] + select({
        "//conditions:default": [],
        ":llm_metal_debug_mldrift_tensors": [
            "-DLLM_METAL_DEBUG_MLDRIFT_TENSORS=1",
        ],
    }),
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":Metal",
        ":audio_encoder",
        ":group_size",
        ":llm",
        ":name_check",
        ":runtime_params",
        ":speculative_utils",
        ":transformer",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:model_hints",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/metal:buffer",
        "@ml_drift//ml_drift/metal:common",
        "@ml_drift//ml_drift/metal:compute_task",
        "@ml_drift//ml_drift/metal:environment",
        "@ml_drift//ml_drift/metal:inference_context",
        "@ml_drift//ml_drift/metal:metal_device",
        "@ml_drift//ml_drift/metal:metal_spatial_tensor",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/ml_drift/byom:byom_factory",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:embedding_updater",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_file_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "//conditions:default": [],
        ":llm_metal_debug_mldrift_tensors": [":llm_metal_debug"],
    }),
)

objc_library(
    name = "llm_metal_runner",
    srcs = ["llm_metal_runner.mm"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
        "-fobjc-arc",
    ],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":Metal",
        ":drafter",
        ":executor_factory",
        ":llm_gpu_runner",
        ":llm_metal",
        ":speculative_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:status_macros",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/metal:common",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
    ],
)

cc_library(
    name = "llm_native_runner",
    deps = selects.with_or({
        (
            "@platforms//os:ios",
            "@platforms//os:macos",
            "@platforms//os:tvos",
            "@platforms//os:visionos",
            "@platforms//os:watchos",
        ): [":llm_metal_runner"],  # buildcleaner: keep
        "//conditions:default": [":llm_opencl_runner"],  # buildcleaner: keep
    }),
)

cc_library(
    name = "llm_gpu_runner_impl",
    deps = [
        ":llm_gpu_runner",
    ] + selects.with_or({
        (
            ":force_webgpu_runner",
            "@platforms//os:linux",
            "@platforms//os:windows",
            "@platforms//os:emscripten",
        ): [
            ":llm_speculative_runner",  # buildcleaner: keep
            ":llm_webgpu_runner",  # buildcleaner: keep
        ],
        "//conditions:default": [":llm_native_runner"],  # buildcleaner: keep
    }),
)
