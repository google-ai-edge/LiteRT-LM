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

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

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
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
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
        "@ml_drift//ml_drift/common/kernels:fully_connected",
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
        "//runtime/util:litert_status_util",
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
        "@ml_drift//ml_drift/common:types",
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
        "@com_google_absl//absl/log:absl_check",
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

cc_test(
    name = "tasks_and_byom_test",
    srcs = ["tasks_and_byom_test.cc"],
    deps = [
        ":audio_encoder",
        ":llm",
        ":transformer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:precision",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift/byom:byom_factory",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:add_with_overflow",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:apply_attention_mask",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:apply_fire_bias",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:flash_decoding_attention",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:fused_rope",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:gelu7",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:mul_with_overflow",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:rotqat_rotate",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:basic_gpu_tensor",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_dummy_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:llm_inference_engine_cc_proto",
        "//runtime/executor/gpu_artisan/proto:transformer_params_cc_proto",
        "//runtime/util:test_utils",
    ],
)
