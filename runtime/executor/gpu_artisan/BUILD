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
    name = "llm_gpu_artisan_processed_context",
    hdrs = ["llm_gpu_artisan_processed_context.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_processed_tokens",
        "//runtime/executor/gpu_artisan/ml_drift:llm",
    ],
)

cc_library(
    name = "gpu_artisan_executor_utils",
    srcs = ["gpu_artisan_executor_utils.cc"],
    hdrs = ["gpu_artisan_executor_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_test(
    name = "gpu_artisan_executor_utils_test",
    srcs = ["gpu_artisan_executor_utils_test.cc"],
    deps = [
        ":gpu_artisan_executor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_gpu_artisan_executor",
    srcs = ["llm_gpu_artisan_executor.cc"],
    hdrs = ["llm_gpu_artisan_executor.h"],
    deps = [
        ":gpu_artisan_executor_utils",
        ":llm_gpu_artisan_processed_context",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:model_data_provider",
        "//runtime/components:model_resources",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/embedding_lookup:embedding_lookup_end_of_multi_modal",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_processed_tokens",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor/gpu_artisan/ml_drift:gpu_config_converters",
        "//runtime/executor/gpu_artisan/ml_drift:gpu_model_info_generator",
        "//runtime/executor/gpu_artisan/ml_drift:litert_thread_pool_executor",
        "//runtime/executor/gpu_artisan/ml_drift:llm",
        "//runtime/executor/gpu_artisan/ml_drift:llm_gpu_runner",
        "//runtime/executor/gpu_artisan/ml_drift:llm_gpu_runner_impl",  # buildcleaner: keep
        "//runtime/executor/gpu_artisan/ml_drift:speculative_utils",
        "//runtime/executor/gpu_artisan/ml_drift:transformer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:gpu_optimized_tensor_loader",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:llm_inference_engine_cc_proto",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/executor/gpu_artisan/utils:constraint_adapter",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "//conditions:default": [],
    }),
)

cc_test(
    name = "llm_gpu_artisan_executor_test",
    srcs = ["llm_gpu_artisan_executor_test.cc"],
    deps = [
        ":llm_gpu_artisan_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:tensor",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components/constrained_decoding:bitmap",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor/gpu_artisan/ml_drift:llm_gpu_runner",
        "//runtime/executor/gpu_artisan/ml_drift:speculative_utils",
        "//runtime/executor/gpu_artisan/ml_drift:transformer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:sampler_params_cc_proto",
        "//runtime/executor/gpu_artisan/utils:bitmap",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
        "//support/util:convert_tensor_buffer",
    ],
)
