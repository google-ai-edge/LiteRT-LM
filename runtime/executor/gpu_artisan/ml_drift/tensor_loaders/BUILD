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
    name = "basic_gpu_tensor",
    hdrs = ["basic_gpu_tensor.h"],
    deps = [
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "llm_tensor_loader",
    srcs = ["llm_tensor_loader.cc"],
    hdrs = ["llm_tensor_loader.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/ml_drift:name_check",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//support/util:memory_mapped_file",
    ],
)

cc_library(
    name = "llm_dummy_tensor_loader",
    srcs = ["llm_dummy_tensor_loader.cc"],
    hdrs = ["llm_dummy_tensor_loader.h"],
    deps = [
        ":llm_tensor_loader",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/ml_drift:name_check",
    ],
)

cc_library(
    name = "llm_file_tensor_loader",
    srcs = ["llm_file_tensor_loader.cc"],
    hdrs = ["llm_file_tensor_loader.h"],
    deps = [
        ":llm_tensor_loader",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/ml_drift:name_check",
        "//runtime/executor/gpu_artisan/utils:model_data",
    ],
)

cc_library(
    name = "gpu_optimized_tensor_loader",
    srcs = ["gpu_optimized_tensor_loader.cc"],
    hdrs = ["gpu_optimized_tensor_loader.h"],
    deps = [
        ":basic_gpu_tensor",
        ":llm_tensor_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/ml_drift:name_check",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "//support/util:memory_mapped_file",
    ],
)

cc_library(
    name = "placeholder_tensor_loader",
    srcs = ["placeholder_tensor_loader.cc"],
    hdrs = ["placeholder_tensor_loader.h"],
    deps = [
        ":llm_tensor_loader",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:model",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_conversion",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "embedding_updater",
    srcs = ["embedding_updater.cc"],
    hdrs = ["embedding_updater.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "//runtime/executor/gpu_artisan/utils:model_data",
    ],
)

cc_test(
    name = "gpu_optimized_tensor_loader_test",
    srcs = ["gpu_optimized_tensor_loader_test.cc"],
    deps = [
        ":basic_gpu_tensor",
        ":gpu_optimized_tensor_loader",
        ":llm_tensor_loader",
        ":placeholder_tensor_loader",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:weights_layout",
        "//runtime/executor/gpu_artisan/proto:llm_file_metadata_cc_proto",
        "//runtime/executor/gpu_artisan/utils:model_data",
        "//runtime/util:test_utils",
    ],
)
