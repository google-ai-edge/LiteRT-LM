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
    name = "byom_factory",
    srcs = ["byom_factory.cc"],
    hdrs = ["byom_factory.h"],
    deps = [
        ":falcon",
        ":phi",
        ":stablelm",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "//runtime/executor/gpu_artisan/ml_drift:transformer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
    ],
)

cc_library(
    name = "llm_base",
    srcs = ["llm_base.cc"],
    hdrs = ["llm_base.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:gpu_model",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/kernels:split",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:gpu_tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift:transformer",
        "//runtime/executor/gpu_artisan/ml_drift/tensor_loaders:llm_tensor_loader",
    ],
)

cc_library(
    name = "stablelm",
    srcs = ["stablelm.cc"],
    hdrs = ["stablelm.h"],
    deps = [
        ":llm_base",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift:llm",
    ],
)

cc_library(
    name = "falcon",
    srcs = ["falcon.cc"],
    hdrs = ["falcon.h"],
    deps = [
        ":llm_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:operations",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift:transformer",
        "//runtime/executor/gpu_artisan/ml_drift/tasks:apply_attention_mask",
    ],
)

cc_library(
    name = "phi",
    srcs = ["phi.cc"],
    hdrs = ["phi.h"],
    deps = [
        ":llm_base",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/common:gpu_model_builder",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift:llm",
    ],
)
