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
    name = "add_with_overflow",
    srcs = ["add_with_overflow.cc"],
    hdrs = ["add_with_overflow.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "apply_attention_mask",
    srcs = ["apply_attention_mask.cc"],
    hdrs = ["apply_attention_mask.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//runtime/executor/gpu_artisan/ml_drift:runtime_params",
    ],
)

cc_library(
    name = "flash_decoding_attention",
    srcs = ["flash_decoding_attention.cc"],
    hdrs = ["flash_decoding_attention.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:kernel_info",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common:util",
        "@ml_drift//ml_drift/common/task:buffer_desc",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:tuning_type",
        "//runtime/executor/gpu_artisan/ml_drift:runtime_params",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "apply_fire_bias",
    srcs = ["apply_fire_bias.cc"],
    hdrs = ["apply_fire_bias.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "fused_rope",
    srcs = ["fused_rope.cc"],
    hdrs = ["fused_rope.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:kernel_info",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:tuning_type",
    ],
)

cc_library(
    name = "gelu7",
    srcs = ["gelu7.cc"],
    hdrs = ["gelu7.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "mul_with_overflow",
    srcs = ["mul_with_overflow.cc"],
    hdrs = ["mul_with_overflow.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_library(
    name = "rotqat_rotate",
    srcs = ["rotqat_rotate.cc"],
    hdrs = ["rotqat_rotate.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:gpu_info",
        "@ml_drift//ml_drift/common:kernel_info",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:types",
        "@ml_drift//ml_drift/common/task:gpu_operation",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "@ml_drift//ml_drift/common/task:tuning_type",
        "//runtime/util:litert_status_util",
    ],
)
