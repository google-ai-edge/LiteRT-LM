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
    name = "bitmap",
    hdrs = ["bitmap.h"],
)

cc_library(
    name = "constraint_utils",
    hdrs = ["constraint_utils.h"],
    deps = [
        ":bitmap",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "constraint_adapter",
    srcs = ["constraint_adapter.cc"],
    hdrs = ["constraint_adapter.h"],
    deps = [
        ":bitmap",
        ":constraint_utils",
        "@com_google_absl//absl/status:statusor",
        "//runtime/components/constrained_decoding:bitmap",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "constraint_adapter_test",
    srcs = ["constraint_adapter_test.cc"],
    deps = [
        ":bitmap",
        ":constraint_adapter",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/components/constrained_decoding:bitmap",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "metadata_utils",
    srcs = ["metadata_utils.cc"],
    hdrs = ["metadata_utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "@litert//tflite:framework_stable",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "metadata_utils_test",
    srcs = ["metadata_utils_test.cc"],
    deps = [
        ":metadata_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "//runtime/executor/gpu_artisan/proto:llm_params_cc_proto",
        "//runtime/util:test_utils",
        "@litert//tflite:framework_stable",
        "@litert//tflite/schema:schema_fbs",
    ],
)
