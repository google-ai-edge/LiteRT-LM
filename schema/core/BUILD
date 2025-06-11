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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "litertlm_utils",
    srcs = ["litertlm_utils.cc"],
    hdrs = ["litertlm_utils.h"],
    deps = [
        ":litertlm_header",
        ":litertlm_header_schema",
        "@flatbuffers",
    ],
)

cc_library(
    name = "litertlm_read",
    srcs = ["litertlm_read.cc"],
    hdrs = ["litertlm_read.h"],
    deps = [
        ":litertlm_header",
        ":litertlm_header_schema",
        ":litertlm_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:scoped_file",
        "@sentencepiece//:sentencepiece_processor",
        "@tensorflow//compiler/mlir/lite:allocation",
        "@litert//tflite:framework",
    ],
)

cc_test(
    name = "litertlm_read_test",
    srcs = ["litertlm_read_test.cc"],
    data = [
        "//schema:testdata",
    ],
    deps = [
        ":litertlm_header_schema",
        ":litertlm_read",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:memory_mapped_file",
        "@sentencepiece//:sentencepiece_processor",
        "@litert//tflite:framework",
        "@litert//tflite:framework_stable",
        "@litert//tflite/c:common",
        "@litert//tflite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "litertlm_export",
    srcs = ["litertlm_export.cc"],
    hdrs = ["litertlm_export.h"],
    deps = [
        ":litertlm_header",
        ":litertlm_header_schema",
        ":litertlm_section",
        ":litertlm_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "litertlm_header",
    hdrs = ["litertlm_header.h"],
    deps = [
        ":litertlm_header_schema",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_library(
    name = "litertlm_section",
    hdrs = ["litertlm_section.h"],
    deps = [
        ":litertlm_header_schema",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_test(
    name = "litertlm_section_test",
    srcs = ["litertlm_section_test.cc"],
    data = [
        "//schema:testdata",
    ],
    deps = [
        ":litertlm_section",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//runtime/proto:llm_metadata_cc_proto",
    ],
)

flatbuffer_cc_library(
    name = "litertlm_header_schema",
    srcs = ["litertlm_header_schema.fbs"],
    flatc_args = [
        "--gen-mutable",
        "--gen-object-api",
    ],
)

cc_test(
    name = "litertlm_header_test",
    srcs = ["litertlm_header_test.cc"],
    deps = [
        ":litertlm_header",
        ":litertlm_header_schema",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@flatbuffers",
    ],
)

cc_library(
    name = "litertlm_print",
    srcs = ["litertlm_print.cc"],
    hdrs = ["litertlm_print.h"],
    deps = [
        ":litertlm_header",
        ":litertlm_header_schema",
        ":litertlm_read",
        ":litertlm_section",
        ":litertlm_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_test(
    name = "litertlm_print_test",
    srcs = ["litertlm_print_test.cc"],
    data = [
        "//schema:testdata",
    ],
    deps = [
        ":litertlm_print",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
    ],
)
