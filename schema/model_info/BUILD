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

# [Google-internal load of `cc_binary`]
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
    name = "model_info",
    srcs = ["model_info.cc"],
    hdrs = ["model_info.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_status_util",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_read",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "model_info_test",
    srcs = ["model_info_test.cc"],
    deps = [
        ":model_info",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//runtime/proto:llm_metadata_cc_proto",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//support/util:test_utils",
        "@litert//tflite:version",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "model_info_cli",
    srcs = ["model_info_cli.cc"],
    deps = [
        ":model_info",
        "@com_google_absl//absl/status:statusor",
    ],
)
