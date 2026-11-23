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

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

exports_files([
    "chat_template_input_schema.json",
])

cc_library(
    name = "json_schema_validator",
    testonly = True,
    srcs = ["json_schema_validator.cc"],
    hdrs = ["json_schema_validator.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "json_schema_validator_test",
    srcs = ["json_schema_validator_test.cc"],
    data = ["chat_template_input_schema.json"],
    deps = [
        ":json_schema_validator",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "chat_template_test_runner_lib",
    testonly = True,
    srcs = ["chat_template_test_runner.cc"],
    deps = [
        ":json_schema_validator",
        "@com_google_googletest//:gtest_for_library",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/proto:llm_metadata_cc_proto",
        "//support/util:test_utils",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1,
)
