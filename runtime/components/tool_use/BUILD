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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

cc_library(
    name = "python_tool_format_utils",
    srcs = ["python_tool_format_utils.cc"],
    hdrs = ["python_tool_format_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "python_tool_format_utils_test",
    srcs = ["python_tool_format_utils_test.cc"],
    deps = [
        ":python_tool_format_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "fc_tool_format_utils",
    srcs = ["fc_tool_format_utils.cc"],
    hdrs = ["fc_tool_format_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "fc_tool_format_utils_test",
    srcs = ["fc_tool_format_utils_test.cc"],
    deps = [
        ":fc_tool_format_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "parser_common",
    srcs = ["parser_common.cc"],
    hdrs = ["parser_common.h"],
    linkopts = select({
        "@platforms//os:windows": ["ntdll.lib"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use/rust:parsers_cpp",
    ],
)

cc_test(
    name = "parser_common_test",
    srcs = ["parser_common_test.cc"],
    deps = [
        ":parser_common",
        "@com_google_googletest//:gtest_main",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use/rust:parsers_cpp",
    ],
)

cc_library(
    name = "python_parser_utils",
    srcs = ["python_parser_utils.cc"],
    hdrs = ["python_parser_utils.h"],
    deps = [
        ":parser_common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use/rust:parsers_cpp",
    ],
)

cc_test(
    name = "python_parser_utils_test",
    srcs = ["python_parser_utils_test.cc"],
    deps = [
        ":python_parser_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "json_parser_utils",
    srcs = ["json_parser_utils.cc"],
    hdrs = ["json_parser_utils.h"],
    deps = [
        ":parser_common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use/rust:parsers_cpp",
    ],
)

cc_test(
    name = "json_parser_utils_test",
    srcs = ["json_parser_utils_test.cc"],
    deps = [
        ":json_parser_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "fc_parser_utils",
    srcs = ["fc_parser_utils.cc"],
    hdrs = ["fc_parser_utils.h"],
    deps = [
        ":parser_common",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use/rust:parsers_cpp",
    ],
)

# Disable heap checking for this test. The Rust antlr4rust library allocates
# global, static DFA/ATN caches to optimize parsing. Rust intentionally does
# not free these static variables at program exit, which the C++ HeapLeakChecker
# catches as a false positive leak.
cc_test(
    name = "fc_parser_utils_test",
    srcs = ["fc_parser_utils_test.cc"],
    args = ["--heap_check="],
    deps = [
        ":fc_parser_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "parser_utils",
    srcs = ["parser_utils.cc"],
    hdrs = ["parser_utils.h"],
    deps = [
        ":fc_parser_utils",
        ":json_parser_utils",
        ":python_parser_utils",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "parser_utils_test",
    srcs = ["parser_utils_test.cc"],
    deps = [
        ":parser_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@nlohmann_json//:json",
        "//runtime/util:test_utils",
    ],
)
