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

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

config_setting(
    name = "use_legacy_engine",
    values = {"copt": "-DUSE_LEGACY_ENGINE"},
)

cc_library(
    name = "engine_interface",
    hdrs = ["engine.h"],
    deps = [
        ":engine_settings",
        ":io_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//runtime/proto:engine_cc_proto",
    ],
)

cc_test(
    name = "io_types_test",
    srcs = ["io_types_test.cc"],
    deps = [
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "//runtime/util:test_utils",
    ],
)

cc_binary(
    name = "litert_lm_main",
    srcs = ["litert_lm_main.cc"],
    # Export LiteRt* symbols for LiteRt accelerator shlibs.
    linkopts = ["-Wl,--export-dynamic-symbol=LiteRt*"] + select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3"],
        "//conditions:default": [],
    }),
    deps = [
        ":engine_interface",
        ":engine_settings",
        ":io_types",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//runtime/executor:llm_executor_settings",
    ] + select({
        "//conditions:default": ["//runtime/core:engine_impl"],
    }),
)

cc_library(
    name = "engine_settings",
    srcs = ["engine_settings.cc"],
    hdrs = ["engine_settings.h"],
    deps = [
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
    ],
)

cc_test(
    name = "engine_settings_test",
    srcs = ["engine_settings_test.cc"],
    deps = [
        ":engine_settings",
        "@com_google_googletest//:gtest_main",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:engine_cc_proto",
    ],
)
