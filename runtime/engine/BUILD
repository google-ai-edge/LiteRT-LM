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
    default_applicable_licenses = ["//third_party/odml:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//third_party/odml/litert_lm:__subpackages__",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "io_types_test",
    srcs = ["io_types_test.cc"],
    deps = [
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status",
    ],
)

cc_binary(
    name = "litert_lm_main",
    srcs = ["litert_lm_main.cc"],
    deps = [
        ":engine_interface",
        ":engine_settings",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "//runtime/executor:llm_executor_config",
    ] + select({
        "//conditions:default": ["//runtime/core:engine_impl"],
    }),
)

cc_library(
    name = "engine_settings",
    hdrs = ["engine_settings.h"],
    deps = [
        "//runtime/executor:llm_executor_config",
    ],
)

cc_test(
    name = "engine_settings_test",
    srcs = ["engine_settings_test.cc"],
    deps = [
        ":engine_settings",
        "@com_google_googletest//:gtest_main",
        "//runtime/executor:llm_executor_config",
    ],
)
