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
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "engine",
    srcs = [
        "engine.cc",
        "litert_lm_logging.cc",
    ],
    hdrs = [
        "engine.h",
        "litert_lm_logging.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "@litert//litert/c/internal:litert_logging",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/core:engine_impl",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/engine:mezo",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
    ],
)

cc_test(
    name = "engine_test",
    srcs = ["engine_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],
    target_compatible_with = select({
        "@platforms//os:ios": [],
        "@platforms//os:linux": [],
        "@platforms//os:osx": [],
        "@platforms//os:windows": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":engine",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/synchronization",
        "@nlohmann_json//:json",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/engine:engine_settings",
        "//runtime/executor:executor_settings_base",
    ],
)
