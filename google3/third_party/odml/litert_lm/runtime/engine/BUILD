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
        ":io_types",
        ":llm_model_settings",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
    ],
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
    ],
)

cc_test(
    name = "io_types_test",
    srcs = ["io_types_test.cc"],
    deps = [
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "//third_party/absl/status",
    ],
)

cc_binary(
    name = "litert_lm_main",
    srcs = ["litert_lm_main.cc"],
    deps = [
        ":engine_interface",
        ":llm_model_settings",
        "//third_party/absl/flags:flag",
        "//third_party/absl/flags:parse",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor_config",
    ] + select({
        "//conditions:default": ["//third_party/odml/litert_lm/runtime/core:engine_impl"],
    }),
)

cc_library(
    name = "llm_model_settings",
    hdrs = ["llm_model_settings.h"],
    deps = [
        "//third_party/odml/litert_lm/runtime/executor:llm_executor_config",
    ],
)

cc_test(
    name = "llm_model_settings_test",
    srcs = ["llm_model_settings_test.cc"],
    deps = [
        ":llm_model_settings",
        "@com_google_googletest//:gtest_main",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor_config",
    ],
)
