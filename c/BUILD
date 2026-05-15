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
# [Google-internal load of `objc_library`]
load("@build_bazel_rules_apple//apple:apple_xcframework.bzl", "apple_xcframework")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files([
    "engine.cc",
    "engine.h",
])

ENGINE_COMMON_DEPS = [
    "@com_google_absl//absl/base:core_headers",
    "@com_google_absl//absl/base:log_severity",
    "@com_google_absl//absl/functional:any_invocable",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/log:initialize",
    "@com_google_absl//absl/log:log_sink_registry",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings:str_format",
    "@com_google_absl//absl/strings:string_view",
    "@com_google_absl//absl/time",
    "@nlohmann_json//:json",
    "//runtime/conversation",
    "//runtime/conversation:io_types",
    "//runtime/conversation/model_data_processor:config_registry",
    "//runtime/conversation/model_data_processor:gemma4_data_processor_config",
    "//runtime/engine:engine_factory",
    "//runtime/engine:engine_interface",
    "//runtime/engine:engine_settings",
    "//runtime/engine:io_types",
    "//runtime/executor:executor_settings_base",
    "//runtime/executor:llm_executor_settings",
    "//runtime/proto:engine_cc_proto",
    "//runtime/proto:llm_metadata_cc_proto",
    "//runtime/proto:sampler_params_cc_proto",
    "//runtime/proto:token_cc_proto",
    "//runtime/util:logging",
]

cc_library(
    name = "engine",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
    alwayslink = True,
)

cc_library(
    name = "engine_cpu",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    visibility = ["//visibility:public"],
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl_cpu_only",
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
        "//runtime/executor:llm_executor_settings",
    ],
)

objc_library(
    name = "CLiteRTLM_objc",
    deps = [
        ":engine",
        "//schema/capabilities:capabilities_c",
    ],
)

apple_xcframework(
    name = "CLiteRTLM",
    bundle_id = "com.google.odml.litertlm.CLiteRTLM",
    features = ["exported_symbols"],
    infoplists = ["Info.plist"],
    ios = {
        "simulator": [
            "arm64",
            "x86_64",
        ],
        "device": [
            "arm64",
        ],
    },
    minimum_os_versions = {
        "ios": "15.0",
    },
    public_hdrs = [
        "engine.h",
        "//schema/capabilities:capabilities_c.h",
    ],
    deps = [
        ":CLiteRTLM_objc",
    ],
)
