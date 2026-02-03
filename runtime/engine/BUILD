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
    name = "engine_interface",
    hdrs = ["engine.h"],
    deps = [
        ":engine_settings",
        ":io_types",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//runtime/components:tokenizer",
    ],
)

cc_library(
    name = "engine_factory",
    hdrs = ["engine_factory.h"],
    deps = [
        ":engine_interface",
        ":engine_settings",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# The selected engine implementations based on the build flags.
cc_library(
    name = "engine_impl_selected",
    deps = select({
        "//conditions:default": ["//runtime/core:engine_impl"],
    }),
)

cc_library(
    name = "io_types",
    srcs = ["io_types.cc"],
    hdrs = ["io_types.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/proto:engine_cc_proto",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "io_types_test",
    srcs = ["io_types_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "litert_lm_lib",
    srcs = ["litert_lm_lib.cc"],
    hdrs = ["litert_lm_lib.h"],
    deps = [
        ":engine_factory",
        ":engine_impl_selected",
        ":engine_interface",
        ":engine_settings",
        ":io_types",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
        "@com_googlesource_code_re2//:re2",
        "@stb//:stb_image",
        "@litert//tflite/profiling:memory_info",
        "@litert//tflite/profiling:memory_usage_monitor",
    ] + select({
        "@platforms//os:linux": [
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "shared_flags",
    srcs = ["shared_flags.cc"],
    hdrs = ["shared_flags.h"],
    deps = [
        "@com_google_absl//absl/flags:flag",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "litert_lm_main",
    srcs = ["litert_lm_main.cc"],
    additional_linker_inputs = select({
        "@platforms//os:windows": [
            "@litert//litert/c:windows_exported_symbols.def",
        ],
        "@platforms//os:linux": [":litert_lm_main.exported_symbols"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@litert//litert:litert_link_capi_so": [],
        # Export LiteRt* symbols for LiteRt accelerator shlibs.
        "@platforms//os:ios": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:macos": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:windows": [
            "/DEF:$(location @litert//litert/c:windows_exported_symbols.def)",
        ],
        "@platforms//os:linux": ["-Wl,--dynamic-list=$(location :litert_lm_main.exported_symbols)"],
        "//conditions:default": ["-Wl,--export-dynamic-symbol=LiteRt*"],
    }) + select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3"],
        "//conditions:default": [],
    }),
    deps = [
        ":engine_factory",
        ":engine_impl_selected",
        ":engine_interface",
        ":engine_settings",
        ":io_types",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/conversation",
        "//runtime/conversation:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/util:litert_status_util",
    ],
)

cc_binary(
    name = "litert_lm_advanced_main",
    srcs = ["litert_lm_advanced_main.cc"],
    additional_linker_inputs = select({
        "@platforms//os:windows": [
            "@litert//litert/c:windows_exported_symbols.def",
        ],
        "@platforms//os:linux": [":litert_lm_main.exported_symbols"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@litert//litert:litert_link_capi_so": [],
        # Export LiteRt* symbols for LiteRt accelerator shlibs.
        "@platforms//os:ios": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:macos": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:windows": [
            "/DEF:$(location @litert//litert/c:windows_exported_symbols.def)",
        ],
        "@platforms//os:linux": ["-Wl,--dynamic-list=$(location :litert_lm_main.exported_symbols)"],
        "//conditions:default": ["-Wl,--export-dynamic-symbol=LiteRt*"],
    }) + select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3"],
        "//conditions:default": [],
    }),
    deps = [
        ":io_types",
        ":litert_lm_lib",
        ":shared_flags",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:litert_status_util",
    ] + select({
        "@platforms//os:windows": [
        ],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "mezo",
    srcs = ["mezo.cc"],
    hdrs = ["mezo.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "mezo_test",
    srcs = ["mezo_test.cc"],
    deps = [
        ":mezo",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "mezo_multi_session_test",
    srcs = ["mezo_multi_session_test.cc"],
    deps = [
        ":engine_settings",
        ":mezo",
        "//runtime/components:lora_manager",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "engine_settings",
    srcs = ["engine_settings.cc"],
    hdrs = ["engine_settings.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/components:tokenizer",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:vision_executor_settings",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_type_utils",
    ],
)

cc_test(
    name = "engine_settings_test",
    srcs = ["engine_settings_test.cc"],
    deps = [
        ":engine_settings",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
        "@litert//litert/cc/internal:scoped_file",
        "//runtime/components:tokenizer",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:test_utils",
    ],
)
