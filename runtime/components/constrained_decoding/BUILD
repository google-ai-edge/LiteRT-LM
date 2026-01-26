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

cc_library(
    name = "bitmap",
    hdrs = ["bitmap.h"],
)

cc_library(
    name = "constraint",
    hdrs = ["constraint.h"],
    deps = [
        ":bitmap",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "external_constraint_config",
    hdrs = ["external_constraint_config.h"],
    deps = [
        ":constraint",
    ],
)

cc_library(
    name = "llg_constraint_config",
    hdrs = ["llg_constraint_config.h"],
)

cc_library(
    name = "fst_constraint_config",
    hdrs = ["fst_constraint_config.h"],
)

cc_library(
    name = "constraint_provider_config",
    hdrs = ["constraint_provider_config.h"],
    deps = [
        ":external_constraint_config",
        ":fst_constraint_config",
        ":llg_constraint_config",
    ],
)

cc_library(
    name = "constraint_provider",
    hdrs = ["constraint_provider.h"],
    deps = [
        ":constraint",
        ":constraint_provider_config",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "constrained_decoder",
    srcs = ["constrained_decoder.cc"],
    hdrs = ["constrained_decoder.h"],
    deps = [
        ":constraint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "fake_constraint",
    srcs = ["fake_constraint.cc"],
    hdrs = ["fake_constraint.h"],
    deps = [
        ":bitmap",
        ":constraint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

alias(
    name = "gemma_model_constraint_provider_lib",
    actual = ":gemma_model_constraint_provider_shared_lib",
)

cc_import(
    name = "gemma_model_constraint_provider_dll_import",
    interface_library = "//prebuilt/windows_x86_64:libGemmaModelConstraintProvider.lib",
    shared_library = "//prebuilt/windows_x86_64:libGemmaModelConstraintProvider.dll",
    target_compatible_with = ["@platforms//os:windows"],
)

cc_library(
    name = "gemma_model_constraint_provider_shared_lib",
    srcs = select({
        "@platforms//os:android": ["//prebuilt/android_arm64:libGemmaModelConstraintProvider.so"],
        "//build_config:linux_x86_64": ["//prebuilt/linux_x86_64:libGemmaModelConstraintProvider.so"],
        "//build_config:linux_arm64": ["//prebuilt/linux_arm64:libGemmaModelConstraintProvider.so"],
        "//build_config:ios_sim_arm64": ["//prebuilt/ios_sim_arm64:libGemmaModelConstraintProvider.dylib"],
        "//build_config:ios_arm64": ["//prebuilt/ios_arm64:libGemmaModelConstraintProvider.dylib"],
        "@platforms//os:macos": ["//prebuilt/macos_arm64:libGemmaModelConstraintProvider.dylib"],
        "@platforms//os:windows": [],
        "//conditions:default": [],
    }),
    hdrs = [
        "gemma_model_constraint_provider.h",
    ],
    linkopts = select({
        "@platforms//os:ios": ["-Wl,-rpath,@loader_path"],
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        ":constraint",
        ":constraint_provider",
        ":constraint_provider_config",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "@sentencepiece//:sentencepiece_processor",
    ] + select({
        "@platforms//os:windows": [":gemma_model_constraint_provider_dll_import"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "external_constraint_provider",
    srcs = ["external_constraint_provider.cc"],
    hdrs = ["external_constraint_provider.h"],
    deps = [
        ":constraint",
        ":constraint_provider",
        ":constraint_provider_config",
        ":external_constraint_config",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "llg_constraint",
    srcs = ["llg_constraint.cc"],
    hdrs = ["llg_constraint.h"],
    linkopts = select({
        "@platforms//os:windows": ["ntdll.lib"],
        "//conditions:default": [],
    }),
    deps = [
        ":bitmap",
        ":constraint",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@crate_index__llguidance-1.3.0//:llguidance_cc"
    ] + select({
        "@platforms//os:windows": [
            "//rust:alloc_defs",
            "//rust:global_allocator",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "llg_constraint_provider",
    srcs = ["llg_constraint_provider.cc"],
    hdrs = ["llg_constraint_provider.h"],
    linkopts = select({
        "@platforms//os:windows": ["ntdll.lib"],
        "//conditions:default": [],
    }),
    deps = [
        ":constraint",
        ":constraint_provider",
        ":constraint_provider_config",
        ":llg_constraint",
        ":llg_constraint_config",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/components:tokenizer",
        "@crate_index__llguidance-1.3.0//:llguidance_cc"
    ] + select({
        "@platforms//os:windows": [
            "//rust:alloc_defs",
            "//rust:global_allocator",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "llguidance_schema_utils",
    srcs = ["llguidance_schema_utils.cc"],
    hdrs = ["llguidance_schema_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "llguidance_schema_utils_test",
    srcs = ["llguidance_schema_utils_test.cc"],
    deps = [
        ":bitmap",
        ":constraint",
        ":llg_constraint_config",
        ":llg_constraint_provider",
        ":llguidance_schema_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components:tokenizer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "llg_constraint_provider_test",
    srcs = ["llg_constraint_provider_test.cc"],
    deps = [
        ":llg_constraint_config",
        ":llg_constraint_provider",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/components:tokenizer",
        "//runtime/util:test_utils",
        "@crate_index__llguidance-1.3.0//:llguidance_cc"
    ],
)

cc_test(
    name = "llg_constraint_test",
    srcs = ["llg_constraint_test.cc"],
    deps = [
        ":bitmap",
        ":constraint",
        ":constraint_provider",
        ":llg_constraint_config",
        ":llg_constraint_provider",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/components:tokenizer",
        "//runtime/util:test_utils",
    ],
)

cc_test(
    name = "external_constraint_provider_test",
    srcs = ["external_constraint_provider_test.cc"],
    deps = [
        ":constraint",
        ":constraint_provider_config",
        ":external_constraint_config",
        ":external_constraint_provider",
        ":fake_constraint",
        ":fst_constraint_config",
        ":llg_constraint_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "constraint_provider_factory",
    srcs = ["constraint_provider_factory.cc"],
    hdrs = ["constraint_provider_factory.h"],
    deps = [
        ":constraint_provider",
        ":constraint_provider_config",
        ":external_constraint_config",
        ":external_constraint_provider",
        ":llg_constraint_config",
        ":llg_constraint_provider",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/components:tokenizer",
    ],
)

cc_test(
    name = "constraint_provider_factory_test",
    srcs = ["constraint_provider_factory_test.cc"],
    deps = [
        ":constraint_provider_config",
        ":constraint_provider_factory",
        ":external_constraint_config",
        ":llg_constraint_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/components:tokenizer",
        "//runtime/util:test_utils",
    ],
)
