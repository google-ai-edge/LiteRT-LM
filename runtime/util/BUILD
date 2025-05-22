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
    name = "convert_tensor_buffer",
    hdrs = ["convert_tensor_buffer.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "convert_tensor_buffer_test",
    srcs = ["convert_tensor_buffer_test.cc"],
    deps = [
        ":convert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/c:litert_tensor_buffer_types",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
        "@litert//litert/test:matchers",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "file_format_util",
    srcs = ["file_format_util.cc"],
    hdrs = ["file_format_util.h"],
    deps = [
        ":litert_status_util",
        ":memory_mapped_file",
        ":scoped_file",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor:llm_executor_settings",
    ],
)

cc_test(
    name = "file_format_util_test",
    srcs = ["file_format_util_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":file_format_util",
        ":memory_mapped_file",
        ":scoped_file",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/executor:llm_executor_settings",
    ],
)

cc_library(
    name = "litert_status_util",
    srcs = ["litert_status_util.cc"],
    hdrs = [
        "litert_status_util.h",
        "status_macros.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_macros",
    ],
)

cc_test(
    name = "litert_status_util_test",
    srcs = ["litert_status_util_test.cc"],
    deps = [
        ":litert_status_util",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/c:litert_common",
        "@litert//litert/cc:litert_expected",
    ],
)

cc_test(
    name = "status_macros_test",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":litert_status_util",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "logging_tensor_buffer",
    srcs = ["logging_tensor_buffer.cc"],
    hdrs = ["logging_tensor_buffer.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_tensor_buffer_types",
        "@litert//litert/cc:litert_element_type",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        ":logging",
        ":logging_tensor_buffer",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/c:litert_tensor_buffer",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_model",
    ] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "tensor_buffer_util",
    srcs = ["tensor_buffer_util.cc"],
    hdrs = ["tensor_buffer_util.h"],
    deps = ["@litert//litert/cc:litert_macros"] + select({
        "//:litert_lm_link_capi_so": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
        "//conditions:default": [
            "@litert//litert/cc/internal:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "tensor_buffer_util_test",
    srcs = ["tensor_buffer_util_test.cc"],
    deps = [
        ":convert_tensor_buffer",
        ":tensor_buffer_util",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "memory_mapped_file",
    srcs = select({
        "@platforms//os:windows": ["memory_mapped_file_win.cc"],
        "//conditions:default": ["memory_mapped_file_posix.cc"],
    }),
    hdrs = ["memory_mapped_file.h"],
    copts = [
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10",
    ],
    deps = [
        ":litert_status_util",
        ":scoped_file",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "memory_mapped_file_test",
    srcs = ["memory_mapped_file_test.cc"],
    deps = [
        ":memory_mapped_file",
        ":scoped_file",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "scoped_file",
    srcs = ["scoped_file.cc"] + select({
        "@platforms//os:windows": ["scoped_file_win.cc"],
        "//conditions:default": ["scoped_file_posix.cc"],
    }),
    hdrs = ["scoped_file.h"],
    deps = [
        ":litert_status_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "scoped_file_test",
    srcs = ["scoped_file_test.cc"],
    deps = [
        ":scoped_file",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "model_asset_bundle_resources",
    srcs = ["model_asset_bundle_resources.cc"],
    hdrs = ["model_asset_bundle_resources.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":litert_status_util",
        ":memory_mapped_file",
        ":scoped_file",
        ":zip_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@litert//litert/cc:litert_macros",
    ],
)

proto_library(
    name = "external_file_proto",
    srcs = ["external_file.proto"],
)

cc_proto_library(
    name = "external_file_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":external_file_proto"],
)

cc_library(
    name = "zip_readonly_mem_file",
    srcs = ["zip_readonly_mem_file.cc"],
    hdrs = ["zip_readonly_mem_file.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@minizip//:zlib_minizip",
        "@zlib//:zlib",
    ],
)

cc_library(
    name = "zip_utils",
    srcs = ["zip_utils.cc"],
    hdrs = ["zip_utils.h"],
    deps = [
        ":external_file_cc_proto",
        ":litert_status_util",
        ":zip_readonly_mem_file",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@minizip//:zlib_minizip",
        "@litert//litert/cc:litert_macros",
        "@zlib//:zlib",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    hdrs = ["test_utils.h"],
    deps = [
        "@com_google_googletest//:gtest_for_library",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_macros",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    compatible_with = [
        "//buildenv/target:prod",
        "//buildenv/target:non_prod",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "litert_lm_loader",
    srcs = ["litert_lm_loader.cc"],
    hdrs = ["litert_lm_loader.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_buffer_ref",
        ":memory_mapped_file",
        ":scoped_file",
    ],
)
