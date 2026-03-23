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
    name = "convert_tensor_buffer",
    hdrs = ["convert_tensor_buffer.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@litert//tflite/types:half",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc:litert_tensor_buffer_types",
        ],
    }),
)

cc_test(
    name = "convert_tensor_buffer_test",
    srcs = ["convert_tensor_buffer_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":convert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "executor_data_util",
    srcs = ["executor_data_util.cc"],
    hdrs = ["executor_data_util.h"],
    deps = [
        ":litert_status_util",
        ":tensor_buffer_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@litert//litert/cc:litert_macros",
        "//runtime/executor:llm_executor_io_types",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc:litert_tensor_buffer_types",
            "@litert//litert/cc/internal:litert_handle",
        ],
    }),
)

cc_test(
    name = "executor_data_util_test",
    srcs = ["executor_data_util_test.cc"],
    deps = [
        ":convert_tensor_buffer",
        ":executor_data_util",
        ":tensor_buffer_util",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
        "//runtime/executor:llm_executor_io_types",
    ],
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
        "//runtime/executor:executor_settings_base",
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
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
    ],
)

cc_library(
    name = "file_util",
    srcs = ["file_util.cc"],
    hdrs = ["file_util.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "file_util_test",
    srcs = ["file_util_test.cc"],
    deps = [
        ":file_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_status_util",
    srcs = [],
    hdrs = [
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
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_tensor_buffer_types",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
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
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "tensor_buffer_util",
    srcs = ["tensor_buffer_util.cc"],
    hdrs = ["tensor_buffer_util.h"],
    deps = ["@litert//litert/cc:litert_macros"] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "tensor_buffer_util_test",
    srcs = ["tensor_buffer_util_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
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
        "@com_google_absl//absl/memory",
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
    hdrs = ["scoped_file.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@litert//litert/cc/internal:scoped_file",
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
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/types:span",
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
        "@zlib",
    ],
)

cc_test(
    name = "zip_utils_test",
    srcs = ["zip_utils_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":memory_mapped_file",
        ":test_utils",
        ":zip_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
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
    name = "litert_lm_loader",
    srcs = ["litert_lm_loader.cc"],
    hdrs = ["litert_lm_loader.h"],
    deps = [
        ":litert_status_util",
        ":memory_mapped_file",
        ":scoped_file",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_buffer_ref",
        "//runtime/components:model_resources",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_read",
    ],
)

cc_library(
    name = "metadata_util",
    srcs = ["metadata_util.cc"],
    hdrs = ["metadata_util.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/proto:llm_metadata_cc_proto",
    ],
)

cc_test(
    name = "metadata_util_test",
    srcs = ["metadata_util_test.cc"],
    deps = [
        ":metadata_util",
        "@com_google_googletest//:gtest_main",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:token_cc_proto",
    ],
)

cc_library(
    name = "lora_data",
    srcs = ["lora_data.cc"],
    hdrs = ["lora_data.h"],
    deps = [
        ":litert_status_util",
        ":lora_util",
        ":scoped_file",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
        "@litert//litert/cc:litert_buffer_ref",
        "@litert//tflite:framework",
        "@litert//tflite/schema:schema_fbs",
    ],
)

cc_test(
    name = "lora_data_test",
    srcs = ["lora_data_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":litert_status_util",
        ":lora_data",
        ":memory_mapped_file",
        ":scoped_file",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_buffer_ref",
        "//runtime/executor:executor_settings_base",
    ],
)

cc_library(
    name = "lora_util",
    srcs = ["lora_util.cc"],
    hdrs = ["lora_util.h"],
    deps = [
        ":litert_status_util",
        ":memory_mapped_file",
        ":scoped_file",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_buffer_ref",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "lora_util_test",
    srcs = ["lora_util_test.cc"],
    deps = [
        ":lora_util",
        ":memory_mapped_file",
        ":scoped_file",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "model_type_utils",
    srcs = ["model_type_utils.cc"],
    hdrs = ["model_type_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/components:tokenizer",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "model_type_utils_test",
    srcs = ["model_type_utils_test.cc"],
    deps = [
        ":model_type_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/components:prompt_template",
        "//runtime/components:tokenizer",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "metrics_util",
    srcs = ["metrics_util.cc"],
    hdrs = ["metrics_util.h"],
    deps = [
        ":litert_status_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "//runtime/engine:io_types",
        "//runtime/engine:litert_lm_lib",
        "//runtime/proto:litert_lm_metrics_cc_proto",
    ],
)

cc_test(
    name = "metrics_util_test",
    srcs = ["metrics_util_test.cc"],
    deps = [
        ":metrics_util",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/time",
        "//runtime/engine:io_types",
        "//runtime/engine:litert_lm_lib",
        "//runtime/proto:engine_cc_proto",
        "//runtime/proto:litert_lm_metrics_cc_proto",
    ],
)
