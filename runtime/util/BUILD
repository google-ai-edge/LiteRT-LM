load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    default_applicable_licenses = ["//:license"],
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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/c:litert_common",
        "@litert//litert/litert/cc:litert_element_type",
        "@litert//litert/litert/cc:litert_expected",
        "@litert//litert/litert/cc:litert_layout",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "convert_tensor_buffer_test",
    srcs = ["convert_tensor_buffer_test.cc"],
    deps = [
        ":convert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/c:litert_common",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_tensor_buffer",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@litert//litert/litert/c:litert_common",
        "@litert//litert/litert/cc:litert_expected",
        "@litert//litert/litert/cc:litert_macros",
    ],
)

cc_test(
    name = "litert_status_util_test",
    srcs = ["litert_status_util_test.cc"],
    deps = [
        ":litert_status_util",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@litert//litert/litert/c:litert_common",
        "@litert//litert/litert/cc:litert_expected",
    ],
)

cc_test(
    name = "status_macros_test",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":litert_status_util",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    deps = [
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "logging_tensor_buffer",
    srcs = ["logging_tensor_buffer.cc"],
    hdrs = ["logging_tensor_buffer.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/c:litert_tensor_buffer_types",
        "@litert//litert/litert/cc:litert_element_type",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        ":logging",
        ":logging_tensor_buffer",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/litert/c:litert_tensor_buffer",
        "@litert//litert/litert/cc:litert_element_type",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "tensor_buffer_util",
    srcs = ["tensor_buffer_util.cc"],
    hdrs = ["tensor_buffer_util.h"],
    deps = [
        "@litert//litert/litert/cc:litert_macros",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "tensor_buffer_util_test",
    srcs = ["tensor_buffer_util_test.cc"],
    deps = [
        ":convert_tensor_buffer",
        ":tensor_buffer_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_mapped_file",
    srcs = select({
        "//tools/cc_target_os:windows": ["memory_mapped_file_win.cc"],
        "//conditions:default": ["memory_mapped_file_posix.cc"],
    }),
    hdrs = ["memory_mapped_file.h"],
    copts = [
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10",
    ],
    deps = [
        ":scoped_file",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "memory_mapped_file_test",
    srcs = ["memory_mapped_file_test.cc"],
    deps = [
        ":memory_mapped_file",
        ":scoped_file",
        "//testing/base/public:gunit",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_library(
    name = "scoped_file",
    srcs = select({
        "//tools/cc_target_os:windows": ["scoped_file_win.cc"],
        "//conditions:default": ["scoped_file_posix.cc"],
    }),
    hdrs = ["scoped_file.h"],
    deps = [
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "model_asset_bundle_resources",
    srcs = ["model_asset_bundle_resources.cc"],
    hdrs = [
        "model_asset_bundle_resources.h",
        "status_macros.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":external_file_cc_proto",
        ":external_file_handler",
        ":zip_utils",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@litert//litert/litert/cc:litert_macros",
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
    name = "external_file_handler",
    srcs = ["external_file_handler.cc"],
    hdrs = [
        "external_file_handler.h",
        "status_macros.h",
    ],
    deps = [
        ":external_file_cc_proto",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@litert//litert/litert/cc:litert_macros",
    ],
)

cc_library(
    name = "zip_readonly_mem_file",
    srcs = ["zip_readonly_mem_file.cc"],
    hdrs = ["zip_readonly_mem_file.h"],
    deps = [
        "@abseil-cpp//absl/strings",
        "@minizip-ng//minizip:zlib_minizip",
        "@zlib//zlib:zlibheaders",
    ],
)

cc_library(
    name = "zip_utils",
    srcs = ["zip_utils.cc"],
    hdrs = [
        "status_macros.h",
        "zip_utils.h",
    ],
    deps = [
        ":external_file_cc_proto",
        ":zip_readonly_mem_file",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@minizip-ng//minizip:zlib_minizip",
        "@litert//litert/litert/cc:litert_macros",
        "@zlib//zlib:zlibheaders",
    ],
)
