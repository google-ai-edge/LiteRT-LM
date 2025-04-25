load("//third_party/protobuf/bazel:cc_proto_library.bzl", "cc_proto_library")
load("//third_party/protobuf/bazel:proto_library.bzl", "proto_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//third_party/odml/litert_lm:__subpackages__",
    ],
)

licenses(["notice"])

proto_library(
    name = "sampler_params_proto",
    srcs = ["sampler_params.proto"],
)

cc_proto_library(
    name = "sampler_params_cc_proto",
    deps = [":sampler_params_proto"],
)
