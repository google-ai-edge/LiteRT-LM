
package(
    default_applicable_licenses = ["//:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
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
