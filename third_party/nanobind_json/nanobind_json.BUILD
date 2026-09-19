load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "nanobind_json",
    hdrs = ["include/nanobind_json/nanobind_json.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@nlohmann_json//:json",
        "@nanobind",
    ],
)
