load("@rules_cc//cc:objc_library.bzl", "objc_library")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "miniaudio",
    srcs = ["miniaudio.c"],
    hdrs = ["miniaudio.h"],
)

genrule(
    name = "generate_miniaudio_mm",
    srcs = ["miniaudio.c"],
    outs = ["miniaudio.mm"],
    cmd = "cp $(location miniaudio.c) $(location miniaudio.mm)",
)

objc_library(
    name = "miniaudio_objc",
    srcs = [":miniaudio.mm"],
    hdrs = ["miniaudio.h"],
    sdk_frameworks = [
        "AudioToolbox",
        "AVFoundation",
    ],
)
