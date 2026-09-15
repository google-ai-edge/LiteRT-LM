# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_preprocessor",
    hdrs = ["audio_preprocessor.h"],
    deps = ["//support/preprocessor:audio_preprocessor"],
)

cc_library(
    name = "audio_preprocessor_miniaudio",
    hdrs = ["audio_preprocessor_miniaudio.h"],
    deps = ["//support/preprocessor:audio_preprocessor_miniaudio"],
)

alias(
    name = "mel_filterbank",
    actual = "//support/preprocessor:mel_filterbank",
)

alias(
    name = "signal_vector_util",
    actual = "//support/preprocessor:signal_vector_util",
)

cc_library(
    name = "image_preprocessor",
    hdrs = ["image_preprocessor.h"],
    deps = ["//support/preprocessor:image_preprocessor"],
)

alias(
    name = "image_preprocessor_impl",
    actual = "//support/preprocessor:image_preprocessor_impl",
)

alias(
    name = "by_pass_audio_preprocessor",
    actual = "//support/preprocessor:by_pass_audio_preprocessor",
)

alias(
    name = "by_pass_image_preprocessor",
    actual = "//support/preprocessor:by_pass_image_preprocessor",
)

cc_library(
    name = "stb_image_preprocessor",
    hdrs = ["stb_image_preprocessor.h"],
    deps = ["//support/preprocessor:stb_image_preprocessor"],
)

alias(
    name = "skia_image_preprocessor",
    actual = "//support/preprocessor:skia_image_preprocessor",
)

alias(
    name = "image_preprocessor_utils",
    actual = "//support/preprocessor:image_preprocessor_utils",
)

# MiniCPM-V multi-slice vision preprocessing (PIL-faithful bicubic resize +
# normalize + reshape_by_patch). The resampler pos_embed is baked into the
# fused vision tflite, so these targets only produce image strips and ViT
# position ids.

cc_library(
    name = "pil_resize",
    srcs = ["pil_resize.cc"],
    hdrs = ["pil_resize.h"],
)

cc_library(
    name = "minicpmv_image_preprocess",
    srcs = ["minicpmv_image_preprocess.cc"],
    hdrs = ["minicpmv_image_preprocess.h"],
    deps = [
        ":pil_resize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@stb//:stblib",
    ],
)

cc_test(
    name = "pil_resize_test",
    srcs = ["pil_resize_test.cc"],
    deps = [
        ":pil_resize",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "minicpmv_image_preprocess_test",
    srcs = ["minicpmv_image_preprocess_test.cc"],
    deps = [
        ":minicpmv_image_preprocess",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
        "//runtime/util:test_utils",
    ],
)
