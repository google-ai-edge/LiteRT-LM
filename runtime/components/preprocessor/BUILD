# [Google-internal load of `cc_library`]

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
