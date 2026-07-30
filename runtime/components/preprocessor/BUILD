# [Google-internal load of `cc_library`]

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_preprocessor",
    hdrs = ["audio_preprocessor.h"],
    deps = ["@litert//support/preprocessor:audio_preprocessor"],
)

cc_library(
    name = "audio_preprocessor_miniaudio",
    hdrs = ["audio_preprocessor_miniaudio.h"],
    deps = ["@litert//support/preprocessor:audio_preprocessor_miniaudio"],
)

alias(
    name = "mel_filterbank",
    actual = "@litert//support/preprocessor:mel_filterbank",
)

alias(
    name = "signal_vector_util",
    actual = "@litert//support/preprocessor:signal_vector_util",
)

cc_library(
    name = "image_preprocessor",
    hdrs = ["image_preprocessor.h"],
    deps = ["@litert//support/preprocessor:image_preprocessor"],
)

alias(
    name = "image_preprocessor_impl",
    actual = "@litert//support/preprocessor:image_preprocessor_impl",
)

alias(
    name = "by_pass_audio_preprocessor",
    actual = "@litert//support/preprocessor:by_pass_audio_preprocessor",
)

alias(
    name = "by_pass_image_preprocessor",
    actual = "@litert//support/preprocessor:by_pass_image_preprocessor",
)

cc_library(
    name = "stb_image_preprocessor",
    hdrs = ["stb_image_preprocessor.h"],
    deps = ["@litert//support/preprocessor:stb_image_preprocessor"],
)

alias(
    name = "skia_image_preprocessor",
    actual = "@litert//support/preprocessor:skia_image_preprocessor",
)

alias(
    name = "image_preprocessor_utils",
    actual = "@litert//support/preprocessor:image_preprocessor_utils",
)

# MiniCPM-V multi-slice vision preprocessing (PIL-faithful bicubic resize +
# normalize + reshape_by_patch). Resampler pos_embed is baked into the fused
# vision tflite; this target only produces image strips and ViT position ids.

cc_library(
    name = "pil_resize",
    srcs = ["pil_resize.cc"],
    hdrs = ["pil_resize.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_library(
    name = "minicpmv_image_preprocess",
    srcs = ["minicpmv_image_preprocess.cc"],
    hdrs = ["minicpmv_image_preprocess.h"],
    deps = [
        ":pil_resize",
        # stb_image.h declarations; the STB_IMAGE_IMPLEMENTATION symbols come
        # from :stb_image_preprocessor (the single TU that compiles them).
        ":stb_image_preprocessor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@stb//:stb_image",
    ],
)
