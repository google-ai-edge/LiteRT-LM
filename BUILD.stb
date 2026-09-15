package(default_visibility = ["//visibility:public"])

# stb is a header-only library: a header only emits its implementation in a
# translation unit that defines the matching STB_*_IMPLEMENTATION macro before
# including it. Without a dedicated translation unit every consumer has to
# define the macro itself, and the moment two of them do the binary fails to
# link with duplicate symbols.
#
# Generate that single implementation translation unit here instead, so that
# consumers can simply include the headers.

genrule(
    name = "stb_image_impl_src",
    outs = ["stb_image_impl.cc"],
    cmd = "{ echo '#define STB_IMAGE_IMPLEMENTATION'; " +
          "echo '#include \"stb_image.h\"'; } > $@",
)

cc_library(
    name = "stb_image",
    srcs = ["stb_image_impl.cc"],
    hdrs = ["stb_image.h"],
)

genrule(
    name = "stb_image_resize2_impl_src",
    outs = ["stb_image_resize2_impl.cc"],
    cmd = "{ echo '#define STB_IMAGE_RESIZE_IMPLEMENTATION'; " +
          "echo '#include \"stb_image_resize2.h\"'; } > $@",
)

cc_library(
    name = "stblib",
    srcs = ["stb_image_resize2_impl.cc"],
    hdrs = [
        "stb_dxt.h",
        "stb_image.h",
        "stb_image_resize2.h",
        "stb_image_write.h",
    ],
    deps = [":stb_image"],
)
