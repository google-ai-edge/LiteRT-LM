# Custom op kernels for LiteRT Optimized implementation.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

cc_library(
    name = "attention_utils",
    srcs = ["attention_utils.cc"],
    hdrs = ["attention_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@ruy//ruy",
    ],
)

cc_library(
    name = "scaled_dot_product_attention_transposed_kernel",
    srcs = ["scaled_dot_product_attention_transposed_kernel.cc"],
    hdrs = ["scaled_dot_product_attention_transposed_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@flatbuffers",
        "//runtime/framework:threadpool",
        "@ruy//ruy",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_custom_op_kernel",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "attention_utils_test",
    srcs = ["attention_utils_test.cc"],
    deps = [
        ":attention_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
