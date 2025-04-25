package(
    default_applicable_licenses = ["//:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "tokenizer",
    hdrs = ["tokenizer.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/cc:litert_macros",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "sampling_cpu_util",
    srcs = ["sampling_cpu_util.cc"],
    hdrs = ["sampling_cpu_util.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "sampler",
    hdrs = ["sampler.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "sampling_cpu_util_test",
    srcs = ["sampling_cpu_util_test.cc"],
    deps = [
        ":sampling_cpu_util",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer",
    srcs = ["sentencepiece_tokenizer.cc"],
    hdrs = ["sentencepiece_tokenizer.h"],
    deps = [
        ":tokenizer",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@sentencepiece//src:sentencepiece_processor",
    ],
)

cc_test(
    name = "sentencepiece_tokenizer_test",
    srcs = ["sentencepiece_tokenizer_test.cc"],
    data = [":testdata"],
    deps = [
        ":sentencepiece_tokenizer",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@litert//litert/litert/cc:litert_layout",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "token_id_util",
    srcs = ["token_id_util.cc"],
    hdrs = ["token_id_util.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "token_id_util_test",
    srcs = ["token_id_util_test.cc"],
    deps = [
        ":token_id_util",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "top_p_cpu_sampler",
    srcs = ["top_p_cpu_sampler.cc"],
    hdrs = ["top_p_cpu_sampler.h"],
    deps = [
        ":sampler",
        ":sampling_cpu_util",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/cc:litert_macros",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:tensor_buffer_util",
    ],
)

cc_test(
    name = "top_p_cpu_sampler_test",
    srcs = ["top_p_cpu_sampler_test.cc"],
    deps = [
        ":top_p_cpu_sampler",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
    ],
)
