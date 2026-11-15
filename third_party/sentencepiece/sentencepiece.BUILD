package(default_visibility = ["//visibility:public"])

proto_library(
    name = "sentencepiece_proto",
    srcs = ["sentencepiece.proto"],
)

cc_proto_library(
    name = "sentencepiece_cc_proto",
    deps = [":sentencepiece_proto"],
)

proto_library(
    name = "sentencepiece_model_proto",
    srcs = ["sentencepiece_model.proto"],
)

cc_proto_library(
    name = "sentencepiece_model_cc_proto",
    deps = [":sentencepiece_model_proto"],
)

cc_library(
    name = "sentencepiece_processor",
    srcs = [
        "bpe_model.cc",
        "char_model.cc",
        "filesystem.cc",
        "model_factory.cc",
        "model_interface.cc",
        "normalizer.cc",
        "sentencepiece_processor.cc",
        "unigram_model.cc",
        "util.cc",
        "word_model.cc",
    ],
    hdrs = [
        "bpe_model.h",
        "char_model.h",
        "common.h",
        "config.h",
        "filesystem.h",
        "freelist.h",
        "model_factory.h",
        "model_interface.h",
        "normalizer.h",
        "sentencepiece_processor.h",
        "sentencepiece_trainer.h",
        "trainer_interface.h",
        "unigram_model.h",
        "util.h",
        "word_model.h",
        "third_party/darts_clone/darts.h",
    ],
    copts = [
        "-DENABLE_NFKC_COMPILE",
        "-DSENTENCEPIECE_PG3_BUILD",
        "-DINSTALL_DATADIR=\\\"\\\"",
    ],
    deps = [
        ":sentencepiece_cc_proto",
        ":sentencepiece_model_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)
