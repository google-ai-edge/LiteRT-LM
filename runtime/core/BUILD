package(
    default_applicable_licenses = ["//:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "engine_impl",
    srcs = ["engine_impl.cc"],
    deps = [
        ":session_factory",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:external_file_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
    ],
)

cc_test(
    name = "engine_impl_test",
    srcs = ["engine_impl_test.cc"],
    # The LiteRT GPU path is not ready yet. Only test the CPU path.
    args = ["--gunit_filter=EngineTest.CreateEngine"],
    data = ["//runtime/testdata"],
    deps = [
        ":engine_impl",  # buildcleaner: keep
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/executor:llm_executor_settings",
    ],
)

cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/components:sampler",
        "//runtime/components:token_id_util",
        "//runtime/components:tokenizer",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.cc"],
    data = [
        "//runtime/components/testdata:sentencepiece.model",
    ],
    deps = [
        ":pipeline",
        "@com_google_googletest//:gtest_main",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "session_basic",
    srcs = ["session_basic.cc"],
    hdrs = ["session_basic.h"],
    deps = [
        ":pipeline",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "//runtime/components:sampler",
        "//runtime/components:tokenizer",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/engine:engine_interface",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "session_basic_test",
    srcs = ["session_basic_test.cc"],
    data = [
        "//runtime/components/testdata:sentencepiece.model",
    ],
    deps = [
        ":session_basic",
        "@com_google_googletest//:gtest_main",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor",
    ],
)

cc_library(
    name = "session_factory",
    srcs = ["session_factory.cc"],
    hdrs = ["session_factory.h"],
    deps = [
        ":session_basic",
        "@abseil-cpp//absl/status:statusor",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/executor:llm_executor",
        "//runtime/proto:sampler_params_cc_proto",
    ],
)

cc_test(
    name = "session_factory_test",
    srcs = ["session_factory_test.cc"],
    deps = [
        ":session_factory",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "//runtime/components:tokenizer",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor",
    ],
)
