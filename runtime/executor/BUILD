
package(
    default_applicable_licenses = ["//:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "litert_compiled_model_executor_utils",
    srcs = ["litert_compiled_model_executor_utils.cc"],
    hdrs = ["litert_compiled_model_executor_utils.h"],
    deps = [
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@litert//litert/litert/cc:litert_buffer_ref",
        "@litert//litert/litert/cc:litert_expected",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:external_file_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
    ],
)

cc_library(
    name = "llm_litert_compiled_model_executor",
    srcs = ["llm_litert_compiled_model_executor.cc"],
    hdrs = ["llm_litert_compiled_model_executor.h"],
    deps = [
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_settings",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/c:litert_common",
        "@litert//litert/litert/cc:litert_compiled_model",
        "@litert//litert/litert/cc:litert_environment",
        "@litert//litert/litert/cc:litert_expected",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_options",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "@litert//litert/litert/runtime/accelerators/gpu:accelerator_options",
        "@litert//litert/litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_litert_compiled_model_executor_test",
    srcs = ["llm_litert_compiled_model_executor_test.cc"],
    data = ["//runtime/testdata"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":litert_compiled_model_executor_utils",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@litert//litert/litert/cc:litert_buffer_ref",
        "@litert//litert/litert/cc:litert_expected",
        "@litert//litert/litert/cc:litert_model",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
    ],
)

cc_library(
    name = "fake_llm_executor",
    srcs = ["fake_llm_executor.cc"],
    hdrs = ["fake_llm_executor.h"],
    deps = [
        ":llm_executor",
        ":llm_executor_base",
        ":llm_executor_settings",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "fake_llm_executor_test",
    srcs = ["fake_llm_executor_test.cc"],
    deps = [
        ":fake_llm_executor",
        ":llm_executor",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "llm_executor",
    srcs = ["llm_executor.cc"],
    hdrs = ["llm_executor.h"],
    deps = [
        ":llm_executor_base",
        "//runtime/util:logging_tensor_buffer",
    ],
)

cc_library(
    name = "llm_executor_base",
    hdrs = ["llm_executor_base.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "llm_executor_test",
    srcs = ["llm_executor_test.cc"],
    deps = [
        ":llm_executor",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/litert/c:litert_tensor_buffer",
        "@litert//litert/litert/cc:litert_element_type",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)

cc_library(
    name = "llm_executor_settings",
    srcs = ["llm_executor_settings.cc"],
    hdrs = ["llm_executor_settings.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//runtime/util:logging",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "llm_executor_settings_test",
    srcs = ["llm_executor_settings_test.cc"],
    deps = [
        ":llm_executor_settings",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "llm_executor_io_types",
    srcs = ["llm_executor_io_types.cc"],
    hdrs = ["llm_executor_io_types.h"],
    deps = [
        "@litert//litert/litert/cc:litert_tensor_buffer",
        "//runtime/util:logging_tensor_buffer",
    ],
)

cc_test(
    name = "llm_executor_io_types_test",
    srcs = ["llm_executor_io_types_test.cc"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_googletest//:gtest_main",
        "@litert//litert/litert/c:litert_dispatch_headers",
        "@litert//litert/litert/cc:litert_element_type",
        "@litert//litert/litert/cc:litert_layout",
        "@litert//litert/litert/cc:litert_model",
        "@litert//litert/litert/cc:litert_tensor_buffer",
    ],
)
