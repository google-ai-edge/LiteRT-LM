# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [Google-internal load of `cc_library`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "audio_executor_settings",
    srcs = ["audio_executor_settings.cc"],
    hdrs = ["audio_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "audio_executor_settings_test",
    srcs = ["audio_executor_settings_test.cc"],
    deps = [
        ":audio_executor_settings",
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "vision_executor_settings",
    srcs = ["vision_executor_settings.cc"],
    hdrs = ["vision_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "vision_executor_settings_test",
    srcs = ["vision_executor_settings_test.cc"],
    deps = [
        ":executor_settings_base",
        ":vision_executor_settings",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "executor_settings_base",
    srcs = ["executor_settings_base.cc"],
    hdrs = ["executor_settings_base.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:file_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:scoped_file",
    ],
)

cc_test(
    name = "executor_settings_base_test",
    srcs = ["executor_settings_base_test.cc"],
    deps = [
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "litert_compiled_model_executor_utils",
    srcs = ["litert_compiled_model_executor_utils.cc"],
    hdrs = ["litert_compiled_model_executor_utils.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:model_resources_task",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:external_file_cc_proto",
        "//runtime/util:file_format_util",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "litert_compiled_model_executor_utils_test",
    srcs = ["litert_compiled_model_executor_utils_test.cc"],
    data = ["//runtime/testdata"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "default_static_gpu_accelerator",
    deps = select({
        # If CAPI is linked dynamically, need to link the accelerator dynamically as well.
        "@litert//litert:litert_link_capi_so": [],
        "//conditions:default": [
        ],
    }) + select({
        "//conditions:default": [],
    }),
)

cc_library(
    name = "llm_litert_compiled_model_cache_utils",
    srcs = ["llm_litert_compiled_model_cache_utils.cc"],
    hdrs = ["llm_litert_compiled_model_cache_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "llm_litert_compiled_model_cache_utils_test",
    srcs = ["llm_litert_compiled_model_cache_utils_test.cc"],
    deps = [
        ":llm_litert_compiled_model_cache_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_litert_compiled_model_executor",
    srcs = ["llm_litert_compiled_model_executor.cc"],
    hdrs = ["llm_litert_compiled_model_executor.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_cache_utils",
        ":llm_processed_context",
        ":magic_number_configs_helper",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/components:lora_manager",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:model_resources_task",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:file_util",
        "//runtime/util:litert_status_util",
        "//runtime/util:lora_util",
        "//runtime/util:scoped_file",
        "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/c/internal:litert_external_litert_buffer_context",  # buildcleaner: keep
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc/options:litert_cpu_options",
            "@litert//litert/cc/options:litert_gpu_options",
            "@litert//litert/cc/options:litert_runtime_options",
        ],
    }),
)

cc_test(
    name = "llm_litert_compiled_model_executor_test",
    srcs = ["llm_litert_compiled_model_executor_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":default_static_gpu_accelerator",
        ":executor_settings_base",
        ":llm_executor_io_types",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:model_resources_task",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
        ],
    }),
)

cc_library(
    name = "llm_processed_context",
    hdrs = ["llm_processed_context.h"],
    deps = [
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "llm_litert_compiled_model_executor_factory",
    srcs = ["llm_litert_compiled_model_executor_factory.cc"],
    hdrs = ["llm_litert_compiled_model_executor_factory.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor",
        ":llm_litert_npu_compiled_model_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/components:model_resources",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_ranked_tensor_type",
        ],
    }),
)

cc_test(
    name = "llm_litert_compiled_model_executor_factory_test",
    srcs = ["llm_litert_compiled_model_executor_factory_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        ":llm_litert_compiled_model_executor_factory",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_asset_bundle_resources",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
        ],
    }),
)

cc_library(
    name = "fake_llm_executor",
    srcs = ["fake_llm_executor.cc"],
    hdrs = ["fake_llm_executor.h"],
    deps = [
        ":executor_settings_base",
        ":llm_executor",
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "fake_llm_executor_test",
    srcs = ["fake_llm_executor_test.cc"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":fake_llm_executor",
        ":llm_executor_io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/test:matchers",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor",
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
        ":llm_executor_io_types",
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "llm_litert_npu_compiled_model_executor",
    srcs = ["llm_litert_npu_compiled_model_executor.cc"],
    hdrs = ["llm_litert_npu_compiled_model_executor.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor",
        ":llm_executor_io_types",
        ":llm_executor_settings",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/components:model_resources_task",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/executor:llm_executor_processed_tokens",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc/options:litert_google_tensor_options",
            "@litert//litert/cc/options:litert_mediatek_options",
            "@litert//litert/cc/options:litert_qualcomm_options",
        ],
    }),
)

cc_library(
    name = "llm_executor_processed_tokens",
    srcs = ["llm_executor_processed_tokens.cc"],
    hdrs = ["llm_executor_processed_tokens.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "llm_executor_processed_tokens_test",
    srcs = ["llm_executor_processed_tokens_test.cc"],
    deps = [
        ":llm_executor_processed_tokens",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor_settings",
    srcs = ["llm_executor_settings.cc"],
    hdrs = ["llm_executor_settings.h"],
    deps = [
        ":executor_settings_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
        "//runtime/util:logging",
    ],
)

cc_test(
    name = "llm_executor_settings_test",
    srcs = ["llm_executor_settings_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "llm_executor_io_types",
    srcs = ["llm_executor_io_types.cc"],
    hdrs = ["llm_executor_io_types.h"],
    deps = [
        ":llm_executor_processed_tokens",
        ":llm_executor_settings",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/util:logging_tensor_buffer",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "llm_executor_io_types_test",
    srcs = ["llm_executor_io_types_test.cc"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:fake_constraint",
    ],
)

cc_library(
    name = "vision_executor",
    hdrs = ["vision_executor.h"],
    deps = [":vision_executor_base"],
)

cc_library(
    name = "vision_executor_base",
    hdrs = ["vision_executor_base.h"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "vision_litert_compiled_model_executor",
    srcs = ["vision_litert_compiled_model_executor.cc"],
    hdrs = ["vision_litert_compiled_model_executor.h"],
    deps = [
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_io_types",
        ":vision_executor",
        ":vision_executor_settings",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/components:model_resources",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:file_util",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc/options:litert_cpu_options",
            "@litert//litert/cc/options:litert_gpu_options",
            "@litert//litert/cc/options:litert_qualcomm_options",
            "@litert//litert/cc/options:litert_runtime_options",
        ],
    }),
)

cc_test(
    name = "vision_litert_compiled_model_executor_test",
    srcs = ["vision_litert_compiled_model_executor_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":vision_executor_settings",
        ":vision_litert_compiled_model_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
        ],
    }),
)

cc_library(
    name = "audio_executor",
    hdrs = ["audio_executor.h"],
    deps = [":audio_executor_base"],
)

cc_library(
    name = "audio_executor_base",
    hdrs = ["audio_executor_base.h"],
    deps = [
        ":llm_executor_io_types",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "audio_litert_compiled_model_executor",
    srcs = ["audio_litert_compiled_model_executor.cc"],
    hdrs = ["audio_litert_compiled_model_executor.h"],
    deps = [
        ":audio_executor",
        ":audio_executor_settings",
        ":executor_settings_base",
        ":litert_compiled_model_executor_utils",
        ":llm_executor_io_types",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/components:model_resources",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_compiled_model",
            "@litert//litert/cc:litert_element_type",
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_options",
            "@litert//litert/cc:litert_ranked_tensor_type",
            "@litert//litert/cc:litert_tensor_buffer",
            "@litert//litert/cc/options:litert_cpu_options",
            "@litert//litert/cc/options:litert_gpu_options",
            "@litert//litert/cc/options:litert_runtime_options",
        ],
    }),
)

cc_test(
    name = "audio_litert_compiled_model_executor_test",
    srcs = ["audio_litert_compiled_model_executor_test.cc"],
    data = ["//runtime/testdata"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":audio_executor_settings",
        ":audio_litert_compiled_model_executor",
        ":executor_settings_base",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
        ],
    }),
)

cc_library(
    name = "magic_number_configs_helper",
    srcs = ["magic_number_configs_helper.cc"],
    hdrs = ["magic_number_configs_helper.h"],
    deps = [
        ":llm_executor_settings",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc/options:litert_magic_number_options",
        "//runtime/components:model_resources",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
            "@litert//litert/cc:litert_expected",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
        ],
    }),
)

cc_test(
    name = "magic_number_configs_helper_test",
    srcs = ["magic_number_configs_helper_test.cc"],
    data = ["//runtime/testdata"],
    deps = [
        ":executor_settings_base",
        ":llm_executor_settings",
        ":magic_number_configs_helper",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "//runtime/components:model_resources",
        "//runtime/components:tokenizer",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",  # buildcleaner: keep
    ],
)
