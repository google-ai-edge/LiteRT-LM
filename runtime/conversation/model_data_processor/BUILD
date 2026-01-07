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
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "config_registry",
    hdrs = [
        "config_registry.h",
    ],
    deps = [
        ":function_gemma_data_processor_config",
        ":gemma3_data_processor_config",
        ":generic_data_processor_config",
        ":qwen3_data_processor_config",
    ],
)

cc_library(
    name = "data_utils",
    srcs = ["data_utils.cc"],
    hdrs = ["data_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/util:memory_mapped_file",
    ],
)

cc_test(
    name = "data_utils_test",
    srcs = ["data_utils_test.cc"],
    deps = [
        ":data_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/util:memory_mapped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "gemma3_data_processor_config",
    hdrs = [
        "gemma3_data_processor_config.h",
    ],
)

cc_library(
    name = "function_gemma_data_processor_config",
    hdrs = ["function_gemma_data_processor_config.h"],
)

cc_library(
    name = "generic_data_processor_config",
    hdrs = [
        "generic_data_processor_config.h",
    ],
)

cc_library(
    name = "qwen3_data_processor_config",
    hdrs = [
        "qwen3_data_processor_config.h",
    ],
)

cc_library(
    name = "model_data_processor",
    hdrs = ["model_data_processor.h"],
    deps = [
        ":config_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
    ],
)

cc_library(
    name = "model_data_processor_factory",
    srcs = ["model_data_processor_factory.cc"],
    hdrs = ["model_data_processor_factory.h"],
    deps = [
        ":config_registry",
        ":function_gemma_data_processor",
        ":function_gemma_data_processor_config",
        ":gemma3_data_processor",
        ":gemma3_data_processor_config",
        ":generic_data_processor",
        ":generic_data_processor_config",
        ":model_data_processor",
        ":qwen3_data_processor",
        ":qwen3_data_processor_config",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/components:prompt_template",
        "//runtime/components:tokenizer",
        "//runtime/conversation:io_types",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/proto:token_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "model_data_processor_factory_test",
    srcs = ["model_data_processor_factory_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":config_registry",
        ":function_gemma_data_processor_config",
        ":gemma3_data_processor_config",
        ":generic_data_processor_config",
        ":model_data_processor",
        ":model_data_processor_factory",
        ":qwen3_data_processor_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/util:litert_status_util",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "generic_data_processor",
    srcs = ["generic_data_processor.cc"],
    hdrs = ["generic_data_processor.h"],
    deps = [
        ":generic_data_processor_config",
        ":model_data_processor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "generic_data_processor_test",
    srcs = ["generic_data_processor_test.cc"],
    deps = [
        ":generic_data_processor",
        ":generic_data_processor_config",
        "@com_google_googletest//:gtest_main",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "qwen3_data_processor",
    srcs = ["qwen3_data_processor.cc"],
    hdrs = ["qwen3_data_processor.h"],
    deps = [
        ":model_data_processor",
        ":qwen3_data_processor_config",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use:parser_utils",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "qwen3_data_processor_test",
    srcs = ["qwen3_data_processor_test.cc"],
    deps = [
        ":qwen3_data_processor",
        ":qwen3_data_processor_config",
        "@com_google_googletest//:gtest_main",
        "@nlohmann_json//:json",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "gemma3_data_processor",
    srcs = ["gemma3_data_processor.cc"],
    hdrs = ["gemma3_data_processor.h"],
    deps = [
        ":data_utils",
        ":gemma3_data_processor_config",
        ":model_data_processor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "@litert//litert/cc:litert_layout",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:constraint_provider",
        "//runtime/components/constrained_decoding:gemma_model_constraint_provider_lib",
        "//runtime/components/preprocessor:audio_preprocessor",
        "//runtime/components/preprocessor:audio_preprocessor_miniaudio",
        "//runtime/components/preprocessor:image_preprocessor",
        "//runtime/components/preprocessor:stb_image_preprocessor",
        "//runtime/components/tool_use:parser_utils",
        "//runtime/components/tool_use:python_tool_format_utils",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
        "//runtime/util:memory_mapped_file",
        "@com_googlesource_code_re2//:re2",
        "@sentencepiece//:sentencepiece_model_cc_proto",
    ],
)

cc_test(
    name = "gemma3_data_processor_test",
    srcs = ["gemma3_data_processor_test.cc"],
    data = [
        "//runtime/components/preprocessor/testdata",
        "//runtime/components/testdata",
    ],
    deps = [
        ":gemma3_data_processor",
        ":gemma3_data_processor_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "@litert//litert/cc:litert_layout",
        "//runtime/components:prompt_template",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/components/preprocessor:audio_preprocessor",
        "//runtime/components/preprocessor:audio_preprocessor_miniaudio",
        "//runtime/components/preprocessor:image_preprocessor",
        "//runtime/components/preprocessor:stb_image_preprocessor",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "function_gemma_data_processor",
    srcs = ["function_gemma_data_processor.cc"],
    hdrs = ["function_gemma_data_processor.h"],
    deps = [
        ":function_gemma_data_processor_config",
        ":model_data_processor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/components/constrained_decoding:gemma_model_constraint_provider_lib",
        "//runtime/components/tool_use:fc_tool_format_utils",
        "//runtime/components/tool_use:parser_utils",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
        "@sentencepiece//:sentencepiece_model_cc_proto",
    ],
)

cc_test(
    name = "function_gemma_data_processor_test",
    srcs = ["function_gemma_data_processor_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    shard_count = 32,
    deps = [
        ":function_gemma_data_processor",
        ":function_gemma_data_processor_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)
