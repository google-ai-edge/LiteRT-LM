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
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "engine_impl",
    srcs = ["engine_impl.cc"],
    deps = [
        ":session_factory",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@litert//litert/cc:litert_macros",
        "//runtime/components:model_resources",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:audio_litert_compiled_model_executor",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor",
        "//runtime/executor:llm_litert_npu_compiled_model_executor",
        "//runtime/executor:magic_number_configs_helper",
        "//runtime/executor:vision_executor",
        "//runtime/executor:vision_executor_settings",
        "//runtime/executor:vision_litert_compiled_model_executor",
        "//runtime/framework:threadpool",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:file_format_util",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_environment",
        ],
    }),
)

cc_test(
    name = "engine_impl_test",
    srcs = ["engine_impl_test.cc"],
    # The LiteRT GPU path is not ready yet. Only test the CPU path.
    args = ["--gunit_filter=-EngineTest.CreateEngineGPU*"],
    data = ["//runtime/testdata"],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":engine_impl",  # buildcleaner: keep
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//runtime/components:sampler",
        "//runtime/components:scoring_cpu_util",
        "//runtime/components:stop_token_detector",
        "//runtime/components:token_id_util",
        "//runtime/components:tokenizer",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/constrained_decoding:constrained_decoder",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor",
        "//runtime/proto:sampler_params_cc_proto",
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
    name = "pipeline_test",
    srcs = ["pipeline_test.cc"],
    data = ["//runtime/components/testdata"],
    deps = [
        ":pipeline",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:stop_token_detector",
        "//runtime/components:tokenizer",
        "//runtime/components:top_p_cpu_sampler",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "session_basic",
    srcs = ["session_basic.cc"],
    hdrs = ["session_basic.h"],
    deps = [
        ":pipeline",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_tensor_buffer_types",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components:stop_token_detector",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:constraint",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:vision_executor",
        "//runtime/framework:threadpool",
        "//runtime/proto:llm_model_type_cc_proto",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:model_type_utils",
        "//runtime/util:tensor_buffer_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_test(
    name = "session_basic_test",
    srcs = ["session_basic_test.cc"],
    data = [
        "//runtime/components/testdata",
        "//runtime/testdata",
    ],
    tags = ["requires-mac-inputs:hard"],  # Required for running on Forge on Mac.
    deps = [
        ":session_basic",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/test:matchers",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/components/constrained_decoding:fake_constraint",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor_settings",
        "//runtime/executor:audio_litert_compiled_model_executor",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/framework:threadpool",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "//runtime/util:scoped_file",
        "//runtime/util:tensor_buffer_util",
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
    name = "session_factory",
    srcs = ["session_factory.cc"],
    hdrs = ["session_factory.h"],
    deps = [
        ":session_basic",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:audio_executor",
        "//runtime/executor:llm_executor",
        "//runtime/executor:vision_executor",
        "//runtime/framework:threadpool",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "session_factory_test",
    srcs = ["session_factory_test.cc"],
    deps = [
        ":session_factory",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor",
        "//runtime/framework:threadpool",
        "//runtime/util:test_utils",
    ],
)
