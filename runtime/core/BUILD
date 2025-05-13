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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:llm_litert_compiled_model_executor",
        "//runtime/framework:thread_options",
        "//runtime/framework:threadpool_interface",
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
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc/internal:litert_tensor_buffer",
        "//runtime/components:sampler",
        "//runtime/components:token_id_util",
        "//runtime/components:tokenizer",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_io_types",
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
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "session_basic",
    srcs = ["session_basic.cc"],
    hdrs = ["session_basic.h"],
    deps = [
        ":pipeline",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "//runtime/components:sampler",
        "//runtime/components:sampler_factory",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/executor:llm_executor_settings",
        "//runtime/framework:threadpool_interface",
        "//runtime/framework:threadpool_pthread_impl",
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
        "@com_google_absl//absl/time",
        "//runtime/components:sentencepiece_tokenizer",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor",
        "//runtime/framework:thread_options",
        "//runtime/framework:threadpool_interface",
        "//runtime/framework:threadpool_pthread_impl",
        "//runtime/util:test_utils",
    ],
)

cc_library(
    name = "session_factory",
    srcs = ["session_factory.cc"],
    hdrs = ["session_factory.h"],
    deps = [
        ":session_basic",
        "@com_google_absl//absl/status:statusor",
        "//runtime/components:tokenizer",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "//runtime/executor:llm_executor",
        "//runtime/framework:threadpool_interface",
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
        "//runtime/executor:fake_llm_executor",
        "//runtime/executor:llm_executor",
        "//runtime/util:test_utils",
    ],
)
