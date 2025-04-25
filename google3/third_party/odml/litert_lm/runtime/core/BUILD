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
    default_applicable_licenses = ["//third_party/odml:license"],
    default_hdrs_check = "strict",
    default_visibility = [
        "//third_party/odml/litert_lm:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "engine_impl",
    srcs = ["engine_impl.cc"],
    deps = [
        ":session_factory",
        "//third_party/absl/log",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/log:check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/odml/litert_lm/runtime/components:sentencepiece_tokenizer",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/engine:engine_interface",
        "//third_party/odml/litert_lm/runtime/engine:llm_model_settings",
        "//third_party/odml/litert_lm/runtime/executor:litert_compiled_model_executor_utils",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor_config",
        "//third_party/odml/litert_lm/runtime/executor:llm_litert_compiled_model_executor",
        "//third_party/odml/litert_lm/runtime/proto:sampler_params_cc_proto",
        "//third_party/odml/litert_lm/runtime/util:external_file_cc_proto",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
        "//third_party/odml/litert_lm/runtime/util:model_asset_bundle_resources",
    ],
)

cc_test(
    name = "engine_impl_test",
    srcs = ["engine_impl_test.cc"],
    # The LiteRT GPU path is not ready yet. Only test the CPU path.
    args = ["--gunit_filter=EngineTest.CreateEngine"],
    data = ["//third_party/odml/litert_lm/runtime/testdata"],
    deps = [
        ":engine_impl",  # buildcleaner: keep
        "@com_google_googletest//:gtest_main",
        "//third_party/absl/log:absl_check",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "//third_party/odml/litert_lm/runtime/engine:engine_interface",
        "//third_party/odml/litert_lm/runtime/engine:llm_model_settings",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor_config",
    ],
)

cc_library(
    name = "pipeline",
    srcs = ["pipeline.cc"],
    hdrs = ["pipeline.h"],
    deps = [
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "@litert//litert/cc:litert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/components:sampler",
        "//third_party/odml/litert_lm/runtime/components:token_id_util",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/engine:io_types",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
        "//third_party/odml/litert_lm/runtime/util:convert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.cc"],
    data = [
        "//third_party/odml/litert_lm/runtime/components/testdata:sentencepiece.model",
    ],
    deps = [
        ":pipeline",
        "@com_google_googletest//:gtest_main",
        "//third_party/odml/litert_lm/runtime/components:sentencepiece_tokenizer",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/components:top_p_cpu_sampler",
        "//third_party/odml/litert_lm/runtime/engine:io_types",
        "//third_party/odml/litert_lm/runtime/executor:fake_llm_executor",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
        "//third_party/odml/litert_lm/runtime/util:convert_tensor_buffer",
    ],
)

cc_library(
    name = "session_basic",
    srcs = ["session_basic.cc"],
    hdrs = ["session_basic.h"],
    deps = [
        ":pipeline",
        "//third_party/absl/log:absl_log",
        "//third_party/absl/memory",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:string_view",
        "//third_party/odml/litert_lm/runtime/components:sampler",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/components:top_p_cpu_sampler",
        "//third_party/odml/litert_lm/runtime/engine:engine_interface",
        "//third_party/odml/litert_lm/runtime/engine:io_types",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
        "//third_party/odml/litert_lm/runtime/proto:sampler_params_cc_proto",
        "//third_party/odml/litert_lm/runtime/util:convert_tensor_buffer",
        "//third_party/odml/litert_lm/runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "session_basic_test",
    srcs = ["session_basic_test.cc"],
    data = [
        "//third_party/odml/litert_lm/runtime/components/testdata:sentencepiece.model",
    ],
    deps = [
        ":session_basic",
        "@com_google_googletest//:gtest_main",
        "//third_party/odml/litert_lm/runtime/components:sentencepiece_tokenizer",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/engine:io_types",
        "//third_party/odml/litert_lm/runtime/executor:fake_llm_executor",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
    ],
)

cc_library(
    name = "session_factory",
    srcs = ["session_factory.cc"],
    hdrs = ["session_factory.h"],
    deps = [
        ":session_basic",
        "//third_party/absl/status:statusor",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/engine:engine_interface",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
        "//third_party/odml/litert_lm/runtime/proto:sampler_params_cc_proto",
    ],
)

cc_test(
    name = "session_factory_test",
    srcs = ["session_factory_test.cc"],
    deps = [
        ":session_factory",
        "@com_google_googletest//:gtest_main",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings:string_view",
        "//third_party/odml/litert_lm/runtime/components:tokenizer",
        "//third_party/odml/litert_lm/runtime/executor:fake_llm_executor",
        "//third_party/odml/litert_lm/runtime/executor:llm_executor",
    ],
)
