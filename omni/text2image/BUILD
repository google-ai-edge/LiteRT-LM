# Copyright 2026 The ODML Authors.
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "prompt_source",
    srcs = ["prompt_source.cc"],
    hdrs = ["prompt_source.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_library(
    name = "image_decoder",
    hdrs = ["image_decoder.h"],
    deps = [
        "//omni/base:io_types",
        "//omni/base:stage",
    ],
)

cc_test(
    name = "prompt_source_test",
    srcs = ["prompt_source_test.cc"],
    deps = [
        ":image_decoder",
        ":prompt_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "//omni/base:io_types",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "text_encoder_stage",
    srcs = ["text_encoder_stage.cc"],
    hdrs = ["text_encoder_stage.h"],
    deps = [
        ":prompt_source",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:stage",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "text_encoder_stage_test",
    srcs = ["text_encoder_stage_test.cc"],
    deps = [
        ":prompt_source",
        ":text_encoder_stage",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:mock_litert_runner",
        "//support/tokenizer",
        "//support/util:test_utils",
    ],
)

cc_library(
    name = "text2image_engine",
    srcs = ["text2image_engine.cc"],
    hdrs = ["text2image_engine.h"],
    deps = [
        ":prompt_source",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "//omni:multi_staged_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2image/flux2:flux2_factory",
        "//omni/text2image/flux2:flux2_model_config",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/framework:threadpool",
        "//runtime/proto:image_gen_metadata_cc_proto",
        "//runtime/proto:image_gen_model_type_cc_proto",
    ],
)

cc_library(
    name = "text2image_session",
    srcs = ["text2image_session.cc"],
    hdrs = ["text2image_session.h"],
    deps = [
        ":prompt_source",
        ":text2image_engine",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "//omni:omni_session",
        "//omni/base:io_types",
    ],
)

cc_test(
    name = "text2image_engine_test",
    srcs = ["text2image_engine_test.cc"],
    deps = [
        ":image_decoder",
        ":prompt_source",
        ":text2image_engine",
        ":text2image_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@flatbuffers",
        "//omni:multi_staged_session",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//omni/text2image/flux2:flux2_model_config",
        "//runtime/framework:threadpool",
        "//runtime/proto:image_gen_metadata_cc_proto",
        "//runtime/proto:image_gen_model_type_cc_proto",
        "//schema/core:litertlm_header_schema",
        "//support/util:test_utils",
    ],
)
