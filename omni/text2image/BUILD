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

# [Google-internal load of `cc_binary`]
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
    name = "text2image_session",
    srcs = ["text2image_session.cc"],
    hdrs = ["text2image_session.h"],
    deps = [
        ":prompt_source",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2image_session_config_cc_proto",
    ],
)

cc_library(
    name = "bonsai_flux2",
    srcs = ["bonsai_flux2.cc"],
    deps = [
        ":prompt_source",
        ":text2image_session",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2image/flux2:flux2_factory",
        "//omni/text2image/flux2:flux2_model_config",
        "//runtime/components:model_resources",
        "//runtime/proto:image_gen_model_type_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2image_session_config_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "flux2_klein",
    srcs = ["flux2_klein.cc"],
    deps = [
        ":prompt_source",
        ":text2image_session",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine_header",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2image/flux2:flux2_factory",
        "//omni/text2image/flux2:flux2_model_config",
        "//runtime/components:model_resources",
        "//runtime/proto:image_gen_model_type_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2image_session_config_cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "text2image_session_test",
    srcs = ["text2image_session_test.cc"],
    deps = [
        ":bonsai_flux2",  # build_cleaner: keep
        ":flux2_klein",  # build_cleaner: keep
        ":image_decoder",
        ":prompt_source",
        ":text2image_session",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_options",
        "//omni:multi_staged_session",
        "//omni:omni_engine",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:stage",
        "//runtime/components:model_resources",
        "//runtime/framework:threadpool",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2image_session_config_cc_proto",
        "//support/util:test_utils",
    ],
)

cc_binary(
    name = "text2image_runner",
    srcs = ["text2image_runner.cc"],
    deps = [
        ":bonsai_flux2",  # build_cleaner: keep
        ":flux2_klein",  # build_cleaner: keep
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//omni:omni_engine",
        "//omni:omni_registry",
        "//omni:omni_session",
        "//omni/base:io_types",
        "//omni/base:model_utils",
        "//runtime/executor:default_static_gpu_accelerator",
        "//runtime/proto:image_gen_model_type_cc_proto",
        "//runtime/proto:text2image_metadata_cc_proto",
        "//runtime/proto:text2image_session_config_cc_proto",
        "@stb//:stb_image_write",
    ],
)
