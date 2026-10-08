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
    name = "flux2_model_config",
    srcs = ["flux2_model_config.cc"],
    hdrs = ["flux2_model_config.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:image_gen_model_type_cc_proto",
    ],
)

cc_library(
    name = "flux2_math",
    srcs = ["flux2_math.cc"],
    hdrs = ["flux2_math.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "flux2_denoiser_stage",
    srcs = ["flux2_denoiser_stage.cc"],
    hdrs = ["flux2_denoiser_stage.h"],
    deps = [
        ":flux2_math",
        ":flux2_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:stage",
        "//omni/base:tensor_utils",
        "//omni/text2image:text_encoder_stage",
    ],
)

cc_library(
    name = "flux2_vae_decoder_stage",
    srcs = ["flux2_vae_decoder_stage.cc"],
    hdrs = ["flux2_vae_decoder_stage.h"],
    deps = [
        ":flux2_denoiser_stage",
        ":flux2_math",
        ":flux2_model_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:stage",
        "//omni/base:tensor_utils",
        "//omni/text2image:image_decoder",
    ],
)

cc_library(
    name = "flux2_factory",
    srcs = ["flux2_factory.cc"],
    hdrs = ["flux2_factory.h"],
    deps = [
        ":flux2_denoiser_stage",
        ":flux2_model_config",
        ":flux2_vae_decoder_stage",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_macros",
        "//omni/base:io_types",
        "//omni/base:litert_runner",
        "//omni/base:model_resources",
        "//omni/base:model_utils",
        "//omni/base:stage",
        "//omni/text2image:prompt_source",
        "//omni/text2image:text_encoder_stage",
        "//runtime/components:model_resources",
        "//runtime/executor:executor_settings_base",
        "//runtime/proto:image_gen_metadata_cc_proto",
        "//runtime/proto:image_gen_model_type_cc_proto",
        "//support/tokenizer",
    ],
)

cc_test(
    name = "flux2_test",
    srcs = ["flux2_test.cc"],
    deps = [
        ":flux2_denoiser_stage",
        ":flux2_factory",
        ":flux2_math",
        ":flux2_model_config",
        ":flux2_vae_decoder_stage",
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
        "//omni/base:model_resources",
        "//omni/base:stage",
        "//omni/text2image:prompt_source",
        "//omni/text2image:text_encoder_stage",
        "//runtime/executor:executor_settings_base",
        "//support/tokenizer",
        "//support/util:test_utils",
    ],
)
