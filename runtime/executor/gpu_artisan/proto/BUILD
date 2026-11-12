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

load(
    "@org_tensorflow//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

tf_proto_library(
    name = "transformer_params",
    srcs = ["transformer_params.proto"],
)

alias(
    name = "transformer_params_proto",
    actual = ":transformer_params",
)

alias(
    name = "transformer_params_cc_proto",
    actual = ":transformer_params_cc",
)

alias(
    name = "transformer_params_py_pb2",
    actual = ":transformer_params_py_proto",
)

tf_proto_library(
    name = "audio_encoder_params",
    srcs = ["audio_encoder_params.proto"],
    deps = [":transformer_params"],
)

alias(
    name = "audio_encoder_params_proto",
    actual = ":audio_encoder_params",
)

alias(
    name = "audio_encoder_params_cc_proto",
    actual = ":audio_encoder_params_cc",
)

alias(
    name = "audio_encoder_params_py_pb2",
    actual = ":audio_encoder_params_py_proto",
)

tf_proto_library(
    name = "vision_encoder_params",
    srcs = ["vision_encoder_params.proto"],
)

alias(
    name = "vision_encoder_params_proto",
    actual = ":vision_encoder_params",
)

alias(
    name = "vision_encoder_params_cc_proto",
    actual = ":vision_encoder_params_cc",
)

alias(
    name = "vision_encoder_params_py_pb2",
    actual = ":vision_encoder_params_py_proto",
)

tf_proto_library(
    name = "prompt_template",
    srcs = ["prompt_template.proto"],
)

alias(
    name = "prompt_template_proto",
    actual = ":prompt_template",
)

alias(
    name = "prompt_template_cc_proto",
    actual = ":prompt_template_cc",
)

alias(
    name = "prompt_template_py_pb2",
    actual = ":prompt_template_py_proto",
)

tf_proto_library(
    name = "sampler_params",
    srcs = ["sampler_params.proto"],
)

alias(
    name = "sampler_params_proto",
    actual = ":sampler_params",
)

alias(
    name = "sampler_params_cc_proto",
    actual = ":sampler_params_cc",
)

alias(
    name = "sampler_params_py_pb2",
    actual = ":sampler_params_py_proto",
)

tf_proto_library(
    name = "llm_params",
    srcs = ["llm_params.proto"],
    deps = [
        ":audio_encoder_params",
        ":prompt_template",
        ":transformer_params",
        ":vision_encoder_params",
    ],
)

alias(
    name = "llm_params_proto",
    actual = ":llm_params",
)

alias(
    name = "llm_params_cc_proto",
    actual = ":llm_params_cc",
)

alias(
    name = "llm_params_py_pb2",
    actual = ":llm_params_py_proto",
)

tf_proto_library(
    name = "llm_file_metadata",
    srcs = ["llm_file_metadata.proto"],
    deps = [":llm_params"],
)

alias(
    name = "llm_file_metadata_proto",
    actual = ":llm_file_metadata",
)

alias(
    name = "llm_file_metadata_cc_proto",
    actual = ":llm_file_metadata_cc",
)

alias(
    name = "llm_file_metadata_py_pb2",
    actual = ":llm_file_metadata_py_proto",
)

tf_proto_library(
    name = "llm_inference_engine",
    srcs = ["llm_inference_engine.proto"],
    deps = [
        ":prompt_template",
        ":sampler_params",
        "//runtime/executor/proto:constrained_decoding_options_proto",
    ],
)

alias(
    name = "llm_inference_engine_proto",
    actual = ":llm_inference_engine",
)

alias(
    name = "llm_inference_engine_cc_proto",
    actual = ":llm_inference_engine_cc",
)

alias(
    name = "llm_inference_engine_py_pb2",
    actual = ":llm_inference_engine_py_proto",
)
