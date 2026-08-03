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
    name = "engine",
    srcs = ["engine.proto"],
)

alias(
    name = "engine_proto",
    actual = ":engine",
)

alias(
    name = "engine_cc_proto",
    actual = ":engine_cc",
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
    name = "llm_metadata",
    srcs = ["llm_metadata.proto"],
    deps = [
        ":llm_model_type_proto",
        ":sampler_params",
        ":token_proto",
    ],
)

alias(
    name = "llm_metadata_proto",
    actual = ":llm_metadata",
)

alias(
    name = "llm_metadata_cc_proto",
    actual = ":llm_metadata_cc",
)

alias(
    name = "llm_metadata_py_pb2",
    actual = ":llm_metadata_py_proto",
)

tf_proto_library(
    name = "token",
    srcs = ["token.proto"],
)

alias(
    name = "token_proto",
    actual = ":token",
)

alias(
    name = "token_cc_proto",
    actual = ":token_cc",
)

alias(
    name = "token_py_pb2",
    actual = ":token_py_proto",
)

tf_proto_library(
    name = "llm_model_type",
    srcs = ["llm_model_type.proto"],
    deps = [":token_proto"],
)

alias(
    name = "llm_model_type_proto",
    actual = ":llm_model_type",
)

alias(
    name = "llm_model_type_cc_proto",
    actual = ":llm_model_type_cc",
)

alias(
    name = "llm_model_type_py_pb2",
    actual = ":llm_model_type_py_proto",
)

tf_proto_library(
    name = "litert_lm_metrics",
    srcs = ["litert_lm_metrics.proto"],
    deps = [":engine_proto"],
)

alias(
    name = "litert_lm_metrics_proto",
    actual = ":litert_lm_metrics",
)

alias(
    name = "litert_lm_metrics_cc_proto",
    actual = ":litert_lm_metrics_cc",
)

alias(
    name = "litert_lm_metrics_py_pb2",
    actual = ":litert_lm_metrics_py_proto",
)
