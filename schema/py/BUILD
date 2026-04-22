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

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library", py_test = "pytype_strict_contrib_test")
load("@rules_python//python:defs.bzl", "py_test")

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "litertlm_core",
    srcs = ["litertlm_core.py"],
    deps = [
        "//schema/core:litertlm_header_schema_py",
        "@absl_py//absl:app",
    ],
)

py_test(
    name = "litertlm_core_test",
    srcs = ["litertlm_core_test.py"],
    deps = [
        ":litertlm_core",
        "//schema/core:litertlm_header_schema_py",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "litertlm_peek",
    srcs = ["litertlm_peek.py"],
    deps = [
        ":litertlm_core",
        "//runtime/proto:llm_metadata_py_pb2",
        "//schema/core:litertlm_header_schema_py",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_binary(
    name = "litertlm_peek_main",
    srcs = ["litertlm_peek_main.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":litertlm_core",
        ":litertlm_peek",
    ],
)

py_test(
    name = "litertlm_peek_test",
    srcs = ["litertlm_peek_test.py"],
    data = ["//schema:testdata"],
    main = "litertlm_peek_test.py",
    deps = [
        ":litertlm_builder",
        ":litertlm_peek",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "litertlm_builder",
    srcs = ["litertlm_builder.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":litertlm_core",
        "@flatbuffers//:runtime_py",
        "//runtime/proto:llm_metadata_py_pb2",
        "//schema/core:litertlm_header_schema_py",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "litertlm_builder_test",
    srcs = ["litertlm_builder_test.py"],
    deps = [
        ":litertlm_builder",
        ":litertlm_core",
        ":litertlm_peek",
        "//runtime/proto:llm_metadata_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_binary(
    name = "litertlm_builder_cli",
    srcs = ["litertlm_builder_cli.py"],
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":litertlm_builder",
        ":litertlm_core",
    ],
)

py_test(
    name = "litertlm_builder_cli_test",
    srcs = ["litertlm_builder_cli_test.py"],
    data = [
        ":litertlm_builder_cli",
    ],
    deps = [
        ":litertlm_core",
        ":litertlm_peek",
        "@rules_python//python/runfiles",
        "//runtime/proto:llm_metadata_py_pb2",
        "@absl_py//absl/testing:absltest",
    ],
)
