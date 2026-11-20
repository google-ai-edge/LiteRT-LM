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
    "@org_tensorflow//tensorflow:pytype.default.bzl",
    "pytype_strict_library",
)

load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "model_sync",
    srcs = [
        "__init__.py",
        "_model_sync.py",
    ],
    deps = [
        "//python/litert_lm:_ffi",  # fixdeps: keep
    ],
)

py_test(
    name = "model_sync_test",
    srcs = ["model_sync_test.py"],
    data = ["//runtime/testdata"],
    tags = ["nowindows"],
    deps = [
        ":model_sync",
        "@flatbuffers//:runtime_py",
        "//python/litert_lm_builder:litertlm_header_schema_py",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
    imports = ["../.."],
)
