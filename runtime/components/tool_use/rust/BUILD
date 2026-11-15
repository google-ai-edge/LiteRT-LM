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

load("//:rust_cxx_bridge.bzl", "rust_cxx_bridge")
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

package(
    default_visibility = [
        ":default_visibility",
    ],
)

package_group(
    name = "default_visibility",
    packages = [
        "//runtime/components/tool_use/...",
    ],
)

rust_library(
    name = "fc_parser",
    srcs = [
        "fc_parser.rs",
        "fc_parser_test.rs",
    ],
    deps = [
        "//runtime/components/tool_use/antlr/generated:antlr_fc_tool_call_parser",
        "@crate_index//:antlr4rust",
        "@crate_index//:serde_json",
    ],
)

rust_library(
    name = "python_parser",
    srcs = [
        "python_parser.rs",
    ],
    deps = [
        "//runtime/components/tool_use/antlr/generated:antlr_python_tool_call_parser",
        "@crate_index//:antlr4rust",
        "@crate_index//:serde_json",
    ],
)

rust_library(
    name = "json_parser",
    srcs = [
        "json_parser.rs",
    ],
    deps = [
        "@crate_index//:serde_json",
    ],
)

rust_library(
    name = "parsers",
    srcs = [
        "parsers.rs",
        "parsers_test.rs",
    ],
    deps = [
        ":python_parser",
        ":fc_parser",
        ":json_parser",
        "@crate_index//:cxx",
        "@crate_index//:indexmap",
        "@crate_index//:serde_json",
    ],
)

rust_cxx_bridge(
    name = "parsers_cpp",
    src = "parsers.rs",
    deps = [
        ":parsers",
    ] + select({
        "@platforms//os:windows": [
            "//rust:alloc_defs",
            "//rust:global_allocator",
        ],
        "//conditions:default": [],
    }),
)

rust_test(
    name = "fc_parser_test",
    crate = ":fc_parser",
    tags = ["noandroid"],
)

rust_test(
    name = "parsers_test",
    crate = ":parsers",
    tags = ["noandroid"],
)
