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
        "//runtime/components/...",
        "//runtime/components/tool_use/...",
    ],
)

rust_library(
    name = "minijinja_template",
    srcs = [
        "minijinja_template.rs",
    ],
    deps = [
        "@crate_index//:cxx",
        "@crate_index//:minijinja",
        "@crate_index//:serde",
        "@crate_index//:serde_json",
        "@crate_index//:chrono",
    ],
)

rust_cxx_bridge(
    name = "minijinja_template_cpp",
    src = "minijinja_template.rs",
    deps = [
        ":minijinja_template",
    ] + select({
        "@platforms//os:windows": [
            "//rust:alloc_defs",
            "//rust:global_allocator",
        ],
        "//conditions:default": [],
    }),
)

rust_test(
    name = "minijinja_template_test",
    crate = ":minijinja_template",
)
