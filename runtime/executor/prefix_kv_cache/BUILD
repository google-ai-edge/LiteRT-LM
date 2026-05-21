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

package(
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "prefix_kv_cache",
    srcs = [
        "radix_tree.cc",
        "prefix_kv_cache_manager.cc",
    ],
    hdrs = [
        "radix_tree.h",
        "prefix_kv_cache_manager.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "radix_tree_test",
    srcs = ["radix_tree_test.cc"],
    deps = [
        ":prefix_kv_cache",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "prefix_kv_cache_manager_test",
    srcs = ["prefix_kv_cache_manager_test.cc"],
    deps = [
        ":prefix_kv_cache",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
    ],
)
