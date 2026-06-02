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

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "movable_mutex_lock",
    hdrs = ["movable_mutex_lock.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "movable_mutex_lock_test",
    srcs = ["movable_mutex_lock_test.cc"],
    deps = [
        ":movable_mutex_lock",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "resource_manager_utils",
    srcs = ["resource_manager_utils.cc"],
    hdrs = ["resource_manager_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "resource_manager_utils_test",
    srcs = ["resource_manager_utils_test.cc"],
    deps = [
        ":resource_manager_utils",
        "@com_google_googletest//:gtest_main",
        "//runtime/util:test_utils",
    ],
)
