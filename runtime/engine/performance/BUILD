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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "pixel_cpu_affinity_utils",
    srcs = ["pixel_cpu_affinity_utils.cc"],
    hdrs = ["pixel_cpu_affinity_utils.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
    ],
)

cc_library(
    name = "cpu_performance_utils",
    srcs = ["cpu_performance_utils.cc"] + select({
        "//tools/cc_target_os:android": ["cpu_performance_utils_android.cc"],
        "//tools/cc_target_os:linux-google": ["cpu_performance_utils_linux.cc"],
        "//tools/cc_target_os:linux-google-arm": ["cpu_performance_utils_linux.cc"],
        "//conditions:default": ["cpu_performance_utils_default.cc"],
    }),
    hdrs = ["cpu_performance_utils.h"],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "cpu_performance_utils_test",
    srcs = ["cpu_performance_utils_test.cc"],
    deps = [
        ":cpu_performance_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
    ],
)
