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
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "thread_options",
    hdrs = ["thread_options.h"],
)

config_setting(
    name = "litert_lm_std_thread",
    define_values = {
        "litert_lm_std_thread": "true",
    },
)

cc_library(
    name = "threadpool",
    srcs = [
        "threadpool.cc",
        "worker_thread.cc",
    ] + select({
        ":litert_lm_std_thread": ["worker_thread_std_thread.cc"],
        "@platforms//os:windows": ["worker_thread_std_thread.cc"],
        "//conditions:default": ["worker_thread_pthread.cc"],
    }),
    hdrs = [
        "threadpool.h",
        "worker_thread.h",
    ],
    deps = [
        ":thread_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "threadpool_test",
    srcs = ["threadpool_test.cc"],
    deps = [
        ":thread_options",
        ":threadpool",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//runtime/util:test_utils",
    ],
)
