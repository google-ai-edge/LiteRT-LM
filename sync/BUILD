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

# [Google-internal load of `cc_binary`]
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
    name = "sha256",
    srcs = ["sha256.cc"],
    hdrs = ["sha256.h"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "sha256_test",
    srcs = ["sha256_test.cc"],
    deps = [
        ":sha256",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "model_sync",
    srcs = ["model_sync.cc"],
    hdrs = ["model_sync.h"],
    deps = [
        ":sha256",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_read",
    ],
)

cc_test(
    name = "model_sync_test",
    srcs = ["model_sync_test.cc"],
    deps = [
        ":model_sync",
        ":sha256",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_read",
        "//support/util:test_utils",
    ],
)

cc_binary(
    name = "model_sync_cli",
    srcs = ["model_sync_cli.cc"],
    deps = [
        ":model_sync",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
