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
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "capabilities",
    hdrs = ["capabilities.h"],
    deps = [":speculative_decoding"],
)

cc_library(
    name = "capabilities_c",
    srcs = ["capabilities_c.cc"],
    hdrs = ["capabilities_c.h"],
    deps = [
        ":capabilities",
    ],
)

cc_library(
    name = "speculative_decoding",
    srcs = ["speculative_decoding.cc"],
    hdrs = ["speculative_decoding.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//runtime/util:litert_status_util",
        "//schema/core:litertlm_header_schema",
        "//schema/core:litertlm_read",
    ],
)

cc_test(
    name = "speculative_decoding_test",
    srcs = ["speculative_decoding_test.cc"],
    deps = [
        ":speculative_decoding",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@flatbuffers",
        "//schema/core:litertlm_header",
        "//schema/core:litertlm_header_schema",
    ],
)
