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
    name = "logits_processor",
    hdrs = ["logits_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//tflite/types:half",
    ],
)

cc_library(
    name = "mock_logits_processor",
    testonly = True,
    hdrs = ["mock_logits_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logits_processor",
        "@com_google_googletest//:gtest_for_library",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components/logits_processor/constrained_decoding:constrained_decoder",
        "@litert//tflite/types:half",
    ],
)

cc_library(
    name = "repetition_penalty_config",
    hdrs = ["repetition_penalty_config.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "repetition_penalty_config_test",
    srcs = ["repetition_penalty_config_test.cc"],
    deps = [
        ":repetition_penalty_config",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "repetition_penalty_processor",
    srcs = ["repetition_penalty_processor.cc"],
    hdrs = ["repetition_penalty_processor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logits_processor",
        ":repetition_penalty_config",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "repetition_penalty_processor_test",
    srcs = ["repetition_penalty_processor_test.cc"],
    deps = [
        ":repetition_penalty_config",
        ":repetition_penalty_processor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_layout",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:test_utils",
        "@litert//tflite/types:half",
    ],
)
