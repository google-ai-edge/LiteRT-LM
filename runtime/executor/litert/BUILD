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

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

cc_library(
    name = "kv_cache",
    srcs = ["kv_cache.cc"],
    hdrs = ["kv_cache.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_model_types",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "//runtime/executor:common_utils",
        "//runtime/executor:kv_cache_interface",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/util:litert_status_util",
        "//runtime/util:tensor_buffer_util",
    ],
)

cc_test(
    name = "kv_cache_test",
    srcs = ["kv_cache_test.cc"],
    data = [
        "//runtime/testdata",
    ],
    deps = [
        ":kv_cache",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/test:matchers",
        "//runtime/components:model_resources",
        "//runtime/components:model_resources_litert_lm",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_lm_loader",
        "//runtime/util:scoped_file",
        "//runtime/util:test_utils",
    ],
)
