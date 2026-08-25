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
    name = "llm_litert_npu_compiled_model_executor_utils",
    srcs = ["llm_litert_npu_compiled_model_executor_utils.cc"],
    hdrs = ["llm_litert_npu_compiled_model_executor_utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:prefetch",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/c:litert_layout",
        "@litert//litert/c:litert_model_types",
        "@litert//litert/c:litert_op_code",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/cc/internal:litert_extended_model",
        "//runtime/components:model_resources",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/executor:litert_compiled_model_executor_utils",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_processed_tokens",
        "//runtime/executor:llm_executor_settings",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "llm_litert_npu_compiled_model_executor_utils_test",
    srcs = ["llm_litert_npu_compiled_model_executor_utils_test.cc"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "llm_litert_npu_rope",
    srcs = ["llm_litert_npu_rope.cc"],
    hdrs = ["llm_litert_npu_rope.h"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_litert_npu_rope_test",
    srcs = ["llm_litert_npu_rope_test.cc"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        ":llm_litert_npu_rope",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "llm_litert_npu_mask",
    srcs = ["llm_litert_npu_mask.cc"],
    hdrs = ["llm_litert_npu_mask.h"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_litert_npu_mask_test",
    srcs = ["llm_litert_npu_mask_test.cc"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        ":llm_litert_npu_mask",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "llm_litert_npu_kv_cache",
    srcs = ["llm_litert_npu_kv_cache.cc"],
    hdrs = ["llm_litert_npu_kv_cache.h"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_layout",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "//runtime/components:model_resources",
        "//runtime/proto:llm_metadata_cc_proto",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "llm_litert_npu_kv_cache_test",
    srcs = ["llm_litert_npu_kv_cache_test.cc"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        ":llm_litert_npu_kv_cache",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
        "@litert//litert/test:matchers",
    ],
)

cc_library(
    name = "llm_litert_npu_embedder",
    srcs = ["llm_litert_npu_embedder.cc"],
    hdrs = ["llm_litert_npu_embedder.h"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:prefetch",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/c:litert_common",
        "@litert//litert/c:litert_model_types",
        "@litert//litert/c:litert_op_code",
        "@litert//litert/cc:litert_common",
        "@litert//litert/cc:litert_compiled_model",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_expected",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_options",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc/internal:litert_extended_model",
        "//runtime/components:model_resources",
        "//runtime/components/embedding_lookup:embedding_lookup_manager",
        "//runtime/executor:llm_executor_io_types",
        "//runtime/executor:llm_executor_processed_tokens",
        "//runtime/executor:llm_executor_settings",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
        "@litert//tflite/types:half",
    ],
)

cc_test(
    name = "llm_litert_npu_embedder_test",
    srcs = ["llm_litert_npu_embedder_test.cc"],
    deps = [
        ":llm_litert_npu_compiled_model_executor_utils",
        ":llm_litert_npu_embedder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_environment",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_ranked_tensor_type",
        "@litert//litert/cc:litert_tensor_buffer",
        "@litert//litert/cc:litert_tensor_buffer_types",
    ],
)
