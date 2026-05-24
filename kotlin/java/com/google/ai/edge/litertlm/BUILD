# Copyright 2025 Google LLC
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

load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")
load("@rules_kotlin//kotlin:jvm.bzl", "kt_jvm_library")

package(default_visibility = ["//kotlin:__subpackages__"])

kt_jvm_library(
    name = "litertlm-jvm",
    srcs = glob(["*.kt"]),
    resources = ["//kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni"],
    resource_strip_prefix = "kotlin/java",
    deps = [
        "@rules_kotlin//kotlin/compiler:kotlin-reflect",
        "@maven//:com_google_code_gson_gson",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_core_jvm"
    ],
)

# Note: To build with dynamic linking, use the following build flags:
kt_android_library(
    name = "litertlm-android",
    srcs = glob(["*.kt"]),
    deps = [
        "@rules_kotlin//kotlin/compiler:kotlin-reflect",
        "@maven//:com_google_code_gson_gson",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android"
    ] + select({
        # When link LiteRT dynamically, includes the shared lib.
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/c:litert_runtime_c_api_shared_lib",
            "@litert//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator_shared_lib",
        ],
        "//conditions:default": [],
    }),
)
