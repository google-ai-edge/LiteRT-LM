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

package(default_visibility = ["//visibility:public"])

kt_android_library(
    name = "litertlm-android",
    srcs = glob(["*.kt"]),
    associates = [
        "//kotlin/kotlin/com/google/ai/edge/litertlm/core:litertlm-core",
        "//kotlin/kotlin/com/google/ai/edge/litertlm/jni:litertlm-jni",
    ],
    deps = [
        "@rules_kotlin//kotlin/compiler:kotlin-reflect",
        "@maven//:com_google_code_gson_gson",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android"
    ],
)
