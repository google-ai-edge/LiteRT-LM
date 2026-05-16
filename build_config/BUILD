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

package(
    default_visibility = [
        "//:__subpackages__",
    ],
)

config_setting(
    name = "android_arm64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "android_x86_64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "ios_arm64",
    values = { "apple_platform_type": "ios", "cpu": "ios_arm64",},
)

config_setting(
    name = "ios_sim_arm64",
    values = { "apple_platform_type": "ios", "cpu": "ios_sim_arm64",},
)

alias(
    name = "patchelf_exe",
    actual = select({
        ":linux_x86_64": "@patchelf_linux_x86_64//:patchelf",
        ":linux_arm64": "@patchelf_linux_arm64//:patchelf",
        "//conditions:default": "@patchelf_linux_x86_64//:patchelf",
    }),
    visibility = ["//visibility:public"],
)
