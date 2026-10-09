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

# Makes the CMake-lane stubs referencable from Bazel targets.
#
# prebuilt/ ships libGemmaModelConstraintProvider for every platform except
# windows_arm64, so an ARM64 Windows build has to fall back to the same stub the
# CMake build already uses.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["gemma_model_constraint_provider.cc"])
