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

load("//models:chat_template_test.bzl", "chat_template_test")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

exports_files([
    "chat_template_e2b_e4b.jinja",
    "chat_template.jinja",
    "LlmMetadataProto.pbtext",
    "LlmMetadataProto_12b.pbtext",
    "LlmMetadataProto_26b_31b.pbtext",
])

# To run the test:
# bazel test -c opt models/gemma4:chat_template_e2b_e4b_test
#
# To update the golden:
# bazel run -c opt models/gemma4:chat_template_e2b_e4b_test -- --update_golden
chat_template_test(
    name = "chat_template_e2b_e4b_test",
    chat_template = "chat_template_e2b_e4b.jinja",
    golden_dir = "testdata/golden",
    input_dir = "testdata/input",
    pbtext_files = [
        "LlmMetadataProto.pbtext",
    ],
)

# To run the test:
# bazel test -c opt models/gemma4:chat_template_test
#
# To update the golden:
# bazel run -c opt models/gemma4:chat_template_test -- --update_golden
chat_template_test(
    name = "chat_template_test",
    chat_template = "chat_template.jinja",
    golden_dir = "testdata/golden_large",
    input_dir = "testdata/input",
    pbtext_files = [
        "LlmMetadataProto_12b.pbtext",
        "LlmMetadataProto_26b_31b.pbtext",
    ],
)
