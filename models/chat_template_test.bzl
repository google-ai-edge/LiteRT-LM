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

"""Starlark macro for testing Jinja chat prompt templates against golden outputs."""

load("@rules_cc//cc:defs.bzl", "cc_test")

def _to_runfiles_path(path):
    """Converts a file path or Bazel label to its expected runfiles path."""
    if path.startswith("//"):
        return path[2:].replace(":", "/")
    if path.startswith(":"):
        path = path[1:]
    return native.package_name() + "/" + path

def _resolve_data_srcs(path, glob_ext):
    """Resolves data targets or globs for data dependencies."""
    if path.startswith("//") or path.startswith(":"):
        return [path]
    else:
        return native.glob([path + "/*" + glob_ext, path + "/**/*" + glob_ext])

def chat_template_test(
        name,
        chat_template,
        input_dir = "testdata/input",
        golden_dir = "testdata/golden",
        pbtext_files = [],
        mirror_chat_templates = [],
        tags = [],
        deps = [],
        data = [],
        **kwargs):
    """Macro to generate a Jinja chat prompt template test.

    Args:
      name: The name of the test target.
      chat_template: The path or label to the .jinja chat template file.
      input_dir: The directory containing input JSON test files (<name>.json).
      golden_dir: The directory containing golden expected rendered text files (<name>.txt).
      pbtext_files: Optional list of LlmMetadataProto.pbtext files to verify that their jinja_prompt_template matches chat_template.
      mirror_chat_templates: Optional list of chat templates mirroring chat_template to verify they are identical.
      tags: Additional tags to pass to cc_test.
      deps: Additional C++ dependencies.
      data: Additional data dependencies.
      **kwargs: Additional keyword arguments passed to cc_test.
    """
    data_files = [chat_template] + _resolve_data_srcs(input_dir, ".json") + _resolve_data_srcs(golden_dir, ".txt") + pbtext_files + mirror_chat_templates + data

    runner_args = [
        "--chat_template=" + _to_runfiles_path(chat_template),
        "--input_dir=" + _to_runfiles_path(input_dir),
        "--golden_dir=" + _to_runfiles_path(golden_dir),
    ]

    if pbtext_files:
        pbtext_runfiles = [_to_runfiles_path(f) for f in pbtext_files]
        runner_args.append("--pbtext_files=" + ",".join(pbtext_runfiles))

    if mirror_chat_templates:
        mirror_runfiles = [_to_runfiles_path(f) for f in mirror_chat_templates]
        runner_args.append("--mirror_chat_templates=" + ",".join(mirror_runfiles))

    cc_test(
        name = name,
        args = runner_args,
        data = data_files,
        tags = tags + ["nomacos", "noios", "nowindows", "noandroid"],
        deps = [
            "//models:chat_template_test_runner_lib",
        ] + deps,
        **kwargs
    )
