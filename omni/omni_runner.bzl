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

"""Bazel macro for defining custom LiteRT-Omni runner binaries."""

# [Google-internal load of `cc_binary`]
load("@rules_cc//cc:defs.bzl", "cc_binary")

_OMNI_RUNNER_MAIN = Label("//omni:omni_runner_main")

def omni_runner_binary(
        name,
        models = [],
        deps = [],
        **kwargs):
    """Defines a `cc_binary` reusing the LiteRT-Omni CLI with selected models.

    Links the model-free `omni_runner_main` library (CLI entrypoint, flags, and
    default GPU accelerator) together with only the `alwayslink = 1` model
    registration libraries listed in `models`, avoiding pulling in unrelated
    built-in Omni models.

    Args:
      name: Name of the generated `cc_binary` target.
      models: List (or `select`) of model registration `cc_library` labels
        (typically with `alwayslink = 1` registering via `OmniRegistry`) to
        link into the runner binary.
      deps: Additional C++ library dependencies for the binary.
      **kwargs: Additional keyword arguments forwarded to `cc_binary`
        (e.g., `copts`, `linkopts`, `visibility`, `tags`, `data`, `testonly`).
    """
    all_deps = [_OMNI_RUNNER_MAIN]
    if models != None:
        all_deps = all_deps + models
    if deps != None:
        all_deps = all_deps + deps
    cc_binary(
        name = name,
        deps = all_deps,
        **kwargs
    )
