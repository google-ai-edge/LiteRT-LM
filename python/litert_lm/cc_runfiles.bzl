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

"""Rule to expose a cc_binary's runfiles as DefaultInfo files.

On Windows, Bazel places all transitive DLL dependencies alongside the binary
in the output directory. These DLLs are tracked in the cc_binary's runfiles
but are not directly referenceable as separate labels. This rule bridges that
gap by exposing them as explicit DefaultInfo.files so py_wheel can pick them up.
"""

def _cc_runfiles_impl(ctx):
    runfiles = ctx.attr.cc_binary[DefaultInfo].default_runfiles
    return DefaultInfo(files = runfiles.files)

cc_runfiles = rule(
    implementation = _cc_runfiles_impl,
    attrs = {
        "cc_binary": attr.label(
            mandatory = True,
            doc = "The cc_binary target whose runfiles to expose as files.",
        ),
    },
)
