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

"""Experimental flags for LiteRT-LM."""

from __future__ import annotations


class ExperimentalFlags:
  """Experimental flags for LiteRT-LM.

  This class is experimental and may change or be removed without notice.
  Flags are configured as class attributes.
  """

  # Whether to enable the AI Edge Debugger, which captures per-token traces
  # and, on CPU/GPU, intermediate activations under
  # `<cache_dir>/litert_lm_debugger/<session_id>/`.
  #
  # Requires `cache_dir` set to a writable directory (not ":nocache" or
  # ":memory"); otherwise engine creation fails. Not supported when weights are
  # streamed. If None, uses the native engine's default (off). Captures hold
  # unencrypted user data; enable only for development or diagnostics.
  #
  # Note: Read only when a new Engine is created. Changing this value does not
  # affect existing Engine, Conversation, or Session instances.
  enable_debugger: bool | None = None
