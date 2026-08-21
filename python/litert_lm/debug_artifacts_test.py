# Copyright 2026 The Google AI Edge Debugger Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the LICENSE file, or at
#
#     http://www.apache.org/licenses/LICENSE-Py/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hermetic unit tests for the LiteRT-LM Session Debug Artifacts Retrieval API."""

import ctypes
import pathlib
import tempfile
from unittest import mock
import warnings

from absl.testing import absltest

import litert_lm


class LiteRtLmSessionDebugArtifactsTest(absltest.TestCase):
  """End-to-End and Pre-condition validation suite for litert_lm.Session.get_debug_artifacts()."""

  def setUp(self):
    """Establishes mocked FFI bindings and Engine/Session contexts."""
    super().setUp()
    self.mock_lib = mock.MagicMock()
    self.mock_ptr = ctypes.c_void_p(12345)
    self.mock_engine = mock.MagicMock()

    # Pre-condition: Assign an initial mockable temporary cache directory
    self.mock_engine.cache_dir = "/tmp/fake_cache_root"

    # Instantiate the target Session abstraction under test using positional
    # parameter injection to satisfy Google3 argument typing.
    self.session = litert_lm.Session(
        self.mock_lib,
        self.mock_ptr,
        self.mock_engine,
    )

  def test_get_debug_artifacts_with_warning_on_macro_disabled(self):
    """Verifies that get_debug_artifacts triggers a RuntimeWarning.

    Yields `None` if the shared library was compiled with
    LITERT_LM_DEBUGGER_ENABLED elided.
    """
    # Simulate Macro elision (`LITERT_LM_DEBUGGER_ENABLED=0`)
    self.mock_lib.litert_lm_experimental_is_debugger_enabled.return_value = (
        False
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
      warnings.simplefilter("always")

      result = self.session.get_debug_artifacts()

      self.assertIsNone(result)
      self.assertLen(captured_warnings, 1)
      self.assertTrue(issubclass(captured_warnings[0].category, RuntimeWarning))
      self.assertIn(
          "LiteRT-LM Debugger is disabled in this runtime build",
          str(captured_warnings[0].message),
      )

  def test_get_debug_artifacts_invalid_session(self):
    """Asserts get_debug_artifacts returns `None` for invalid session."""
    self.mock_lib.litert_lm_experimental_is_debugger_enabled.return_value = True
    # Emulate an invalid/null opaque struct handle returned from the C ABI
    self.mock_lib.litert_lm_experimental_session_get_debug_info.return_value = (
        None
    )

    result = self.session.get_debug_artifacts()
    self.assertIsNone(result)

  def test_get_debug_artifacts_empty_session_dir(self):
    """Tests traversal grace when the session directory does not exist."""
    self.mock_lib.litert_lm_experimental_is_debugger_enabled.return_value = True
    self.mock_lib.litert_lm_experimental_session_get_debug_info.return_value = (
        12345
    )
    self.mock_lib.litert_lm_experimental_session_debug_info_get_capture_dir.return_value = (
        b"litert_lm_debugger/42"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      # Inject the physical temporary context into the parent Engine
      self.mock_engine.cache_dir = tmp_dir

      # Invocation with zero physical subdirectory creation
      result = self.session.get_debug_artifacts()
      self.assertIsNone(result)

    self.mock_lib.litert_lm_experimental_session_debug_info_delete.assert_called_once_with(
        12345
    )

  def test_get_debug_artifacts_populated_session_dir_and_sorting(self):
    """Validates multi-session sandbox isolation and safetensors sorting."""
    self.mock_lib.litert_lm_experimental_is_debugger_enabled.return_value = True
    self.mock_lib.litert_lm_experimental_session_get_debug_info.return_value = (
        12345
    )
    self.mock_lib.litert_lm_experimental_session_debug_info_get_capture_dir.return_value = (
        b"litert_lm_debugger/42"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      # Compute the identical, isolated multi-session subdirectory utilized by
      # Python Session.get_debug_artifacts() and C++ RegisterDebugSession.
      root_path = pathlib.Path(tmp_dir)
      session_dir = root_path / "litert_lm_debugger" / "42"
      session_dir.mkdir(parents=True, exist_ok=True)

      # 1. Populate unordered intermediate step tensors and trace logs
      tensor_step_1 = (
          session_dir / "decode_post_probe_decode_out_0_step_1.safetensors"
      )
      tensor_step_0 = (
          session_dir / "decode_post_probe_decode_out_0_step_0.safetensors"
      )
      tensor_step_2 = (
          session_dir / "decode_post_probe_decode_out_0_step_2.safetensors"
      )
      trace_log = session_dir / "generated_tokens.jsonl"

      for path in [tensor_step_1, tensor_step_0, tensor_step_2, trace_log]:
        path.touch()

      # 2. Assign the localized Temporary Root Anchor to our execution Engine
      self.mock_engine.cache_dir = tmp_dir

      # 3. Trigger Artifact Resolution
      artifacts = self.session.get_debug_artifacts()

      # 4. Rigorous Assertions
      self.assertIsNotNone(artifacts)
      self.assertIsInstance(artifacts, litert_lm.interfaces.DebugArtifacts)

      # Assert monotonic sorting (`_step_0` must precede `_step_1`).
      self.assertEqual(
          artifacts.tensor_paths,
          [tensor_step_0, tensor_step_1, tensor_step_2],
      )

      # Assert Exact Structural Resolution of the Inference Trace Log URI
      self.assertEqual(artifacts.trace_log_path, trace_log)

    self.mock_lib.litert_lm_experimental_session_debug_info_delete.assert_called_once_with(
        12345
    )


class LiteRtLmConversationDebugArtifactsTest(absltest.TestCase):
  """Validation suite for litert_lm.Conversation.get_debug_artifacts()."""

  def setUp(self):
    super().setUp()
    self.mock_lib = mock.MagicMock()
    self.mock_ptr = ctypes.c_void_p(67890)
    self.mock_engine = mock.MagicMock()
    self.mock_engine.cache_dir = "/tmp/fake_cache_root"

    self.conversation = litert_lm.conversation.Conversation(
        self.mock_lib,
        self.mock_ptr,
        engine=self.mock_engine,
    )

  def test_get_debug_artifacts_with_warning_on_macro_disabled(self):
    """Verifies that get_debug_artifacts triggers a RuntimeWarning when disabled."""
    self.mock_lib.litert_lm_experimental_is_debugger_enabled.return_value = (
        False
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
      warnings.simplefilter("always")

      result = self.conversation.get_debug_artifacts()

      self.assertIsNone(result)
      self.assertLen(captured_warnings, 1)
      self.assertTrue(issubclass(captured_warnings[0].category, RuntimeWarning))
      self.assertIn(
          "LiteRT-LM Debugger is disabled in this runtime build",
          str(captured_warnings[0].message),
      )

  def test_get_debug_artifacts_populated_dir(self):
    """Validates retrieval of debug artifacts through Conversation."""
    self.mock_lib.litert_lm_experimental_is_debugger_enabled.return_value = True
    self.mock_lib.litert_lm_experimental_conversation_get_session_debug_info.return_value = (
        67890
    )
    self.mock_lib.litert_lm_experimental_session_debug_info_get_capture_dir.return_value = (
        b"litert_lm_debugger/99"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      root_path = pathlib.Path(tmp_dir)
      session_dir = root_path / "litert_lm_debugger" / "99"
      session_dir.mkdir(parents=True, exist_ok=True)

      tensor_file = (
          session_dir / "decode_post_probe_decode_out_0_step_0.safetensors"
      )
      trace_log = session_dir / "generated_tokens.jsonl"
      tensor_file.touch()
      trace_log.touch()

      self.mock_engine.cache_dir = tmp_dir

      artifacts = self.conversation.get_debug_artifacts()

      self.assertIsNotNone(artifacts)
      self.assertIsInstance(artifacts, litert_lm.interfaces.DebugArtifacts)
      self.assertEqual(artifacts.tensor_paths, [tensor_file])
      self.assertEqual(artifacts.trace_log_path, trace_log)

    self.mock_lib.litert_lm_experimental_session_debug_info_delete.assert_called_once_with(
        67890
    )


if __name__ == "__main__":
  absltest.main()
