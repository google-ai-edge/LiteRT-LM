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

"""Unit tests for the LiteRT-LM run command."""

import contextlib
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import prompt_toolkit

import litert_lm
from litert_lm_cli import model
import litert_lm_cli.commands.run as run_cmd


class _FakeChunk:

  def __init__(self, text: str):
    self._text = text
    self.channels: dict[str, str] = {}

  def __str__(self) -> str:
    return self._text


class RunTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_home = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_home.cleanup)
    self.enter_context(
        mock.patch.dict("os.environ", {"HOME": self.temp_home.name})
    )

    self.mock_model = mock.MagicMock(spec=model.Model)
    self.mock_model.exists.return_value = True
    self.mock_model.model_path = "/path/to/model.litertlm"
    self.mock_model.to_str.return_value = "test-model"

    self.enter_context(
        mock.patch.object(
            model,
            "resolve_config_option",
            side_effect=lambda val, model_obj, key: val,
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            model,
            "resolve_ringbuffers_local_attention",
            side_effect=lambda val, model_obj: val,
            autospec=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            model,
            "parse_backend",
            return_value=None,
            autospec=True,
        )
    )

    self.events: list[str] = []
    self.conversations: list[mock.MagicMock] = []
    self.sessions: list[mock.MagicMock] = []

    @contextlib.contextmanager
    def make_conversation(**kwargs):
      del kwargs
      idx = len(self.conversations)
      conv = mock.MagicMock(spec=litert_lm.AbstractConversation)
      conv.send_message_async.return_value = iter([_FakeChunk(f"reply-{idx}")])
      self.conversations.append(conv)
      self.events.append(f"enter_conv_{idx}")
      try:
        yield conv
      finally:
        self.events.append(f"exit_conv_{idx}")

    @contextlib.contextmanager
    def make_session(**kwargs):
      del kwargs
      idx = len(self.sessions)
      sess = mock.MagicMock(spec=litert_lm.AbstractSession)
      sess.run_decode_async.return_value = iter([])
      self.sessions.append(sess)
      self.events.append(f"enter_sess_{idx}")
      try:
        yield sess
      finally:
        self.events.append(f"exit_sess_{idx}")

    self.mock_engine = mock.MagicMock()
    self.mock_engine.create_conversation.side_effect = make_conversation
    self.mock_engine.create_session.side_effect = make_session

    @contextlib.contextmanager
    def make_engine(*args, **kwargs):
      del args, kwargs
      self.events.append("enter_engine")
      try:
        yield self.mock_engine
      finally:
        self.events.append("exit_engine")

    self.mock_engine_cls = self.enter_context(
        mock.patch.object(
            litert_lm,
            "Engine",
            side_effect=make_engine,
            autospec=True,
        )
    )

    self.mock_prompt_session = mock.MagicMock()
    self.enter_context(
        mock.patch.object(
            prompt_toolkit,
            "PromptSession",
            return_value=self.mock_prompt_session,
            autospec=True,
        )
    )

  @parameterized.parameters("/clear", "  /clear  ", "/reset")
  def test_interactive_reset_command_recreates_conversation_without_reload(
      self, reset_cmd
  ):
    self.mock_prompt_session.prompt.side_effect = [
        "First question",
        reset_cmd,
        "Second question",
        EOFError,
    ]

    run_cmd.run_interactive(self.mock_model)

    # Engine is created only once; conversation is created twice.
    self.mock_engine_cls.assert_called_once()
    self.assertLen(self.conversations, 2)
    self.conversations[0].send_message_async.assert_called_once_with(
        "First question"
    )
    self.conversations[1].send_message_async.assert_called_once_with(
        "Second question"
    )
    # Existing conversation must be closed before creating the replacement.
    self.assertEqual(
        self.events,
        [
            "enter_engine",
            "enter_conv_0",
            "exit_conv_0",
            "enter_conv_1",
            "exit_conv_1",
            "exit_engine",
        ],
    )

  def test_interactive_clear_command_recreates_raw_session(self):
    self.mock_prompt_session.prompt.side_effect = [
        "Raw prompt 1",
        "/clear",
        "Raw prompt 2",
        EOFError,
    ]

    run_cmd.run_interactive(self.mock_model, no_template=True)

    self.mock_engine_cls.assert_called_once()
    self.assertLen(self.sessions, 2)
    self.sessions[0].run_prefill.assert_called_once_with(["Raw prompt 1"])
    self.sessions[1].run_prefill.assert_called_once_with(["Raw prompt 2"])
    self.assertEqual(
        self.events,
        [
            "enter_engine",
            "enter_sess_0",
            "exit_sess_0",
            "enter_sess_1",
            "exit_sess_1",
            "exit_engine",
        ],
    )

  def test_non_exact_slash_prompt_is_forwarded_to_model(self):
    self.mock_prompt_session.prompt.side_effect = [
        "/clearly explain gravity",
        EOFError,
    ]

    run_cmd.run_interactive(self.mock_model)

    self.assertLen(self.conversations, 1)
    self.conversations[0].send_message_async.assert_called_once_with(
        "/clearly explain gravity"
    )

  def test_single_prompt_flag_does_not_intercept_clear_literal(self):
    run_cmd.run_interactive(self.mock_model, prompt="/clear")

    self.assertLen(self.conversations, 1)
    self.conversations[0].send_message_async.assert_called_once_with("/clear")

  def test_interactive_thinking_and_interrupt_handling(self):
    self.mock_prompt_session.prompt.side_effect = [
        "",
        KeyboardInterrupt,
        "Hello",
        EOFError,
    ]

    run_cmd.run_interactive(
        self.mock_model,
        thinking=True,
        attachments=("/tmp/img.png",),
    )

    self.assertLen(self.conversations, 1)
    self.conversations[0].send_message_async.assert_called_once()


if __name__ == "__main__":
  absltest.main()
