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

"""Tests for the messages that end a conversation turn."""

import collections.abc
import dataclasses
import json
import pathlib
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest

import litert_lm

FLAGS = flags.FLAGS

_STATS_JSON = {
    "module_name": "LLM",
    "metrics": {"plain_steps": 3, "acceptance_rate": 0.5},
    "latencies_us": {"plain_step_time": 51234},
    "substats": [{
        "module_name": "Drafter",
        "metrics": {"draft_steps": 2},
        "latencies_us": {},
        "substats": [],
    }],
}
_STATS = litert_lm.ExecutorStats(
    module_name="LLM",
    metrics={"plain_steps": 3, "acceptance_rate": 0.5},
    latencies_in_microseconds={"plain_step_time": 51234},
    substats=(
        litert_lm.ExecutorStats(
            module_name="Drafter", metrics={"draft_steps": 2}
        ),
    ),
)
_ANSWER = {
    "role": "assistant",
    "content": [{"type": "text", "text": "It is sunny."}],
}
_TOOL_CALL = {
    "role": "assistant",
    "tool_calls": [{"function": {"name": "my_tool", "arguments": {}}}],
}
# A tool call without a "function", which fails the turn.
_MALFORMED_TOOL_CALL = {"role": "assistant", "tool_calls": [{}]}


def _end_of_turn_json(finish_reason="stop", executor_stats=None):
  return {
      "role": "assistant",
      "finish_reason": finish_reason,
      "executor_stats": executor_stats,
  }


class EndOfTurnMessageTest(absltest.TestCase):

  def test_parses_end_of_turn_fields(self):
    message = litert_lm.Message.from_json(
        _ANSWER | _end_of_turn_json("max_tokens", _STATS_JSON)
    )

    self.assertEqual(message.finish_reason, "max_tokens")
    self.assertEqual(message.executor_stats, _STATS)
    self.assertEqual(str(message), "It is sunny.")

  def test_parses_the_shared_testdata(self):
    # The tests of every binding parse this file, and the C++ end_of_turn_test
    # checks that it matches the writer.
    path = (
        pathlib.Path(FLAGS.test_srcdir)
        / "litert_lm/runtime/testdata/end_of_turn.json"
    )

    message = litert_lm.Message.from_json(json.loads(path.read_text()))

    self.assertEqual(message.finish_reason, "max_tokens")
    self.assertEqual(message.executor_stats, _STATS)
    self.assertEmpty(message.contents)
    # Unknown keys stay in the message, unlike the end-of-turn fields.
    self.assertEqual(
        message.to_json(),
        {"role": "assistant", "future_field": {"ignored": [1]}},
    )

  def test_end_of_turn_fields_are_not_sent_back(self):
    message = litert_lm.Message.from_json(
        _ANSWER | _end_of_turn_json("stop", _STATS_JSON)
    )

    self.assertEqual(message.to_json(), _ANSWER)
    self.assertEqual(dict(message), _ANSWER)

  def test_null_executor_stats(self):
    message = litert_lm.Message.from_json(_end_of_turn_json("appended"))

    self.assertEqual(message.finish_reason, "appended")
    self.assertIsNone(message.executor_stats)

  def test_other_messages_have_no_end_of_turn_fields(self):
    for message in (
        litert_lm.Message.from_json(_ANSWER),
        litert_lm.Message.from_json(
            _ANSWER | {"finish_reason": None, "executor_stats": _STATS_JSON}
        ),
        litert_lm.Message.model(litert_lm.Contents.of("Hi")),
        litert_lm.Message.user("Hi"),
    ):
      with self.subTest(message=message):
        self.assertIsNone(message.finish_reason)
        self.assertIsNone(message.executor_stats)

  def test_end_of_turn_fields_are_read_only(self):
    message = litert_lm.Message.from_json(_end_of_turn_json())

    with self.assertRaises(AttributeError):
      setattr(message, "finish_reason", "max_tokens")
    with self.assertRaises(AttributeError):
      setattr(message, "executor_stats", _STATS)

  def test_dict_edits_keep_end_of_turn_fields(self):
    message = litert_lm.Message.from_json(
        _ANSWER | _end_of_turn_json("stop", _STATS_JSON)
    )

    message["content"] = "Edited."

    self.assertEqual(message.finish_reason, "stop")
    self.assertEqual(message.executor_stats, _STATS)

  def test_missing_fields_are_empty(self):
    self.assertEqual(
        litert_lm.ExecutorStats.from_json({}), litert_lm.ExecutorStats()
    )

  def test_ignores_malformed_fields(self):
    stats = litert_lm.ExecutorStats.from_json({
        "module_name": None,
        "metrics": {"ok": 1, "text": "1", "flag": True, "none": None},
        "latencies_us": [1, 2],
        "substats": [{"module_name": "Sub"}, "not a dict"],
    })

    self.assertEqual(
        stats,
        litert_lm.ExecutorStats(
            metrics={"ok": 1},
            substats=(litert_lm.ExecutorStats(module_name="Sub"),),
        ),
    )
    message = litert_lm.Message.from_json(
        _end_of_turn_json(executor_stats="bad")
    )
    self.assertIsNone(message.executor_stats)

  def test_ignores_fields_of_the_wrong_type(self):
    stats = litert_lm.ExecutorStats.from_json({
        "module_name": 1,
        "metrics": {"ok": 0.5, "nan": float("nan")},
        "latencies_us": {
            "ok": 7,
            "fractional": 12.7,
            "flag": True,
            "inf": float("inf"),
        },
    })

    self.assertEqual(
        stats,
        litert_lm.ExecutorStats(
            metrics={"ok": 0.5}, latencies_in_microseconds={"ok": 7}
        ),
    )
    self.assertIsNone(
        litert_lm.Message.from_json(_end_of_turn_json(1)).finish_reason
    )

  def test_executor_stats_is_frozen(self):
    with self.assertRaises(dataclasses.FrozenInstanceError):
      setattr(litert_lm.ExecutorStats(), "module_name", "LLM")


class ConversationEndOfTurnTest(absltest.TestCase):
  """Checks how turns end against a fake C API."""

  def setUp(self):
    super().setUp()
    self.mock_tool = mock.create_autospec(litert_lm.Tool, instance=True)
    self.mock_tool.execute.return_value = "sunny"
    self.conv = litert_lm.conversation.Conversation(
        lib=mock.MagicMock(),
        conv_ptr=1,
        tools_map={"my_tool": self.mock_tool},
    )
    self.enter_context(
        mock.patch.object(self.conv, "_create_optional_args", return_value=None)
    )

  def _fake_send_message(
      self, responses: collections.abc.Sequence[collections.abc.Mapping[str, Any]]
  ) -> None:
    """Makes `send_message` return `responses` in order."""
    remaining_responses = iter(responses)

    def create_checked(lib, name, *args):
      del lib, args  # Unused.
      self.assertEqual(name, "litert_lm_conversation_send_message")
      self._next_response = next(remaining_responses)
      return 1

    self.enter_context(
        mock.patch.object(
            litert_lm.conversation, "create_checked", side_effect=create_checked
        )
    )
    self.enter_context(
        mock.patch.object(
            litert_lm.conversation,
            "get_checked",
            side_effect=lambda *_: json.dumps(self._next_response).encode(),
        )
    )

  def _fake_send_message_async(
      self,
      turns: collections.abc.Sequence[
          collections.abc.Sequence[collections.abc.Mapping[str, Any]]
      ],
  ) -> None:
    """Makes `send_message_async` stream `turns` in order.

    Args:
      turns: The chunks of each turn. A chunk is a dict with optional keys
        "text" (message dict), "error" (str), "finish_reason" (str),
        "executor_stats" (dict or str) and "is_final" (bool).
    """
    remaining_turns = iter(turns)
    chunks = {}

    def send_message_stream(conv_ptr, message, context, args, callback, data):
      del conv_ptr, message, context, args  # Unused.
      chunks.clear()
      for chunk_ptr, chunk in enumerate(next(remaining_turns), start=1):
        chunks[chunk_ptr] = chunk
        callback(data, chunk_ptr)
      return 0

    def encode(value):
      if value is None:
        return None
      return (value if isinstance(value, str) else json.dumps(value)).encode()

    def get_checked(lib, name, restype, chunk_ptr):
      del lib, restype  # Unused.
      chunk = chunks[chunk_ptr]
      match name:
        case "litert_lm_stream_chunk_get_error":
          return encode(chunk.get("error"))
        case "litert_lm_stream_chunk_get_text":
          return encode(chunk.get("text"))
        case "litert_lm_stream_chunk_is_final":
          return chunk.get("is_final", False)
        case "litert_lm_stream_chunk_get_finish_reason":
          return encode(chunk.get("finish_reason"))
        case "litert_lm_stream_chunk_get_executor_stats_json":
          return encode(chunk.get("executor_stats"))
        case _:
          raise ValueError(f"Unexpected call to {name}")

    self.conv._lib.litert_lm_conversation_send_message_stream.side_effect = (
        send_message_stream
    )
    self.enter_context(
        mock.patch.object(
            litert_lm.conversation, "get_checked", side_effect=get_checked
        )
    )
    # Calls the callback directly instead of through ctypes.
    self.enter_context(
        mock.patch.object(
            litert_lm.conversation,
            "STREAM_CALLBACK_TYPE",
            side_effect=lambda callback: callback,
        )
    )

  def test_send_message_returns_end_of_turn_fields(self):
    self._fake_send_message([_ANSWER | _end_of_turn_json("stop", _STATS_JSON)])

    message = self.conv.send_message("What's the weather?")

    self.assertEqual(message.to_json(), _ANSWER)
    self.assertEqual(message.finish_reason, "stop")
    self.assertEqual(message.executor_stats, _STATS)

  def test_send_message_returns_last_tool_call_round(self):
    self._fake_send_message([
        _TOOL_CALL | _end_of_turn_json("stop"),
        _ANSWER | _end_of_turn_json("max_tokens", _STATS_JSON),
    ])

    message = self.conv.send_message("What's the weather?")

    self.mock_tool.execute.assert_called_once()
    self.assertEqual(message.to_json(), _ANSWER)
    self.assertEqual(message.finish_reason, "max_tokens")
    self.assertEqual(message.executor_stats, _STATS)

  def test_send_message_async_ends_with_end_of_turn_message(self):
    self._fake_send_message_async([[
        {"text": _ANSWER},
        {
            "is_final": True,
            "finish_reason": "stop",
            "executor_stats": _STATS_JSON,
        },
    ]])

    answer, end_of_turn = self.conv.send_message_async("What's the weather?")

    self.assertEqual(answer.to_json(), _ANSWER)
    self.assertIsNone(answer.finish_reason)
    self.assertEqual(end_of_turn.to_json(), {"role": "assistant"})
    self.assertEqual(end_of_turn.finish_reason, "stop")
    self.assertEqual(end_of_turn.executor_stats, _STATS)

  def test_send_message_async_without_executor_stats(self):
    self._fake_send_message_async(
        [[{"text": _ANSWER}, {"is_final": True, "finish_reason": "max_tokens"}]]
    )

    *_, end_of_turn = self.conv.send_message_async("Hi")

    self.assertEqual(end_of_turn.finish_reason, "max_tokens")
    self.assertIsNone(end_of_turn.executor_stats)

  def test_send_message_async_yields_only_last_tool_call_round_end(self):
    self._fake_send_message_async([
        [
            {"text": _TOOL_CALL},
            {"is_final": True, "finish_reason": "stop"},
        ],
        [
            {"text": _ANSWER},
            {
                "is_final": True,
                "finish_reason": "max_tokens",
                "executor_stats": _STATS_JSON,
            },
        ],
    ])

    responses = list(self.conv.send_message_async("What's the weather?"))

    self.mock_tool.execute.assert_called_once()
    self.assertEqual(
        [r.to_json() for r in responses], [_ANSWER, {"role": "assistant"}]
    )
    self.assertEqual(responses[-1].finish_reason, "max_tokens")
    self.assertEqual(responses[-1].executor_stats, _STATS)

  def test_send_message_async_cancelled_turn_has_no_end_of_turn(self):
    self._fake_send_message_async(
        [[{"text": _ANSWER}, {"is_final": True, "error": "CANCELLED"}]]
    )

    responses = list(self.conv.send_message_async("Hi"))

    self.assertEqual([r.to_json() for r in responses], [_ANSWER])

  def test_send_message_async_failing_tool_call_round_has_no_end_of_turn(self):
    self._fake_send_message_async([[
        {"text": _MALFORMED_TOOL_CALL},
        {"is_final": True, "finish_reason": "stop"},
    ]])
    stream = self.conv.send_message_async("Hi")

    with self.assertRaisesRegex(ValueError, "Missing 'function'"):
      next(stream)

  def test_send_message_async_ignores_invalid_executor_stats(self):
    self._fake_send_message_async([[
        {"text": _ANSWER},
        {"is_final": True, "finish_reason": "stop", "executor_stats": "{"},
    ]])

    *_, end_of_turn = self.conv.send_message_async("Hi")

    self.assertEqual(end_of_turn.finish_reason, "stop")
    self.assertIsNone(end_of_turn.executor_stats)

  def test_send_message_async_final_chunk_without_finish_reason(self):
    self._fake_send_message_async([[
        {"text": _ANSWER},
        {"is_final": True, "executor_stats": _STATS_JSON},
    ]])

    responses = list(self.conv.send_message_async("Hi"))

    # Only a final chunk with a finish reason ends the turn.
    self.assertEqual([r.to_json() for r in responses], [_ANSWER])


if __name__ == "__main__":
  absltest.main()
