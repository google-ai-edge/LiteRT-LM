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

import json

from absl.testing import absltest
from absl.testing import parameterized

import litert_lm


class MessagesTest(parameterized.TestCase):

  def test_role_values(self):
    self.assertEqual(litert_lm.Role.SYSTEM.value, "system")
    self.assertEqual(litert_lm.Role.USER.value, "user")
    self.assertEqual(litert_lm.Role.MODEL.value, "assistant")
    self.assertEqual(litert_lm.Role.TOOL.value, "tool")

  def test_tool_call_to_json(self):
    tool_call = litert_lm.ToolCall(
        name="get_weather", arguments={"location": "London"}
    )
    expected = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": {"location": "London"},
        },
    }
    self.assertEqual(tool_call.to_json(), expected)

  def test_text_content(self):
    content = litert_lm.Content.Text(text="hello")
    self.assertEqual(content.to_json(), {"type": "text", "text": "hello"})
    self.assertEqual(str(content), "hello")

  def test_image_bytes_content(self):
    content = litert_lm.Content.ImageBytes(bytes=b"fake_image_data")
    # base64 of b"fake_image_data" is "ZmFrZV9pbWFnZV9kYXRh"
    self.assertEqual(
        content.to_json(),
        {"type": "image", "blob": "ZmFrZV9pbWFnZV9kYXRh"},
    )

  def test_image_file_content(self):
    content = litert_lm.Content.ImageFile(absolute_path="/path/to/image.png")
    self.assertEqual(
        content.to_json(),
        {"type": "image", "path": "/path/to/image.png"},
    )

  def test_audio_bytes_content(self):
    content = litert_lm.Content.AudioBytes(bytes=b"fake_audio_data")
    # base64 of b"fake_audio_data" is "ZmFrZV9hdWRpb19kYXRh"
    self.assertEqual(
        content.to_json(),
        {"type": "audio", "blob": "ZmFrZV9hdWRpb19kYXRh"},
    )

  def test_audio_file_content(self):
    content = litert_lm.Content.AudioFile(absolute_path="/path/to/audio.mp3")
    self.assertEqual(
        content.to_json(),
        {"type": "audio", "path": "/path/to/audio.mp3"},
    )

  def test_tool_response_content(self):
    content = litert_lm.Content.ToolResponse(
        name="get_weather", response="sunny"
    )
    self.assertEqual(
        content.to_json(),
        {
            "type": "tool_response",
            "name": "get_weather",
            "response": "sunny",
        },
    )

  def test_contents_of_empty(self):
    contents = litert_lm.Contents.empty()
    self.assertIsInstance(contents, list)
    self.assertEmpty(contents)
    self.assertFalse(bool(contents))
    self.assertEqual(contents.to_json(), [])
    self.assertEqual(str(contents), "")

  def test_contents_of_string(self):
    contents = litert_lm.Contents.of("hello")
    self.assertIsInstance(contents, list)
    self.assertLen(contents, 1)
    self.assertTrue(bool(contents))
    content = contents[0]
    self.assertIsInstance(content, litert_lm.Content.Text)
    self.assertEqual(content.text, "hello")
    self.assertEqual(str(contents), "hello")

  def test_contents_of_content(self):
    c = litert_lm.Content.Text("hello")
    contents = litert_lm.Contents.of(c)
    self.assertLen(contents, 1)
    self.assertEqual(contents[0], c)

  def test_contents_of_sequence(self):
    c1 = litert_lm.Content.Text("hello")
    c2 = litert_lm.Content.Text(" world")
    contents = litert_lm.Contents.of([c1, c2])
    self.assertLen(contents, 2)
    self.assertEqual(contents, [c1, c2])
    self.assertEqual(contents[0], c1)
    self.assertEqual(contents[1], c2)
    self.assertEqual(contents[1:], [c2])
    self.assertIs(contents.contents, contents)
    self.assertEqual(str(contents), "hello world")

  def test_contents_of_varargs(self):
    c1 = litert_lm.Content.Text("hello")
    contents = litert_lm.Contents.of(c1, " world")
    self.assertLen(contents.contents, 2)
    self.assertEqual(contents.contents[0], c1)
    c2 = contents.contents[1]
    self.assertIsInstance(c2, litert_lm.Content.Text)
    assert isinstance(c2, litert_lm.Content.Text)
    self.assertEqual(c2.text, " world")
    self.assertEqual(str(contents), "hello world")

  def test_message_system(self):
    msg = litert_lm.Message.system("system instruction")
    self.assertEqual(msg.role, litert_lm.Role.SYSTEM)
    self.assertEqual(
        msg.to_json(),
        {
            "role": "system",
            "content": [{"type": "text", "text": "system instruction"}],
        },
    )

  def test_message_user(self):
    msg = litert_lm.Message.user("hello")
    self.assertEqual(msg.role, litert_lm.Role.USER)
    self.assertEqual(
        msg.to_json(),
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        },
    )

  def test_message_model(self):
    msg = litert_lm.Message.model(litert_lm.Contents.of("response"))
    self.assertEqual(msg.role, litert_lm.Role.MODEL)
    self.assertEqual(
        msg.to_json(),
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "response"}],
        },
    )

  def test_message_tool(self):
    msg = litert_lm.Message.tool(litert_lm.Contents.of("tool result"))
    self.assertEqual(msg.role, litert_lm.Role.TOOL)
    self.assertEqual(
        msg.to_json(),
        {
            "role": "tool",
            "content": [{"type": "text", "text": "tool result"}],
        },
    )

  def test_message_with_tool_calls(self):
    tc = litert_lm.ToolCall(
        name="get_weather", arguments={"location": "London"}
    )
    msg = litert_lm.Message.model(tool_calls=[tc])
    self.assertEqual(
        msg.to_json(),
        {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "London"},
                },
            }],
        },
    )

  def test_message_with_channels(self):
    msg = litert_lm.Message.model(
        contents=litert_lm.Contents.of("thinking"),
        channels={"reasoning": "thinking"},
    )
    self.assertEqual(
        msg.to_json(),
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "thinking"}],
            "channels": {"reasoning": "thinking"},
        },
    )

  def test_message_mapping_and_dict_compatibility(self):
    raw_dict = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hello world"}],
        "channels": {"thought": "reasoning"},
        "reasoning_content": "reasoning",
    }
    msg = litert_lm.Message.from_json(raw_dict)

    # Verify domain object attributes
    self.assertIsInstance(msg, litert_lm.Message)
    self.assertEqual(msg.role, litert_lm.Role.MODEL)
    self.assertEqual(str(msg), "hello world")
    self.assertEqual(msg.channels, {"thought": "reasoning"})

    # Verify dict / Mapping compatibility
    self.assertIsInstance(msg, dict)
    self.assertEqual(msg["role"], "assistant")
    self.assertEqual(msg["content"], [{"type": "text", "text": "hello world"}])
    self.assertEqual(msg.get("channels"), {"thought": "reasoning"})
    self.assertEqual(msg.get("reasoning_content"), "reasoning")
    self.assertIsNone(msg.get("nonexistent"))
    self.assertIn("content", msg)
    self.assertNotIn("tool_calls", msg)
    self.assertLen(msg, 4)
    self.assertEqual(
        set(msg.keys()), {"role", "content", "channels", "reasoning_content"}
    )
    self.assertEqual(msg, raw_dict)
    self.assertEqual(raw_dict, msg)
    self.assertEqual(msg.to_json(), raw_dict)
    self.assertEqual(json.dumps(msg), json.dumps(raw_dict))

    # Verify dict mutation persistence across reads
    msg["custom_key"] = "custom_val"
    self.assertEqual(msg["custom_key"], "custom_val")
    self.assertEqual(msg.get("custom_key"), "custom_val")
    del msg["custom_key"]
    self.assertNotIn("custom_key", msg)

    # Verify mutating standard keys updates domain attributes
    msg["role"] = "user"
    self.assertEqual(msg.role, litert_lm.Role.USER)
    msg["content"] = [{"type": "text", "text": "updated text"}]
    self.assertEqual(str(msg), "updated text")

    # Verify deleting standard keys and reassigning via attribute
    del msg["role"]
    self.assertNotIn("role", msg)
    msg.role = litert_lm.Role.SYSTEM
    self.assertEqual(msg["role"], "system")

    # Verify assigning string to msg.role works
    msg.role = "user"  # type: ignore[assignment]
    self.assertEqual(msg["role"], "user")

    # Verify dict mutator methods: update, setdefault, pop, clear
    msg.update({"role": "assistant", "extra_updated": 99})
    self.assertEqual(msg.role, litert_lm.Role.MODEL)
    self.assertEqual(msg["extra_updated"], 99)
    self.assertEqual(
        msg.setdefault("setdefault_key", "default_val"), "default_val"
    )
    self.assertEqual(msg["setdefault_key"], "default_val")
    self.assertEqual(msg.pop("extra_updated"), 99)
    self.assertNotIn("extra_updated", msg)
    msg.clear()
    self.assertEmpty(msg)
    self.assertNotIn("role", msg)

    # Verify repr and equality helpers
    self.assertIn("Message(", repr(msg))
    c1 = litert_lm.Contents.of("a")
    c2 = litert_lm.Contents.of("a")
    c3 = litert_lm.Contents.of("b")
    self.assertEqual(c1, c2)
    self.assertNotEqual(c1, c3)
    self.assertNotEqual(c1, "a")
    self.assertIn("Contents(", repr(c1))

  def test_message_from_json_tool_call_arguments_formats(self):
    raw_dict = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tool_json_str",
                    "arguments": '{"city": "Tokyo"}',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_none",
                    "arguments": None,
                },
            },
        ],
    }
    msg = litert_lm.Message.from_json(raw_dict)
    self.assertLen(msg.tool_calls, 2)
    self.assertEqual(msg.tool_calls[0].arguments, {"city": "Tokyo"})
    self.assertEqual(msg.tool_calls[1].arguments, {})

  def test_message_from_json_multimodal_and_tools(self):
    raw_dict = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Here is an image:"},
            {"type": "image", "path": "/tmp/test.png"},
            {"type": "image", "blob": "ZmFrZV9pbWFnZV9kYXRh"},
            {"type": "audio", "path": "/tmp/test.mp3"},
            {"type": "audio", "blob": "ZmFrZV9hdWRpb19kYXRh"},
            {"type": "tool_response", "name": "calc", "response": 42},
        ],
        "tool_calls": [{
            "type": "function",
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
        }],
    }
    msg = litert_lm.Message.from_json(raw_dict)
    self.assertLen(msg.contents.contents, 6)
    self.assertIsInstance(msg.contents.contents[0], litert_lm.Content.Text)
    self.assertIsInstance(msg.contents.contents[1], litert_lm.Content.ImageFile)
    self.assertIsInstance(
        msg.contents.contents[2], litert_lm.Content.ImageBytes
    )
    self.assertIsInstance(msg.contents.contents[3], litert_lm.Content.AudioFile)
    self.assertIsInstance(
        msg.contents.contents[4], litert_lm.Content.AudioBytes
    )
    self.assertIsInstance(
        msg.contents.contents[5], litert_lm.Content.ToolResponse
    )
    self.assertLen(msg.tool_calls, 1)
    self.assertEqual(msg.tool_calls[0].name, "get_weather")
    self.assertEqual(msg.tool_calls[0].arguments, {"city": "Paris"})
    self.assertEqual(msg.tool_calls[0].id, "call_1")
    self.assertEqual(msg.to_json(), raw_dict)


if __name__ == "__main__":
  absltest.main()
