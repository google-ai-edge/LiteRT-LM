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

from absl.testing import absltest
from absl.testing import parameterized

import litert_lm


class MessagesTest(parameterized.TestCase):

  def test_role_values(self):
    self.assertEqual(litert_lm.Role.SYSTEM.value, "system")
    self.assertEqual(litert_lm.Role.USER.value, "user")
    self.assertEqual(litert_lm.Role.MODEL.value, "model")
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
    self.assertEmpty(contents.contents)
    self.assertEqual(contents.to_json(), [])
    self.assertEqual(str(contents), "")

  def test_contents_of_string(self):
    contents = litert_lm.Contents.of("hello")
    self.assertLen(contents.contents, 1)
    content = contents.contents[0]
    self.assertIsInstance(content, litert_lm.Content.Text)
    assert isinstance(content, litert_lm.Content.Text)
    self.assertEqual(content.text, "hello")
    self.assertEqual(contents.to_json(), [{"type": "text", "text": "hello"}])
    self.assertEqual(str(contents), "hello")

  def test_contents_of_content(self):
    text_content = litert_lm.Content.Text(text="hello")
    contents = litert_lm.Contents.of(text_content)
    self.assertLen(contents.contents, 1)
    self.assertEqual(contents.contents[0], text_content)

  def test_contents_of_sequence(self):
    c1 = litert_lm.Content.Text(text="hello")
    c2 = litert_lm.Content.Text(text=" world")
    contents = litert_lm.Contents.of([c1, c2])
    self.assertLen(contents.contents, 2)
    self.assertEqual(contents.contents[0], c1)
    self.assertEqual(contents.contents[1], c2)
    self.assertEqual(str(contents), "hello world")

  def test_contents_of_varargs(self):
    c1 = litert_lm.Content.Text(text="hello")
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
            "role": "model",
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
            "role": "model",
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
            "role": "model",
            "content": [{"type": "text", "text": "thinking"}],
            "channels": {"reasoning": "thinking"},
        },
    )


if __name__ == "__main__":
  absltest.main()
