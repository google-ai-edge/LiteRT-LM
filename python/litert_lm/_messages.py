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
"""Message and Content helper classes for LiteRT-LM."""

from __future__ import annotations

import abc
import base64
import collections.abc
import dataclasses
import enum
from typing import Any, Mapping, Sequence


class Role(enum.Enum):
  """The role of the message in a conversation."""

  SYSTEM = "system"
  USER = "user"
  MODEL = "model"
  TOOL = "tool"


@dataclasses.dataclass
class ToolCall:
  """Tool call returned by the model."""

  name: str
  arguments: dict[str, Any]

  def to_json(self) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "arguments": self.arguments,
        },
    }


class Content(abc.ABC):
  """Represents a content in the Message of the conversation."""

  @abc.abstractmethod
  def to_json(self) -> dict[str, Any]:
    raise NotImplementedError


@dataclasses.dataclass
class Text(Content):
  """Text content."""

  text: str

  def to_json(self) -> dict[str, Any]:
    return {"type": "text", "text": self.text}

  def __str__(self) -> str:
    return self.text


@dataclasses.dataclass
class ImageBytes(Content):
  """Image provided as raw bytes."""

  bytes: bytes

  def to_json(self) -> dict[str, Any]:
    return {
        "type": "image",
        "blob": base64.b64encode(self.bytes).decode("utf-8"),
    }


@dataclasses.dataclass
class ImageFile(Content):
  """Image provided by a file."""

  absolute_path: str

  def to_json(self) -> dict[str, Any]:
    return {"type": "image", "path": self.absolute_path}


@dataclasses.dataclass
class AudioBytes(Content):
  """Audio provided as raw bytes."""

  bytes: bytes

  def to_json(self) -> dict[str, Any]:
    return {
        "type": "audio",
        "blob": base64.b64encode(self.bytes).decode("utf-8"),
    }


@dataclasses.dataclass
class AudioFile(Content):
  """Audio provided by a file."""

  absolute_path: str

  def to_json(self) -> dict[str, Any]:
    return {"type": "audio", "path": self.absolute_path}


@dataclasses.dataclass
class ToolResponse(Content):
  """Tool response provided by the user."""

  name: str
  response: Any

  def to_json(self) -> dict[str, Any]:
    return {
        "type": "tool_response",
        "name": self.name,
        "response": self.response,
    }


# Attach subclasses to Content
Content.Text = Text
Content.ImageBytes = ImageBytes
Content.ImageFile = ImageFile
Content.AudioBytes = AudioBytes
Content.AudioFile = AudioFile
Content.ToolResponse = ToolResponse


class Contents:
  """Represents a list of Content in a Message."""

  def __init__(self, contents: Sequence[Content]):
    self.contents = list(contents)

  def to_json(self) -> list[dict[str, Any]]:
    return [c.to_json() for c in self.contents]

  def __str__(self) -> str:
    return "".join(str(c) for c in self.contents if isinstance(c, Text))

  @classmethod
  def empty(cls) -> Contents:
    """Creates an empty Contents list."""
    return cls([])

  @classmethod
  def of(cls, *args: str | Content | Sequence[Content]) -> Contents:
    """Creates a Contents from text, Content, or a list of Content."""
    if not args:
      return cls([])
    if len(args) == 1:
      (arg,) = args
      if isinstance(arg, str):
        return cls([Content.Text(arg)])
      if isinstance(arg, Content):
        return cls([arg])
      if isinstance(arg, collections.abc.Sequence) and not isinstance(
          arg, (str, bytes)
      ):
        return cls(arg)

    contents = []
    for arg in args:
      if isinstance(arg, str):
        contents.append(Content.Text(arg))
      elif isinstance(arg, Content):
        contents.append(arg)
      else:
        raise TypeError(f"Unsupported type in Contents.of: {type(arg)}")
    return cls(contents)


class Message:
  """Represents a message in the conversation."""

  def __init__(
      self,
      role: Role,
      contents: Contents | None = None,
      tool_calls: Sequence[ToolCall] = (),
      channels: Mapping[str, str] | None = None,
  ):
    self.role = role
    self.contents = contents or Contents.empty()
    self.tool_calls = list(tool_calls)
    self.channels = dict(channels) if channels is not None else {}

  def to_json(self) -> dict[str, Any]:
    res = {"role": self.role.value}
    if self.contents.contents:
      res["content"] = self.contents.to_json()
    if self.tool_calls:
      res["tool_calls"] = [tc.to_json() for tc in self.tool_calls]
    if self.channels:
      res["channels"] = self.channels
    return res

  def __str__(self) -> str:
    return str(self.contents)

  @classmethod
  def system(cls, text_or_contents: str | Contents) -> Message:
    """Creates a system Message."""
    contents = (
        text_or_contents
        if isinstance(text_or_contents, Contents)
        else Contents.of(text_or_contents)
    )
    return cls(Role.SYSTEM, contents)

  @classmethod
  def user(cls, text_or_contents: str | Contents) -> Message:
    """Creates a user Message."""
    contents = (
        text_or_contents
        if isinstance(text_or_contents, Contents)
        else Contents.of(text_or_contents)
    )
    return cls(Role.USER, contents)

  @classmethod
  def model(
      cls,
      contents: Contents | None = None,
      tool_calls: Sequence[ToolCall] = (),
      channels: Mapping[str, str] | None = None,
  ) -> Message:
    """Creates a model Message."""
    return cls(Role.MODEL, contents, tool_calls, channels)

  @classmethod
  def tool(cls, contents: Contents) -> Message:
    """Creates a tool Message."""
    return cls(Role.TOOL, contents)


def normalize_message(
    message: str | Contents | Message | collections.abc.Mapping[str, Any],
) -> collections.abc.Mapping[str, Any]:
  """Normalizes various input types into a standard message dictionary.

  Args:
      message: The input message to normalize. Supported types are: `str` (for
        most simple text input, automatically wrapped as a user message),
        `Contents` (for multi-modality interleaving, automatically wrapped as a
        user message), `Message` (full message object, useful when automatic
        tool calling is disabled and a tool response is required, or for
        system/model preface messages), or `collections.abc.Mapping` (super
        flexible raw dictionary format).

  Returns:
      A standardized message dictionary with 'role' and 'content' keys.
  """
  if isinstance(message, str):
    return {"role": "user", "content": message}
  if isinstance(message, Contents):
    return {"role": "user", "content": message.to_json()}
  if isinstance(message, Message):
    return message.to_json()
  if isinstance(message, collections.abc.Mapping):
    return message
  raise TypeError(f"Unsupported message type: {type(message)}")
