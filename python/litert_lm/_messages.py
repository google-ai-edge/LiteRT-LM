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
import json
import math
from typing import Any, Mapping, Sequence


class Role(enum.Enum):
  """The role of the message in a conversation."""

  SYSTEM = "system"
  USER = "user"
  MODEL = "assistant"
  TOOL = "tool"


@dataclasses.dataclass
class ToolCall:
  """Tool call returned by the model."""

  name: str
  arguments: dict[str, Any]
  id: str | None = None
  type: str = "function"

  def to_json(self) -> dict[str, Any]:
    res: dict[str, Any] = {
        "type": self.type,
        "function": {
            "name": self.name,
            "arguments": self.arguments,
        },
    }
    if self.id is not None:
      res["id"] = self.id
    return res


class Content(abc.ABC):
  """Represents a content in the Message of the conversation."""

  # pylint: disable=invalid-name
  # These sub-classes are added here to improve the type checking (pyrefly).
  # The names are chosen before adding the attributes and also consistent with
  # other langugage bindings.
  Text: type[Text]
  ImageBytes: type[ImageBytes]
  ImageFile: type[ImageFile]
  AudioBytes: type[AudioBytes]
  AudioFile: type[AudioFile]
  ToolResponse: type[ToolResponse]
  VideoFrames: type[VideoFrames]
  # pylint: enable=invalid-name

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


@dataclasses.dataclass
class VideoFrame:
  """One frame of a `VideoFrames`, already decoded by the caller.

  Exactly one of `image_bytes`, `image_path` or `pixels` must be set. LiteRT-LM
  does not decode video containers; decode and sample frames in the application
  and pass them here. The runtime handles resizing and patchification.

  Attributes:
    image_bytes: The frame as encoded image bytes (e.g. JPEG or PNG).
    image_path: Absolute path of the frame as an encoded image file.
    pixels: The frame as a raw `[height, width, 3]` uint8 RGB array (e.g. a
      numpy array), sent without re-encoding.
    timestamp: Presentation time of the frame in the source video, in seconds.
      If unset, the runtime uses `index / VideoFrames.fps`.
  """

  image_bytes: bytes | None = None
  image_path: str | None = None
  pixels: Any = None
  timestamp: float | None = None

  def __post_init__(self):
    num_set = sum(
        value is not None
        for value in (self.image_bytes, self.image_path, self.pixels)
    )
    if num_set != 1:
      raise ValueError(
          "Exactly one of image_bytes, image_path or pixels must be set."
      )

  def to_json(self) -> dict[str, Any]:
    """Returns the frame as a `"frames"` entry of a video content item."""
    res: dict[str, Any]
    if self.image_bytes is not None:
      res = {"blob": base64.b64encode(self.image_bytes).decode("utf-8")}
    elif self.image_path is not None:
      res = {"path": self.image_path}
    else:
      res = _raw_rgb_frame_to_json(self.pixels)
    if self.timestamp is not None:
      res["timestamp"] = float(self.timestamp)
    return res


def _raw_rgb_frame_to_json(pixels: Any) -> dict[str, Any]:
  """Serializes a `[height, width, 3]` uint8 array as base64 RGB888 pixels."""
  shape = tuple(getattr(pixels, "shape", ()))
  if len(shape) != 3 or shape[2] != 3:
    raise ValueError(
        f"Video frame pixels must have shape [height, width, 3], got {shape}."
    )
  dtype = getattr(pixels, "dtype", None)
  if dtype is not None and str(dtype) != "uint8":
    raise ValueError(f"Video frame pixels must be uint8, got {dtype}.")
  # C-contiguous HWC bytes (`tobytes()` copies non-contiguous arrays).
  data = pixels.tobytes() if hasattr(pixels, "tobytes") else bytes(pixels)
  return {
      "pixels": base64.b64encode(data).decode("utf-8"),
      "width": int(shape[1]),
      "height": int(shape[0]),
  }


@dataclasses.dataclass
class VideoFrames(Content):
  """Video provided as a sequence of frames sampled by the caller.

  Requires a model whose metadata reports the Video input modality. The
  runtime groups the frames as the model expects (for example, pairs of
  consecutive frames for vision encoders with `temporal_patch_size = 2`,
  repeating the last frame if the count is odd).

  Attributes:
    frames: The sampled frames in presentation order, all of the same size.
    fps: Sampling rate used for frames without a `timestamp` (the runtime
      defaults to 2 frames per second).
  """

  frames: Sequence[VideoFrame]
  fps: float | None = None

  def to_json(self) -> dict[str, Any]:
    if not self.frames:
      raise ValueError("A video must have at least one frame.")
    res: dict[str, Any] = {
        "type": "video",
        "frames": [frame.to_json() for frame in self.frames],
    }
    if self.fps is not None:
      res["fps"] = float(self.fps)
    return res


class _RawRgbFrame:
  """Raw RGB888 frame bytes parsed back from a video content item."""

  def __init__(self, data: bytes, width: int, height: int):
    self.shape = (height, width, 3)
    self.dtype = "uint8"
    self._data = data

  def tobytes(self) -> bytes:
    return self._data


def _video_frames_from_json(item: Mapping[str, Any]) -> VideoFrames:
  """Parses a `{"type": "video"}` content item into a `VideoFrames`."""
  frames = []
  for frame in item.get("frames", []):
    timestamp = frame.get("timestamp")
    if "path" in frame:
      frames.append(VideoFrame(image_path=frame["path"], timestamp=timestamp))
    elif "blob" in frame:
      frames.append(
          VideoFrame(
              image_bytes=base64.b64decode(frame["blob"]), timestamp=timestamp
          )
      )
    elif "pixels" in frame:
      frames.append(
          VideoFrame(
              pixels=_RawRgbFrame(
                  base64.b64decode(frame["pixels"]),
                  int(frame["width"]),
                  int(frame["height"]),
              ),
              timestamp=timestamp,
          )
      )
  return VideoFrames(frames=frames, fps=item.get("fps"))


# Attach subclasses to Content
Content.Text = Text
Content.ImageBytes = ImageBytes
Content.ImageFile = ImageFile
Content.AudioBytes = AudioBytes
Content.AudioFile = AudioFile
Content.ToolResponse = ToolResponse
Content.VideoFrames = VideoFrames


class Contents(list[Content]):
  """Represents a list of Content in a Message."""

  def __init__(self, contents: Sequence[Content] = ()):
    super().__init__(contents)

  @property
  def contents(self) -> list[Content]:
    """Returns self for backward compatibility.

    Note:
      `Contents` now inherits from `list[Content]` directly. Callers should
      iterate or index `message.contents` directly instead of
      `message.contents.contents`.
    """
    return self

  def to_json(self) -> list[dict[str, Any]]:
    return [c.to_json() for c in self]

  def __str__(self) -> str:
    return "".join(str(c) for c in self if isinstance(c, Text))

  def __repr__(self) -> str:
    return f"Contents({super().__repr__()})"

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


def _is_number(value: Any) -> bool:
  return (
      isinstance(value, (int, float))
      and not isinstance(value, bool)
      and math.isfinite(value)
  )


def _is_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


@dataclasses.dataclass(frozen=True)
class ExecutorStats:
  """Executor stats of a turn, e.g. speculative decoding counters.

  Metric and latency names depend on the executor and may change.

  Attributes:
      module_name: The module the stats are for, e.g. "LLM".
      metrics: Counters and ratios by name, e.g. "acceptance_rate".
      latencies_in_microseconds: Latencies in microseconds by name.
      substats: The stats of the module's submodules, e.g. the drafter.
  """

  module_name: str = ""
  metrics: collections.abc.Mapping[str, int | float] = dataclasses.field(
      default_factory=dict
  )
  latencies_in_microseconds: collections.abc.Mapping[str, int] = (
      dataclasses.field(default_factory=dict)
  )
  substats: tuple[ExecutorStats, ...] = ()

  @classmethod
  def from_json(cls, data: collections.abc.Mapping[str, Any]) -> ExecutorStats:
    """Parses the `executor_stats` JSON object, ignoring malformed fields."""
    metrics = data.get("metrics")
    latencies_us = data.get("latencies_us")
    substats = data.get("substats")
    module_name = data.get("module_name")
    return cls(
        module_name=module_name if isinstance(module_name, str) else "",
        metrics={
            str(key): value
            for key, value in (
                metrics.items()
                if isinstance(metrics, collections.abc.Mapping)
                else ()
            )
            if _is_number(value)
        },
        latencies_in_microseconds={
            str(key): value
            for key, value in (
                latencies_us.items()
                if isinstance(latencies_us, collections.abc.Mapping)
                else ()
            )
            if _is_int(value)
        },
        substats=tuple(
            cls.from_json(substat)
            for substat in (substats if isinstance(substats, list) else ())
            if isinstance(substat, collections.abc.Mapping)
        ),
    )


# The keys of the message that ends a turn. They describe the turn rather than
# the message, so they are parsed into properties and never sent back.
_END_OF_TURN_KEYS = ("finish_reason", "executor_stats")


# NOTE: Inherits from `dict[str, Any]` for backward compatibility with existing
# callers expecting a `Mapping`. Please migrate to `Message` attributes (e.g.,
# `role`, `contents`, `tool_calls`, `channels`), as `dict` support may be
# dropped in the future.
class Message(dict[str, Any]):
  """Represents a message in the conversation.

  Note:
    Inherits from `dict[str, Any]` for backward compatibility. Callers should
    migrate to using `Message` properties directly, as dictionary access support
    may be removed in a future release.
  """

  role: Role
  contents: Contents
  tool_calls: list[ToolCall]
  channels: dict[str, str]
  _has_role: bool
  _raw_content: Any
  _extra_fields: dict[str, Any]
  _finish_reason: str | None
  _executor_stats: ExecutorStats | None

  def __init__(
      self,
      role: Role,
      contents: Contents | None = None,
      tool_calls: Sequence[ToolCall] = (),
      channels: Mapping[str, str] | None = None,
  ):
    super().__init__()
    super().__setattr__("_has_role", True)
    super().__setattr__("_raw_content", None)
    super().__setattr__("_extra_fields", {})
    super().__setattr__("_finish_reason", None)
    super().__setattr__("_executor_stats", None)
    super().__setattr__("role", role)
    super().__setattr__(
        "contents", contents if contents is not None else Contents.empty()
    )
    super().__setattr__("tool_calls", list(tool_calls))
    super().__setattr__(
        "channels", dict(channels) if channels is not None else {}
    )
    self._sync_dict()

  def __setattr__(self, name: str, value: Any) -> None:
    if name == "contents":
      super().__setattr__("_raw_content", None)
    elif name == "role":
      super().__setattr__("_has_role", True)
    super().__setattr__(name, value)
    if name in (
        "role",
        "contents",
        "tool_calls",
        "channels",
        "_has_role",
        "_raw_content",
        "_extra_fields",
    ):
      self._sync_dict()

  def __setitem__(self, key: str, value: Any) -> None:
    super().__setitem__(key, value)
    self._sync_domain_from_dict()

  def __delitem__(self, key: str) -> None:
    super().__delitem__(key)
    self._sync_domain_from_dict()

  def update(self, *args: Any, **kwargs: Any) -> None:
    super().update(*args, **kwargs)
    self._sync_domain_from_dict()

  def pop(self, key: str, *args: Any) -> Any:
    val = super().pop(key, *args)
    self._sync_domain_from_dict()
    return val

  def clear(self) -> None:
    super().clear()
    self._sync_domain_from_dict()

  def setdefault(self, key: str, default: Any = None) -> Any:
    val = super().setdefault(key, default)
    self._sync_domain_from_dict()
    return val

  def _sync_domain_from_dict(self) -> None:
    """Synchronizes domain attributes from the underlying dictionary state."""
    (
        role,
        contents,
        tool_calls,
        channels,
        has_role,
        raw_content,
        extra_fields,
    ) = self._parse_json_fields(dict(self))
    super().__setattr__("role", role)
    super().__setattr__("contents", contents)
    super().__setattr__("tool_calls", tool_calls)
    super().__setattr__("channels", channels or {})
    super().__setattr__("_has_role", has_role)
    super().__setattr__("_raw_content", raw_content)
    super().__setattr__("_extra_fields", extra_fields)

  def _build_json_dict(self) -> dict[str, Any]:
    """Builds a JSON-serializable dictionary from domain attributes."""
    res: dict[str, Any] = {}
    if self._has_role:
      res["role"] = (
          self.role.value if isinstance(self.role, Role) else self.role
      )
    if self._raw_content is not None:
      res["content"] = self._raw_content
    elif self.contents:
      res["content"] = self.contents.to_json()
    if self.tool_calls:
      res["tool_calls"] = [tc.to_json() for tc in self.tool_calls]
    if self.channels:
      res["channels"] = self.channels
    if self._extra_fields:
      res.update(self._extra_fields)
    return res

  def _sync_dict(self) -> None:
    """Synchronizes the underlying dict storage with domain attributes."""
    new_dict = self._build_json_dict()
    for k in list(super().keys()):
      if k not in new_dict:
        super().__delitem__(k)
    super().update(new_dict)

  def to_json(self) -> dict[str, Any]:
    self._sync_dict()
    return dict(super().items())

  def __str__(self) -> str:
    return str(self.contents)

  def __repr__(self) -> str:
    return (
        f"Message(role={self.role!r}, contents={self.contents!r},"
        f" tool_calls={self.tool_calls!r}, channels={self.channels!r})"
    )

  @property
  def finish_reason(self) -> str | None:
    """Why the turn ended, set only on the message that ends a turn.

    That message is the one `send_message` returns, or the last element of the
    `send_message_async` stream, which has no content. A set `finish_reason` is
    how callers detect the end of a turn. "stop": the model finished its
    response, or the response reached `max_output_tokens`. "max_tokens": the
    conversation filled the engine's context window (`max_num_tokens`), which
    cut the response off. "appended": the message was appended without a
    response. Other values may be added; they also mean the turn ended.

    None on every other message, including the ones the caller builds.
    """
    return self._finish_reason

  @property
  def executor_stats(self) -> ExecutorStats | None:
    """The executor stats of the turn this message ends, or None.

    Only set on a message with a `finish_reason`, and only when the executor
    reports stats.
    """
    return self._executor_stats

  @classmethod
  def _parse_json_fields(cls, data: collections.abc.Mapping[str, Any]) -> tuple[
      Role,
      Contents,
      list[ToolCall],
      dict[str, str] | None,
      bool,
      Any,
      dict[str, Any],
  ]:
    """Parses JSON dictionary fields into domain components."""
    has_role = "role" in data
    role_str = data.get("role", "assistant")
    try:
      role = Role(role_str)
    except ValueError:
      role = Role.MODEL if role_str == "model" else Role.USER

    content_data = data.get("content")
    contents_list: list[Content] = []
    if isinstance(content_data, str):
      contents_list.append(Content.Text(content_data))
    elif isinstance(content_data, collections.abc.Sequence) and not isinstance(
        content_data, (str, bytes)
    ):
      for item in content_data:
        if isinstance(item, str):
          contents_list.append(Content.Text(item))
        elif isinstance(item, collections.abc.Mapping):
          item_type = item.get("type")
          if item_type == "text":
            contents_list.append(Content.Text(item.get("text", "")))
          elif item_type == "image":
            if "path" in item:
              contents_list.append(Content.ImageFile(item["path"]))
            elif "blob" in item:
              contents_list.append(
                  Content.ImageBytes(base64.b64decode(item["blob"]))
              )
          elif item_type == "audio":
            if "path" in item:
              contents_list.append(Content.AudioFile(item["path"]))
            elif "blob" in item:
              contents_list.append(
                  Content.AudioBytes(base64.b64decode(item["blob"]))
              )
          elif item_type == "tool_response":
            contents_list.append(
                Content.ToolResponse(
                    name=item.get("name", ""), response=item.get("response")
                )
            )
          elif item_type == "video":
            contents_list.append(_video_frames_from_json(item))

    parsed_contents = Contents(contents_list)
    raw_content = None
    if "content" in data:
      if parsed_contents.to_json() != content_data or not contents_list:
        raw_content = content_data

    tool_calls_data = data.get("tool_calls", [])
    tool_calls: list[ToolCall] = []
    if isinstance(tool_calls_data, collections.abc.Sequence) and not isinstance(
        tool_calls_data, (str, bytes)
    ):
      for tc in tool_calls_data:
        if isinstance(tc, collections.abc.Mapping):
          fn = tc.get("function", tc)
          if isinstance(fn, collections.abc.Mapping):
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
              try:
                raw_args = json.loads(raw_args)
              except json.JSONDecodeError:
                raw_args = {}
            arguments = (
                dict(raw_args)
                if isinstance(raw_args, collections.abc.Mapping)
                else {}
            )
            tool_calls.append(
                ToolCall(
                    name=fn.get("name", ""),
                    arguments=arguments,
                    id=tc.get("id"),
                    type=tc.get("type", "function"),
                )
            )

    channels_data = data.get("channels")
    channels = (
        dict(channels_data)
        if isinstance(channels_data, collections.abc.Mapping)
        else None
    )

    extra_fields = {
        k: v
        for k, v in data.items()
        if k not in ("role", "content", "tool_calls", "channels")
        and k not in _END_OF_TURN_KEYS
    }

    return (
        role,
        parsed_contents,
        tool_calls,
        channels,
        has_role,
        raw_content,
        extra_fields,
    )

  @classmethod
  def from_json(cls, data: collections.abc.Mapping[str, Any]) -> Message:
    """Creates a Message from a JSON dictionary."""
    (
        role,
        parsed_contents,
        tool_calls,
        channels,
        has_role,
        raw_content,
        extra_fields,
    ) = cls._parse_json_fields(data)
    msg = cls(
        role=role,
        contents=parsed_contents,
        tool_calls=tool_calls,
        channels=channels,
    )
    msg._has_role = has_role
    msg._raw_content = raw_content
    msg._extra_fields = extra_fields
    finish_reason = data.get("finish_reason")
    executor_stats = data.get("executor_stats")
    msg._finish_reason = (
        finish_reason if isinstance(finish_reason, str) else None
    )
    # Executor stats are only kept on a message that ends a turn.
    msg._executor_stats = None
    if msg._finish_reason is not None and isinstance(
        executor_stats, collections.abc.Mapping
    ):
      msg._executor_stats = ExecutorStats.from_json(executor_stats)
    msg._sync_dict()
    return msg

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
