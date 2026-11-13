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
"""Parses decision requests in the Gemini Decisions API REST JSON format.

Request (`input` and `questions` are required; any other field, and any value
of the wrong JSON type, is rejected):

  {
    "model": string,
    "input": string | Content | [string | Content, ...],
    "questions": {
      "<question_id>": {
        "type": "noul" | "choice" | "score",
        "instructions": string | Content | [string | Content, ...],
        "criteria": ...
      }, ...
    },
    "temperature": number
  }

  Content: {"type": "text", "text": string, "annotations": [...]}

- `model` is optional and ignored: the model is set when the decision engine is
  created. `annotations` is optional and ignored.
- Several `input` or `instructions` parts are joined with newlines. Image,
  audio, video and document content in `input` is not supported yet;
  `instructions` must be text.
- `criteria` depends on `type`:
  - `noul` (optional): `{"true_description": string, "false_description":
    string}`, both optional.
  - `choice` (required): `{"<option_id>": "<description>", ...}`. The options
    are shown to the model in JSON key order, labeled A to Z, so the order of
    the keys matters. Note that JavaScript objects list integer-like keys
    (e.g. "1", "10") first and in numeric order, whatever their insertion
    order; use non-numeric option IDs to keep a chosen order from JavaScript.
  - `score` (required): `["<level 0 description>", ...]`, from the lowest
    level to the highest.
- `temperature` is the optional softmax temperature (finite, > 0) of all
  questions.
- A JSON object must not repeat a key.

The remaining validation (question and option IDs, counts) is done by the
decision engine.

Response, with the answers in question order:

  {
    "answers": {
      "<question_id>": {"type": "noul", "confidence": number, "noul": number},
      "<question_id>": {
        "type": "choice", "confidence": number, "choice": "<option_id>",
        "probabilities": {"<option_id>": number, ...}
      },
      "<question_id>": {
        "type": "score", "confidence": number, "score": number,
        "probabilities": {"0": number, ...},
        "legend": {"0": "<level 0 description>", ...}
      }, ...
    }
  }
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import math
from typing import Any

_MEDIA_CONTENT_TYPES = ("image", "audio", "video", "document")


@dataclasses.dataclass(frozen=True)
class ParsedRequest:
  """A parsed decision request.

  Attributes:
    input: The text parts of `input`.
    questions: The questions by ID, in request order, normalized to `{"type",
      "instructions": string, "criteria"}` (`criteria` is omitted for a noul
      question without criteria).
    temperature: The temperature, or None.
  """

  input: list[str]
  questions: dict[str, dict[str, Any]]
  temperature: float | None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  out: dict[str, Any] = {}
  for key, value in pairs:
    if key in out:
      raise ValueError(f'Duplicate key "{key}" in the request.')
    out[key] = value
  return out


def _check_keys(obj: Mapping[str, Any], where: str, allowed: set[str]) -> None:
  for key in obj:
    if key not in allowed:
      raise ValueError(f'Unknown field "{key}" in {where}.')


def _check_type(
    value: Any,
    expected: type[Any] | tuple[type[Any], ...],
    where: str,
    what: str,
) -> None:
  if not isinstance(value, expected) or isinstance(value, bool):
    raise TypeError(f"{where} must be {what}.")


def _parse_content(content: Any, where: str, allow_media: bool) -> str:
  """Returns the text of a content block: `{"type": "text", "text": ...}`."""
  _check_type(content, Mapping, where, "a string or a content object")
  content_type: Any = content.get("type")
  _check_type(content_type, str, f"{where}.type", "a string")
  if content_type != "text":
    if allow_media and content_type in _MEDIA_CONTENT_TYPES:
      raise NotImplementedError(
          f'{where}: "{content_type}" content is not supported yet.'
      )
    raise ValueError(f'{where}.type must be "text", got "{content_type}".')
  _check_keys(content, where, {"type", "text", "annotations"})
  if "text" not in content:
    raise ValueError(f"{where}.text is required.")
  _check_type(content["text"], str, f"{where}.text", "a string")
  return content["text"]


def _parse_text_parts(value: Any, where: str, allow_media: bool) -> list[str]:
  """Parses `string | Content | [string | Content, ...]`."""
  if isinstance(value, str):
    return [value]
  if isinstance(value, Mapping):
    return [_parse_content(value, where, allow_media)]
  _check_type(value, list, where, "a string, a content object or an array")
  if not value:
    raise ValueError(f"{where} must not be empty.")
  return [
      item
      if isinstance(item, str)
      else _parse_content(item, f"{where}[{i}]", allow_media)
      for i, item in enumerate(value)
  ]


def _parse_criteria(
    question: Mapping[str, Any], question_type: str, where: str
) -> Any:
  """Returns the normalized criteria of `question`, or None if absent."""
  where = f"{where}.criteria"
  criteria: Any = question.get("criteria")
  if question_type == "noul":
    if "criteria" not in question:
      return None
    _check_type(criteria, Mapping, where, "an object")
    _check_keys(criteria, where, {"true_description", "false_description"})
    for key, description in criteria.items():
      _check_type(description, str, f"{where}.{key}", "a string")
    return dict(criteria)
  if "criteria" not in question:
    raise ValueError(f'{where} is required for "{question_type}" questions.')
  if question_type == "choice":
    _check_type(
        criteria, Mapping, where, "an object mapping option IDs to descriptions"
    )
    for option_id, description in criteria.items():
      _check_type(description, str, f'{where}["{option_id}"]', "a string")
    return dict(criteria)
  _check_type(criteria, list, where, "an array of strings")
  for i, description in enumerate(criteria):
    _check_type(description, str, f"{where}[{i}]", "a string")
  return list(criteria)


def _parse_question(question: Any, where: str) -> dict[str, Any]:
  _check_type(question, Mapping, where, "an object")
  _check_keys(question, where, {"type", "instructions", "criteria"})
  question_type: Any = question.get("type")
  _check_type(question_type, str, f"{where}.type", "a string")
  if question_type not in ("noul", "choice", "score"):
    raise ValueError(
        f'{where}.type must be "noul", "choice" or "score", got'
        f' "{question_type}".'
    )
  if "instructions" not in question:
    raise ValueError(f"{where}.instructions is required.")
  instructions = _parse_text_parts(
      question["instructions"], f"{where}.instructions", allow_media=False
  )
  out: dict[str, Any] = {
      "type": question_type,
      "instructions": "\n".join(instructions),
  }
  criteria = _parse_criteria(question, question_type, where)
  if criteria is not None:
    out["criteria"] = criteria
  return out


def parse_request(request: Mapping[str, Any] | str) -> ParsedRequest:
  """Parses and checks a decision request.

  Args:
    request: The request, as a JSON string or a parsed JSON mapping.

  Returns:
    The parsed request.

  Raises:
    ValueError: If the request is malformed: invalid JSON, a duplicate key (for
      a JSON string), an unknown or missing field, or an empty value.
    TypeError: If a request field has the wrong JSON type.
    NotImplementedError: If `input` has image, audio, video or document
      content.
  """
  if isinstance(request, str):
    request = json.loads(request, object_pairs_hook=_reject_duplicate_keys)
  _check_type(request, Mapping, "The request", "a JSON object")
  _check_keys(
      request, "request", {"model", "input", "questions", "temperature"}
  )
  if "model" in request:
    _check_type(request["model"], str, "request.model", "a string")
  if "input" not in request:
    raise ValueError("request.input is required.")
  input_parts = _parse_text_parts(
      request["input"], "request.input", allow_media=True
  )
  if "questions" not in request:
    raise ValueError("request.questions is required.")
  questions = request["questions"]
  _check_type(
      questions,
      Mapping,
      "request.questions",
      "an object mapping question IDs to questions",
  )
  parsed_questions = {
      question_id: _parse_question(
          question, f'request.questions["{question_id}"]'
      )
      for question_id, question in questions.items()
  }
  temperature: Any = request.get("temperature")
  if temperature is not None or "temperature" in request:
    _check_type(temperature, (int, float), "request.temperature", "a number")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
      raise ValueError(
          f"request.temperature must be finite and > 0, got {temperature}."
      )
  return ParsedRequest(
      input=input_parts, questions=parsed_questions, temperature=temperature
  )
