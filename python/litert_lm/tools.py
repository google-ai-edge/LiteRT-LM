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

"""Tool interfaces for LiteRT LM."""

from __future__ import annotations

import collections.abc
import inspect
import re
import typing
from typing import Any

from .interfaces import Tool


def _parse_param_descriptions(docstring: str) -> dict[str, str]:
  """Parses Args section of docstring."""
  descriptions = {}
  if not docstring:
    return descriptions
  lines = docstring.split("\n")
  in_args = False
  current_arg = None
  for line in lines:
    stripped = line.strip()
    if stripped == "Args:":
      in_args = True
      current_arg = None
      continue
    if in_args and stripped in ("Returns:", "Raises:", "Yields:"):
      in_args = False
      break

    if not in_args:
      continue

    # Expect indentation for arguments
    match = re.match(r"\s+([\w.]+)(?:\s*\(.*?\))?:\s*(.*)", line)
    if match:
      current_arg = match.group(1)
      descriptions[current_arg] = match.group(2).strip()
    elif current_arg and line.startswith(" " * 4):
      descriptions[current_arg] += " " + stripped
    elif not stripped:
      current_arg = None
  return descriptions


def _py_type_to_openapi(py_type: Any) -> dict[str, Any]:
  """Converts a Python type to an OpenAPI schema fragment."""
  if py_type is int:
    return {"type": "integer"}
  if py_type is float:
    return {"type": "number"}
  if py_type is bool:
    return {"type": "boolean"}
  if py_type is str:
    return {"type": "string"}

  origin = typing.get_origin(py_type)
  if origin in (list, collections.abc.Sequence, collections.abc.Iterable):
    args = typing.get_args(py_type)
    if args:
      return {"type": "array", "items": _py_type_to_openapi(args[0])}
    return {"type": "array"}

  # Fallback to string
  return {"type": "string"}


class _FunctionTool(Tool):
  """A Tool implementation that wraps a Python function."""

  def __init__(self, func: collections.abc.Callable[..., Any]):
    self._func = func

  def get_tool_description(self) -> dict[str, Any]:
    """Returns the OpenAPI schema for the function."""
    sig = inspect.signature(self._func)
    doc = inspect.getdoc(self._func) or ""
    param_descriptions = _parse_param_descriptions(doc)

    parameters = {
        "type": "object",
        "properties": {
            name: {
                **_py_type_to_openapi(param.annotation),
                **(
                    {"description": param_descriptions[name]}
                    if name in param_descriptions
                    else {}
                ),
            }
            for name, param in sig.parameters.items()
        },
        "required": [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ],
    }

    return {
        "type": "function",
        "function": {
            "name": self._func.__name__,
            "description": doc.split("\n")[0] if doc else "",
            "parameters": parameters,
        },
    }

  def execute(self, param: collections.abc.Mapping[str, Any]) -> Any:
    return self._func(**param)


def tool_from_function(func: collections.abc.Callable[..., Any]) -> Tool:
  """Converts a Python function into a Tool."""
  return _FunctionTool(func)
