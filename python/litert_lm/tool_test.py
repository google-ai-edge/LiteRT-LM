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

from collections.abc import Sequence
import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized

import litert_lm


# Helper functions for parameterized tests
def multiply(a: int, b: int) -> int:
  """Multiplies two integers.

  Args:
      a: The first integer.
      b: The second integer.
  """
  return a * b


def greet(name: str, greeting: str = "Hello") -> str:
  """Greets a person."""
  return f"{greeting}, {name}!"


def get_weather(location: str):
  """Gets weather for a location.

  Args:
      location (str): The name of the city.
  """
  return f"Weather in {location} is sunny."


class ToolTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="basic_function",
          func=multiply,
          expected_desc={
              "type": "function",
              "function": {
                  "name": "multiply",
                  "description": "Multiplies two integers.",
                  "parameters": {
                      "type": "object",
                      "properties": {
                          "a": {
                              "type": "integer",
                              "description": "The first integer.",
                          },
                          "b": {
                              "type": "integer",
                              "description": "The second integer.",
                          },
                      },
                      "required": ["a", "b"],
                  },
              },
          },
          execute_cases=[({"a": 2, "b": 3}, 6)],
      ),
      dict(
          testcase_name="function_with_defaults",
          func=greet,
          expected_name="greet",
          expected_required=["name"],
          execute_cases=[
              ({"name": "Alice"}, "Hello, Alice!"),
              ({"name": "Bob", "greeting": "Hi"}, "Hi, Bob!"),
          ],
      ),
      dict(
          testcase_name="type_hints_in_docstring",
          func=get_weather,
          expected_properties={
              "location": {
                  "type": "string",
                  "description": "The name of the city.",
              }
          },
          execute_cases=[
              ({"location": "London"}, "Weather in London is sunny.")
          ],
      ),
  )
  def test_tool_from_function_cases(
      self,
      func,
      execute_cases,
      expected_desc=None,
      expected_name=None,
      expected_required=None,
      expected_properties=None,
  ):
    tool = litert_lm.tool_from_function(func)
    self.assertIsInstance(tool, litert_lm.Tool)
    desc = tool.get_tool_description()

    if expected_desc is not None:
      self.assertEqual(desc, expected_desc)
    if expected_name is not None:
      self.assertEqual(desc["function"]["name"], expected_name)
    if expected_required is not None:
      self.assertEqual(
          desc["function"]["parameters"]["required"], expected_required
      )
    if expected_properties is not None:
      self.assertEqual(
          desc["function"]["parameters"]["properties"], expected_properties
      )

    for args, expected_result in execute_cases:
      result = tool.execute(args)
      self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      ("list", list),
      ("sequence", Sequence),
  )
  def test_tool_from_function_array(self, type_hint):

    def product(numbers: type_hint[float]) -> float:
      """Calculates the product of a list of numbers.

      Args:
          numbers: List of numbers to multiply.
      """
      res = 1.0
      for n in numbers:
        res *= n
      return res

    tool = litert_lm.tool_from_function(product)
    desc = tool.get_tool_description()

    self.assertEqual(desc["function"]["name"], "product")
    expected_properties = {
        "numbers": {
            "type": "array",
            "items": {"type": "number"},
            "description": "List of numbers to multiply.",
        }
    }
    self.assertEqual(
        desc["function"]["parameters"]["properties"], expected_properties
    )
    self.assertEqual(tool.execute({"numbers": [2.0, 3.5]}), 7.0)

  def test_create_conversation_with_open_api_tools(self):
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    model_path = str(
        pathlib.Path(test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

    class TestTool(litert_lm.Tool):

      def __init__(self, definition):
        self._definition = definition

      def get_tool_description(self):
        return self._definition

      def execute(self, param):
        return "sunny"

    engine = litert_lm.Engine(model_path)
    open_api_tools = [
        TestTool({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        })
    ]
    with engine.create_conversation(
        tools=open_api_tools, automatic_tool_calling=False
    ) as conv:
      self.assertLen(conv.tools, 1)
      self.assertEqual(conv.tools[0], open_api_tools[0])
      self.assertFalse(conv.automatic_tool_calling)

  def test_tool_execution_exception(self):
    def fail(a: int):
      """This tool always fails."""
      raise ValueError("Planned failure")

    tool = litert_lm.tool_from_function(fail)
    # The C++ layer catches exceptions and returns them as strings.
    # We can't easily test the C++ try-catch here without a full engine run,
    # but we can verify the tool itself behaves as expected.
    with self.assertRaises(ValueError):
      tool.execute({"a": 1})

  def test_tool_execution_non_json_serializable(self):
    class Unserializable:
      pass

    def get_complex():
      """Returns an unserializable object."""
      return Unserializable()

    tool = litert_lm.tool_from_function(get_complex)
    # Again, the C++ nb::cast<nlohmann::json> would fail and be caught,
    # but we can verify the Python part returns the object.
    self.assertIsInstance(tool.execute({}), Unserializable)

  def test_create_conversation_with_mixed_tools(self):
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    model_path = str(
        pathlib.Path(test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

    class TestTool(litert_lm.Tool):

      def __init__(self, definition):
        self._definition = definition

      def get_tool_description(self):
        return self._definition

      def execute(self, param):
        return "result"

    def my_func(x: int) -> int:
      """A function tool."""
      return x

    engine = litert_lm.Engine(model_path)
    tools = [
        my_func,
        TestTool({
            "type": "function",
            "function": {
                "name": "custom_tool",
                "description": "desc",
            },
        }),
    ]
    with engine.create_conversation(tools=tools) as conv:
      self.assertLen(conv.tools, 2)
      self.assertEqual(conv.tools[0], my_func)

  def test_tool_with_no_arguments(self):
    def get_time() -> str:
      """Returns the current time."""
      return "2026-04-14 20:00:00"

    tool = litert_lm.tool_from_function(get_time)
    desc = tool.get_tool_description()
    self.assertEqual(desc["function"]["name"], "get_time")
    self.assertEmpty(desc["function"]["parameters"]["properties"])
    # Verify it can be executed with empty args
    self.assertEqual(tool.execute({}), "2026-04-14 20:00:00")

  def test_create_conversation_with_malformed_tool_description(self):
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    model_path = str(
        pathlib.Path(test_srcdir)
        / "litert_lm/runtime/testdata/test_lm.litertlm"
    )

    class MalformedTool(litert_lm.Tool):

      def get_tool_description(self):
        return {"missing": "function_key"}

      def execute(self, param):
        return None

    engine = litert_lm.Engine(model_path)
    with self.assertRaisesRegex(
        ValueError, "Tool description must contain \\['function'\\]\\['name'\\]"
    ):
      engine.create_conversation(tools=[MalformedTool()])


if __name__ == "__main__":
  absltest.main()
