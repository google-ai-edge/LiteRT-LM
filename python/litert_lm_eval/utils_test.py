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

"""Tests for utils module."""

from absl.testing import absltest
from litert_lm_eval import utils


class UtilsTest(absltest.TestCase):

  def test_parse_unknown_args_empty(self):
    args = []
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {})

  def test_parse_unknown_args_boolean_flag(self):
    args = ["--write_out"]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {"write_out": True})

  def test_parse_unknown_args_boolean_flag_with_values(self):
    args = ["--predict_only", "true", "--use_cache", "False"]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {"predict_only": True, "use_cache": False})

  def test_parse_unknown_args_int_value(self):
    args = ["--batch_size", "16", "--limit", "100"]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {"batch_size": 16, "limit": 100})

  def test_parse_unknown_args_float_value(self):
    args = ["--temperature", "0.5"]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {"temperature": 0.5})

  def test_parse_unknown_args_string_value(self):
    args = ["--device", "cuda"]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {"device": "cuda"})

  def test_parse_unknown_args_mixed(self):
    args = [
        "--write_out",
        "--device",
        "cpu",
        "--limit",
        "10.5",
        "--use_cache",
        "false",
        "--batch_size",
        "32",
    ]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(
        kwargs,
        {
            "write_out": True,
            "device": "cpu",
            "limit": 10.5,
            "use_cache": False,
            "batch_size": 32,
        },
    )

  def test_parse_unknown_args_ignore_non_flags(self):
    args = ["ignore_me", "--valid_flag", "value", "ignore_me_too"]
    kwargs = utils.parse_unknown_args(args)
    self.assertEqual(kwargs, {"valid_flag": "value"})


if __name__ == "__main__":
  absltest.main()
