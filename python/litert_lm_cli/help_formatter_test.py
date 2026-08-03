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

"""Unit tests for the custom colorized help formatter."""

from absl.testing import absltest
import click
from click.testing import CliRunner

from litert_lm_cli import help_formatter

# ANSI color and style constants
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
BRIGHT_CYAN = "\x1b[96m"

BOLD_GREEN = f"{GREEN}{BOLD}"
BOLD_BRIGHT_CYAN = f"{BRIGHT_CYAN}{BOLD}"


@click.group(cls=help_formatter.ColorGroup, name="test-cli")
def cli():
  """CLI tool for testing."""


@cli.command(help="A dummy command to test help formatter.")
@click.option("--dummy-option", is_flag=True, help="A dummy option.")
def dummy():
  pass


class HelpFormatterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_group_help_formatting(self):
    result = self.runner.invoke(cli, ["--help"], color=True)
    self.assertEqual(result.exit_code, 0)

    output = result.output

    # Check for unindented description
    self.assertIn(
        f"CLI tool for testing.\n\n{BOLD_GREEN}Usage: {RESET}", output
    )

    self.assertIn(
        f"{BOLD_BRIGHT_CYAN}test-cli{RESET} {CYAN}[OPTIONS] COMMAND"
        f" [ARGS]...{RESET}",
        output,
    )

    # Check for bold green headings
    self.assertIn(f"{BOLD_GREEN}Commands{RESET}:", output)
    self.assertIn(f"{BOLD_GREEN}Global options{RESET}:", output)

    # Check for bold bright cyan terms
    self.assertIn(f"{BOLD_BRIGHT_CYAN}dummy{RESET}", output)
    self.assertIn(f"{BOLD_BRIGHT_CYAN}--help{RESET}", output)

  def test_command_help_formatting(self):
    result = self.runner.invoke(cli, ["dummy", "--help"], color=True)
    self.assertEqual(result.exit_code, 0)

    output = result.output

    # Check for unindented description
    self.assertIn(
        f"A dummy command to test help formatter.\n\n{BOLD_GREEN}Usage:"
        f" {RESET}",
        output,
    )

    self.assertIn(
        f"{BOLD_BRIGHT_CYAN}test-cli dummy{RESET} {CYAN}[OPTIONS]{RESET}",
        output,
    )

    # Check for bold green headings
    self.assertIn(f"{BOLD_GREEN}Options{RESET}:", output)

    # Check for bold bright cyan terms
    self.assertIn(f"{BOLD_BRIGHT_CYAN}--dummy-option{RESET}", output)
    self.assertIn(f"{BOLD_BRIGHT_CYAN}--help{RESET}", output)


if __name__ == "__main__":
  absltest.main()
