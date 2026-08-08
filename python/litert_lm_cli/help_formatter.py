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

"""Custom help formatter for LiteRT-LM CLI."""

import textwrap

import click


class ColorHelpFormatter(click.HelpFormatter):
  """Custom help formatter for colorized help output."""

  def write_heading(self, heading: str) -> None:
    super().write_heading(click.style(heading, fg="green", bold=True))

  def write_usage(
      self, prog: str, args: str = "", prefix: str | None = None
  ) -> None:
    super().write_usage(
        click.style(prog, fg="bright_cyan", bold=True),
        click.style(args, fg="cyan"),
        prefix=click.style(prefix or "Usage: ", fg="green", bold=True),
    )

  def write_dl(self, rows, col_max=30, col_spacing=2) -> None:
    """Writes a definition list with colorized terms."""
    super().write_dl(
        [(click.style(t, fg="bright_cyan", bold=True), h) for t, h in rows],
        col_max,
        col_spacing,
    )


class ColorContext(click.Context):
  """Custom context that uses ColorHelpFormatter."""

  formatter_class = ColorHelpFormatter


class ColorCommand(click.Command):
  """Custom command that uses ColorContext."""

  context_class = ColorContext

  def format_help(
      self, ctx: click.Context, formatter: click.HelpFormatter
  ) -> None:
    if self.help:
      formatter.write(f"{textwrap.dedent(self.help).strip()}\n")
    formatter.write("\n")
    super().format_usage(ctx, formatter)
    self.format_options(ctx, formatter)

  def _format_params(
      self,
      ctx: click.Context,
      formatter: click.HelpFormatter,
      section_name: str,
  ) -> None:
    if params := [
        r for p in self.get_params(ctx) if (r := p.get_help_record(ctx))
    ]:
      with formatter.section(section_name):
        formatter.write_dl(params)

  def format_options(
      self, ctx: click.Context, formatter: click.HelpFormatter
  ) -> None:
    self._format_params(ctx, formatter, "Options")


COMMAND_SECTIONS: dict[str, list[str]] = {
    "Inference Commands": ["run", "benchmark", "serve"],
    "Model Management Commands": ["list", "import", "rename", "delete"],
    "LiteRT-LM File Commands": ["pack", "unpack", "convert"],
}


class ColorGroup(click.Group, ColorCommand):
  """Custom group that uses ColorContext and defaults to ColorCommand."""

  def format_commands(
      self, ctx: click.Context, formatter: click.HelpFormatter
  ) -> None:
    subcommands = self.list_commands(ctx)
    cmd_map = {}
    for subcommand in subcommands:
      cmd = self.get_command(ctx, subcommand)
      if cmd is None or cmd.hidden:
        continue
      help_str = cmd.get_short_help_str(limit=formatter.width)
      cmd_map[subcommand] = help_str

    if not cmd_map:
      return

    max_len = max(len(cmd) for cmd in cmd_map.keys())

    seen = set()
    for section_name, cmd_list in COMMAND_SECTIONS.items():
      rows = []
      for cmd_name in cmd_list:
        if cmd_name in cmd_map:
          rows.append((cmd_name.ljust(max_len), cmd_map[cmd_name]))
          seen.add(cmd_name)
      if rows:
        with formatter.section(section_name):
          formatter.write_dl(rows)

    other_rows = [
        (cmd_name.ljust(max_len), help_str)
        for cmd_name, help_str in cmd_map.items()
        if cmd_name not in seen
    ]
    if other_rows:
      with formatter.section("Other Commands"):
        formatter.write_dl(other_rows)

  def format_options(
      self, ctx: click.Context, formatter: click.HelpFormatter
  ) -> None:
    self.format_commands(ctx, formatter)
    self._format_params(ctx, formatter, "Global options")

  def command(self, *args, **kwargs):
    kwargs.setdefault("cls", ColorCommand)
    return super().command(*args, **kwargs)
