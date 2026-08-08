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

"""Main script for litert-lm binary."""

import importlib
import sys

import click

import litert_lm
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import version
from litert_lm_cli.commands import benchmark as _benchmark_module
from litert_lm_cli.commands import convert as _convert_module
from litert_lm_cli.commands import delete as _delete_module
from litert_lm_cli.commands import list as _list_module
from litert_lm_cli.commands import pack as _pack_module
from litert_lm_cli.commands import rename as _rename_module
from litert_lm_cli.commands import run as _run_module
from litert_lm_cli.commands import serve as _serve_module
from litert_lm_cli.commands import unpack as _unpack_module

# Import 'import' subcommand dynamically to bypass Python keyword restriction.
_import_module = importlib.import_module(
    "litert_lm_cli.commands.import"
)


@click.group(
    cls=help_formatter.ColorGroup,
    name="litert-lm",
    context_settings=dict(
        show_default=True,
        max_content_width=120,
        help_option_names=["-h", "--help"],
    ),
)
@common.config_option
@click.version_option(version=version.VERSION)
def cli():
  """CLI tool for LiteRT-LM models."""


_serve_module.register(cli)
_convert_module.register(cli)
_list_module.register(cli)
_import_module.register(cli)
_delete_module.register(cli)
_rename_module.register(cli)
_benchmark_module.register(cli)
_run_module.register(cli)
_pack_module.register(cli)
_unpack_module.register(cli)


def main(argv=None) -> None:
  """Entry point for console_scripts and binary execution."""
  del argv  # Unused.
  litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
  cli()


if __name__ == "__main__":
  main()
