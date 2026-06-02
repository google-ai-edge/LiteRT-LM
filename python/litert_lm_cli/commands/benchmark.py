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

"""Benchmark subcommand for LiteRT-LM CLI."""

import traceback

import click

import litert_lm
from litert_lm_cli import common
from litert_lm_cli import help_formatter
from litert_lm_cli import huggingface_download
from litert_lm_cli import model

try:
  # pylint: disable=g-import-not-at-top
  from litert_lm.adb import adb_benchmark  # pytype: disable=import-error

  _HAS_ADB = True
except ImportError:
  _HAS_ADB = False


def run_benchmark(
    model_obj: model.Model,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    is_android: bool = False,
    backend: str = "cpu",
    enable_speculative_decoding: bool | None = None,
    max_num_tokens: int | None = None,
    cache: str = "disk",
):
  """Benchmarks the model."""
  if not model_obj.exists():
    click.echo(
        click.style(
            f"Could not find {model_obj.to_str()} locally in"
            f" {model_obj.model_path}.",
            fg="red",
        )
    )
    return

  try:
    backend_val = model.parse_backend(backend)
    cache_dir_val = common.cache_dir_value_from_cache_mode(cache)

    if is_android:
      if not _HAS_ADB:
        raise ImportError("litert_lm.adb dependencies are not available.")
      benchmark_obj = adb_benchmark.AdbBenchmark(
          model_obj.model_path,
          backend=backend_val,
          prefill_tokens=prefill_tokens,
          decode_tokens=decode_tokens,
          max_num_tokens=max_num_tokens,
          cache_dir=cache_dir_val,
      )
    else:
      benchmark_obj = litert_lm.Benchmark(
          model_obj.model_path,
          backend=backend_val,
          prefill_tokens=prefill_tokens,
          decode_tokens=decode_tokens,
          cache_dir=cache_dir_val,
          enable_speculative_decoding=enable_speculative_decoding,
          max_num_tokens=max_num_tokens,
      )

    click.echo(
        f"Benchmarking model: {model_obj.to_str()} ({model_obj.model_path})"
    )
    resolved_backend_str = backend_val.get_name()
    click.echo(f"Backend                    : {resolved_backend_str}")
    click.echo(f"Number of tokens in prefill: {prefill_tokens}")
    click.echo(f"Number of tokens in decode : {decode_tokens}")
    if max_num_tokens is not None:
      click.echo(f"Max number of tokens       : {max_num_tokens}")

    spec_dec_str = "auto"
    if enable_speculative_decoding is True:
      spec_dec_str = "true"
    elif enable_speculative_decoding is False:
      spec_dec_str = "false"
    click.echo(f"Cache                      : {cache}")
    click.echo(f"Speculative decoding       : {spec_dec_str}")
    if is_android:
      click.echo("Target                     : Android")

    result = benchmark_obj.run()

    click.echo("----- Results -----")
    click.echo(
        f"Prefill speed:        {result.last_prefill_tokens_per_second:.2f}"
        " tokens/s"
    )
    click.echo(
        f"Decode speed:         {result.last_decode_tokens_per_second:.2f}"
        " tokens/s"
    )
    click.echo(f"Init time:            {result.init_time_in_second:.4f} s")
    click.echo(
        f"Time to first token:  {result.time_to_first_token_in_second:.4f} s"
    )

  except Exception:  # pylint: disable=broad-exception-caught
    click.echo(click.style("An error occurred during benchmarking", fg="red"))
    traceback.print_exc()


@click.command(
    cls=help_formatter.ColorCommand,
    help="""Benchmarks a LiteRT-LM model.
  \b
  Examples:
    # Benchmark using a model ID from 'litert-lm list'
    litert-lm benchmark my-model

    # Benchmark using a local path
    litert-lm benchmark ./model.litertlm

    # Benchmark directly from a HuggingFace repository
    litert-lm benchmark --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference")
@click.option(
    "-p",
    "--prefill-tokens",
    default=256,
    type=int,
    help="The number of tokens to prefill.",
)
@click.option(
    "-d",
    "--decode-tokens",
    default=256,
    type=int,
    help="The number of tokens to decode.",
)
@click.option(
    "--max-num-tokens",
    type=int,
    default=None,
    help=(
        "Maximum number of tokens for the KV cache. If not set, it will be"
        " chosen based on --prefill_tokens and --decode_tokens."
    ),
)
@common.common_inference_options
def benchmark(
    model_reference: str,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    backend: str = "cpu",
    android: bool = False,
    enable_speculative_decoding: bool | None = None,
    verbose: bool = False,
    from_huggingface_repo: str | None = None,
    huggingface_token: str | None = None,
    max_num_tokens: int | None = None,
    cache: str = "disk",
):
  """Benchmarks a LiteRT-LM model.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prefill_tokens: The number of tokens to prefill.
    decode_tokens: The number of tokens to decode.
    backend: The backend to use (cpu, gpu or npu).
    android: Run on Android via ADB.
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
    max_num_tokens: Maximum number of tokens for the KV cache.
    cache: The cache mode to use (no, memory, or disk).
  """
  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  if from_huggingface_repo:
    model_path = huggingface_download.download_from_huggingface(
        repo_id=from_huggingface_repo,
        filename=model_reference,
        token=huggingface_token,
    )
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)

  if max_num_tokens is None:
    # Replicates the logic from
    # runtime/engine/engine_settings.cc
    max_num_tokens = ((prefill_tokens + 1023) // 4096 + 1) * 4096

  run_benchmark(
      model_obj,
      prefill_tokens=prefill_tokens,
      decode_tokens=decode_tokens,
      is_android=android,
      backend=backend,
      enable_speculative_decoding=enable_speculative_decoding,
      max_num_tokens=max_num_tokens,
      cache=cache,
  )


def register(cli: click.Group) -> None:
  """Registers the benchmark command."""
  cli.add_command(benchmark)
