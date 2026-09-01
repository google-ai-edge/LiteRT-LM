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
from litert_lm_cli import cli_helpers
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
    *,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    is_android: bool = False,
    backend: str | None = None,
    speculative_decoding: bool | None = None,
    enable_speculative_decoding: bool | None = None,
    max_num_tokens: int | None = None,
    cache: str | None = None,
    cpu_thread_count: int | None = None,
    activation_data_type: litert_lm.ActivationDataType | None = None,
    ringbuffers_local_attention: bool | None = None,
    gpu_decode_steps_per_sync: int | None = None,
    enable_ynnpack: bool | None = None,
    runs: int = 1,
    skip_warmup: bool = False,
) -> None:
  """Benchmarks the model."""
  if speculative_decoding is None:
    speculative_decoding = enable_speculative_decoding

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
    speculative_decoding = model.resolve_config_option(
        speculative_decoding, model_obj, "speculative_decoding"
    )
    cache = model.resolve_config_option(cache, model_obj, "cache")
    activation_data_type_opt = model.resolve_config_option(
        activation_data_type, model_obj, "activation_data_type"
    )
    if isinstance(activation_data_type_opt, str):
      activation_data_type_val = litert_lm.ActivationDataType.from_str(
          activation_data_type_opt
      )
    else:
      activation_data_type_val = activation_data_type_opt
    enable_ynnpack = model.resolve_config_option(
        enable_ynnpack, model_obj, "enable_ynnpack"
    )

    backend_val = model.parse_backend(
        backend,
        model_obj=model_obj,
        cpu_thread_count=cpu_thread_count,
        gpu_decode_steps_per_sync=gpu_decode_steps_per_sync,
    )
    assert backend_val is not None
    cache_dir_val = common.cache_dir_value_from_cache_mode(cache)

    # For CLI benchmarking, we default to enabling ringbuffers when they are
    # available, for best results.
    if ringbuffers_local_attention is None:
      ringbuffers_local_attention = True

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
          enable_speculative_decoding=speculative_decoding,
          max_num_tokens=max_num_tokens,
          activation_data_type=activation_data_type_val,
          use_ringbuffers_local_attention=ringbuffers_local_attention,
          enable_ynnpack=enable_ynnpack,
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
    if speculative_decoding is not None:
      spec_dec_str = "true" if speculative_decoding else "false"
    click.echo(f"Cache                      : {cache or 'disk'}")
    click.echo(f"Speculative decoding       : {spec_dec_str}")
    if is_android:
      click.echo("Target                     : Android")

    info_list = []

    if not skip_warmup:
      click.echo("Running warmup..")
      benchmark_obj.run()

    for i in range(runs):
      click.echo(f"Running iteration {i + 1} of {runs}..")
      result = benchmark_obj.run()
      info_list.append(result)

    if not info_list:
      raise RuntimeError("No benchmark info collected")

    avg_info = litert_lm.interfaces.BenchmarkInfo(
        init_time_in_second=info_list[0].init_time_in_second,
        time_to_first_token_in_second=sum(
            i.time_to_first_token_in_second for i in info_list
        ) / len(info_list),
        last_prefill_token_count=info_list[-1].last_prefill_token_count,
        last_prefill_tokens_per_second=sum(
            i.last_prefill_tokens_per_second for i in info_list
        ) / len(info_list),
        last_decode_token_count=info_list[-1].last_decode_token_count,
        last_decode_tokens_per_second=sum(
            i.last_decode_tokens_per_second for i in info_list
        ) / len(info_list),
    )

    click.echo("----- Results -----")
    click.echo(
        f"Prefill speed:        {avg_info.last_prefill_tokens_per_second:.2f}"
        " tokens/s"
    )
    click.echo(
        f"Decode speed:         {avg_info.last_decode_tokens_per_second:.2f}"
        " tokens/s"
    )
    click.echo(f"Init time:            {avg_info.init_time_in_second:.4f} s")
    click.echo(
        f"Time to first token:  {avg_info.time_to_first_token_in_second:.4f} s"
    )

  except Exception:  # pylint: disable=broad-exception-caught
    click.echo(click.style("An error occurred during benchmarking", fg="red"))
    traceback.print_exc()


@click.command(
    cls=help_formatter.ColorCommand,
    help="""Benchmarks a model.
  \b
  Examples:
    # Benchmark using a model ID from 'litert-lm list'
    litert-lm benchmark my-model

    # Benchmark using a local path
    litert-lm benchmark ./model.litertlm

    # Benchmark directly from a HuggingFace repository
    litert-lm benchmark --from-huggingface-repo org/repo model.litertlm""",
)
@click.argument("model_reference", required=False)
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
@click.option(
    "--runs",
    type=click.IntRange(min=1),
    default=1,
    help="The number of benchmarking iterations to run and average.",
)
@click.option(
    "--skip-warmup",
    is_flag=True,
    help="Skip the warmup run before benchmarking.",
)
@common.common_inference_options
def benchmark(
    model_reference: str | None = None,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    backend: str | None = None,
    android: bool = False,
    speculative_decoding: bool | None = None,
    enable_speculative_decoding: bool | None = None,
    verbose: bool = False,
    from_huggingface_repo: str | None = None,
    huggingface_token: str | None = None,
    max_num_tokens: int | None = None,
    cache: str | None = None,
    cpu_thread_count: int | None = None,
    activation_data_type: str | None = None,
    ringbuffers_local_attention: bool | None = None,
    gpu_decode_steps_per_sync: int | None = None,
    enable_ynnpack: bool | None = None,
    runs: int = 1,
    skip_warmup: bool = False,
) -> None:
  """Benchmarks a LiteRT-LM model.

  Args:
    model_reference: A relative or absolute path to a .litertlm model file, or a
      model ID from `litert-lm list`. If from-huggingface-repo is set, this is
      the filename in the repository.
    prefill_tokens: The number of tokens to prefill.
    decode_tokens: The number of tokens to decode.
    backend: The backend to use (cpu, gpu or npu).
    android: Run on Android via ADB.
    speculative_decoding: Speculative decoding mode (True, False, or None for
      auto).
    enable_speculative_decoding: Speculative decoding mode (True, False, or None
      for auto).
    verbose: Whether to enable verbose logging.
    from_huggingface_repo: The HuggingFace repository ID.
    huggingface_token: The HuggingFace API token.
    max_num_tokens: Maximum number of tokens for the KV cache.
    cache: The cache mode to use (no, memory, or disk).
    cpu_thread_count: The number of threads to use for CPU backend.
    activation_data_type: The activation data type to use for inference.
    ringbuffers_local_attention: Whether to use ringbuffers for local attention
      KV cache to minimize memory usage.
    gpu_decode_steps_per_sync: The number of decode steps per sync for GPU
      backend. Only applied to supported GPU models. Otherwise, ignored.
    enable_ynnpack: Whether to delegate supported CPU operations to YNNPACK
      before XNNPACK.
    runs: The number of benchmarking iterations to run and average.
    skip_warmup: Skip the warmup run before benchmarking.
  """
  if speculative_decoding is None:
    speculative_decoding = enable_speculative_decoding

  if verbose:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.VERBOSE)

  model_reference = model_reference or cli_helpers.resolve_model_file(
      from_huggingface_repo,
      huggingface_token,
  )

  if from_huggingface_repo:
    model_path = huggingface_download.download_from_huggingface(
        repo_id=from_huggingface_repo,
        filename=model_reference,
        token=huggingface_token,
    )
    model_obj = model.Model.from_model_path(model_path)
  else:
    model_obj = model.Model.from_model_reference(model_reference)

  max_num_tokens = model.resolve_config_option(
      max_num_tokens, model_obj, "max_num_tokens"
  )
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
      enable_speculative_decoding=speculative_decoding,
      max_num_tokens=max_num_tokens,
      cache=cache,
      cpu_thread_count=cpu_thread_count,
      activation_data_type=(
          litert_lm.ActivationDataType.from_str(activation_data_type)
          if activation_data_type
          else None
      ),
      ringbuffers_local_attention=ringbuffers_local_attention,
      gpu_decode_steps_per_sync=gpu_decode_steps_per_sync,
      enable_ynnpack=enable_ynnpack,
      runs=runs,
      skip_warmup=skip_warmup,
  )


def register(cli: click.Group) -> None:
  """Registers the benchmark command."""
  cli.add_command(benchmark)
