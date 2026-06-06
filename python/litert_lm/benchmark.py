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
"""Benchmark wrapper for LiteRT-LM."""

from importlib import resources
import json
import os
import subprocess
import sys

from . import interfaces


class Benchmark(interfaces.AbstractBenchmark):
  """Benchmark wrapper that delegates to the C++ benchmark binary."""

  def _run_binary(self, binary_path: str) -> interfaces.BenchmarkInfo:
    # Construct arguments
    args = [
        binary_path,
        f"--model_path={self.model_path}",
        f"--backend={self.backend.get_name()}",
        f"--prefill_tokens={self.prefill_tokens}",
        f"--decode_tokens={self.decode_tokens}",
    ]
    if self.max_num_tokens is not None:
      args.append(f"--max_num_tokens={self.max_num_tokens}")
    if self.cache_dir:
      args.append(f"--cache_dir={self.cache_dir}")

    spec_dec_str = "auto"
    if self.enable_speculative_decoding is True:
      spec_dec_str = "true"
    elif self.enable_speculative_decoding is False:
      spec_dec_str = "false"
    args.append(f"--speculative_decoding={spec_dec_str}")

    # Run the binary
    try:
      res = subprocess.run(
          args,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          check=True,
      )
    except subprocess.CalledProcessError as e:
      raise RuntimeError(f"Benchmark binary failed: {e.stderr}") from e

    # Parse JSON output
    try:
      output = json.loads(res.stdout)
    except json.JSONDecodeError as e:
      raise RuntimeError(
          f"Failed to parse benchmark output: {res.stdout}"
      ) from e

    return interfaces.BenchmarkInfo(
        init_time_in_second=output["init_time_in_second"],
        time_to_first_token_in_second=output["time_to_first_token_in_second"],
        last_prefill_token_count=output["last_prefill_token_count"],
        last_prefill_tokens_per_second=output["last_prefill_tokens_per_second"],
        last_decode_token_count=output["last_decode_token_count"],
        last_decode_tokens_per_second=output["last_decode_tokens_per_second"],
        peak_mem_mb=output["peak_mem_mb"],
        peak_private_mb=output["peak_private_mb"],
    )

  def run(self) -> interfaces.BenchmarkInfo:
    # Find the binary
    binary_name = "benchmark_cc"
    if sys.platform == "win32":
      binary_name = "benchmark_cc.exe"

    # 1. Try to use Bazel runfiles environment variables directly.
    # This is robust for local runs (local=True) and Forge runs.
    for env_var in ["TEST_SRCDIR", "RUNFILES_DIR"]:
      if runfiles_dir := os.environ.get(env_var):
        candidate = os.path.join(
            runfiles_dir,
            "google3/python/litert_lm",
            binary_name,
        )
        if os.path.exists(candidate):
          return self._run_binary(candidate)

    # 2. Fallback to importlib.resources (for packaged runs / OSS)
    ref = resources.files(__package__) / binary_name
    with resources.as_file(ref) as binary_path:
      if not binary_path.exists():
        raise FileNotFoundError(
            f"Could not find benchmark C++ binary at {binary_path}"
        )
      return self._run_binary(str(binary_path))
