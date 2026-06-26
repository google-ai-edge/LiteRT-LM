# Copyright 2026 Google LLC.
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

"""Pytest fixtures and hooks for litert_lm engine tests.

This file contains pytest configurations, fixtures, and hooks used by the
tests in this directory. It handles command-line argument parsing,
dynamic test parametrization, and provides helper functions for running
the litert_lm C++ engine.
"""

import os
import subprocess
from typing import Callable

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
  """Adds custom command line arguments to pytest."""

  parser.addoption(
      "--model-path",
      action="store",
      required=True,
      help="Absolute or relative path to the .litertlm model file",
  )

  parser.addoption(
      "--build-system",
      action="store",
      choices=["bazel", "cmake"],
      required=True,
      help="The build system to use for the engine binary (bazel or cmake).",
  )


@pytest.fixture(scope="session")
def model_path(request: pytest.FixtureRequest) -> str:
  """Extracts the model path from the CLI and ensures the file exists."""
  path = request.config.getoption("--model-path")
  if not os.path.exists(path):
    pytest.fail(f"CRITICAL: Model file not found at {path}")
  return path


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
  """Generates test parameters for the 'engine_binary' fixture.

  This function is a pytest hook that dynamically parametrizes tests
  that depend on the 'engine_binary' fixture, based on the build system
  specified via command-line options.

  Args:
    metafunc: The metafunc object provided by pytest.
  """
  if "engine_binary" not in metafunc.fixturenames:
    return

  build_system = metafunc.config.getoption("--build-system")
  repo_root = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "..", "..")
  )
  exe_ext = ".exe" if os.name == "nt" else ""

  if build_system == "bazel":
    binary_path = os.path.join(
        repo_root,
        "bazel-bin",
        "runtime",
        "engine",
        f"litert_lm_main{exe_ext}",
    )
  else:
    binary_path = os.path.join(
        repo_root, "cmake", "build", f"litert_lm_main{exe_ext}"
    )

  if not os.path.exists(binary_path):
    pytest.fail(
        f"CRITICAL: Engine binary not found at {binary_path}. "
        f"Did you run the {build_system} build?"
    )

  metafunc.parametrize("engine_binary", [binary_path])


@pytest.fixture
def run_engine(
    engine_binary: str, model_path: str
) -> Callable[..., str]:
  """Provides a helper fixture that tests can call to execute the C++ engine.

  It automatically handles timeouts and fatal crashes.

  Args:
    engine_binary: Path to the engine executable.
    model_path: Path to the model file.
  """

  def _run(backend: str, prompt: str, timeout: int = 120) -> str:
    cmd = [
        engine_binary,
        f"--backend={backend}",
        f"--model_path={model_path}",
        f"--input_prompt={prompt}",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )

    # Instantly fail the test if the C++ engine segfaults or OOMs
    if result.returncode != 0:
      error_tail = (result.stderr if result.stderr else result.stdout).strip()[
          -800:
      ]
      pytest.fail(
          f"Engine crashed with code {result.returncode}!\nCrash"
          f" Tail:\n{error_tail}"
      )

    return result.stdout + result.stderr

  return _run
