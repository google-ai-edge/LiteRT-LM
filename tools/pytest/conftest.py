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
import sys
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

  parser.addoption(
      "--backend",
      action="store",
      default="cpu",
      choices=["cpu", "gpu"],
      help="The backend to use for the engine binary (cpu or gpu).",
  )

  parser.addoption(
      "--dynamic",
      action="store_true",
      default=False,
      help=(
          "If set, configures the environment to support dynamically linked"
          " libraries."
      ),
  )

  parser.addoption(
      "--prebuilt",
      action="store_true",
      default=False,
      help=(
          "If set, uses the prebuilt directory for dynamic libraries."
      ),
  )


@pytest.fixture(scope="session", name="model_path")
def fixture_model_path(request: pytest.FixtureRequest) -> str:
  """Extracts the model path from the CLI and ensures the file exists."""
  path = request.config.getoption("--model-path")
  if not os.path.exists(path):
    pytest.fail(f"CRITICAL: Model file not found at {path}")
  return path


@pytest.fixture(scope="session", name="backend")
def fixture_backend(request: pytest.FixtureRequest) -> str:
  """Extracts the backend configuration from the CLI."""
  return request.config.getoption("--backend")


@pytest.fixture(scope="session", name="prebuilt_dir")
def fixture_prebuilt_dir(request: pytest.FixtureRequest) -> str:
  """Extracts the prebuilt directory from the CLI."""
  return request.config.getoption("--prebuilt")


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


def _get_dynamic_lib_dirs(
    engine_binary: str, request: pytest.FixtureRequest
) -> list[str]:
  """Returns a list of directories containing dynamic libraries."""
  prebuilt_dir = request.config.getoption("--prebuilt")

  repo_root = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "..", "..")
  )
  lib_dirs = set()

  if prebuilt_dir:
    lib_dirs.add(os.path.dirname(os.path.join(repo_root, "prebuilt")))

  else:
    if sys.platform == "win32":
      ext = ".dll"
    elif sys.platform == "darwin":
      ext = ".dylib"
    else:
      ext = ".so"

    path_parts = os.path.normpath(engine_binary).split(os.sep)
    if "bazel-bin" in path_parts:
      bazel_bin_idx = path_parts.index("bazel-bin")
      bazel_bin_root = os.sep.join(path_parts[:bazel_bin_idx + 1])

      if os.path.exists(bazel_bin_root):
        for item in os.listdir(bazel_bin_root):
          if item.startswith("_solib_"):
            lib_dirs.add(os.path.join(bazel_bin_root, item))

    runfiles_dir = engine_binary + ".runfiles"
    if os.path.exists(runfiles_dir):
      for root, _, files in os.walk(runfiles_dir):
        if any(f.endswith(ext) for f in files):
          lib_dirs.add(root)

  return list(lib_dirs)


@pytest.fixture
def run_engine(
    engine_binary: str,
    model_path: str,
    backend: str,
    request: pytest.FixtureRequest,
) -> Callable[..., str]:
  """Provides a helper fixture that tests can call to execute the C++ engine.

  It automatically handles timeouts and fatal crashes.

  Args:
    engine_binary: Path to the engine executable.
    model_path: Path to the model file.
    backend: The backend to use for the engine binary (cpu or gpu).
    request: The pytest request object.
  """

  is_dynamic = request.config.getoption("--dynamic")

  def _run(prompt: str, timeout: int = 120) -> str:

    env = os.environ.copy()

    if is_dynamic:
      discovered_dirs = _get_dynamic_lib_dirs(engine_binary, request)
      print(f"\n[DEBUG] Found dynamic libraries in: {discovered_dirs}\n")
      lib_paths = os.pathsep.join(discovered_dirs)

      if sys.platform == "win32":
        env["PATH"] = lib_paths + os.pathsep + env.get("PATH", "")
      elif sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = (
            lib_paths + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
        )
      else:
        env["LD_LIBRARY_PATH"] = (
            lib_paths + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        )

    cmd = [
        engine_binary,
        f"--backend={backend}",
        f"--model_path={model_path}",
        f"--input_prompt={prompt}",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
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
