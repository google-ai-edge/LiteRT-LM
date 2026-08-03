#!/bin/bash
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

# This script builds the litert-lm-api Python package into a PyPI-ready
# wheel using Bazelisk, and verifies the built wheel in an isolated environment.

set -ex

# Get the workspace root
WORKSPACE_ROOT=$(bazelisk info workspace)
cd "${WORKSPACE_ROOT}"

# Get the bazel-bin path
BAZEL_BIN=$(bazelisk info bazel-bin)
WHEEL_DIR="${BAZEL_BIN}/python/litert_lm"
rm -rf "${WHEEL_DIR}"

echo "Building wheel using Bazelisk..."
bazelisk build //python/litert_lm:wheel "$@"

TEST_VENV="${WORKSPACE_ROOT}/python/litert_lm/test_venv"

# Run this Verification Suite isolated inside every target Python version
for PY_VER in "3.10" "3.11" "3.12" "3.13" "3.14"; do
  echo "------------------------------------------------"
  echo "Setting up temporary virtual environment for Python ${PY_VER}..."
  echo "------------------------------------------------"
  rm -rf "${TEST_VENV}"

  # Force uv to use or download the specific target Python version
  uv venv --python="${PY_VER}" "${TEST_VENV}"

  # Universal Cross-Platform venv Activation & Python binary selection (Windows uses Scripts/, POSIX uses bin/)
  if [[ -d "${TEST_VENV}/Scripts" ]]; then
    source "${TEST_VENV}/Scripts/activate"
    PY_EXE="python"
  else
    source "${TEST_VENV}/bin/activate"
    PY_EXE="python3"
  fi

  echo "Installing the freshly built wheel..."
  uv pip install --index-url https://pypi.org/simple "${WHEEL_DIR}"/*.whl

  # CRITICAL: We move your Current Working Directory into the isolated venv folder before running!
  # This entirely prevents the classic Python sys.path gotcha where Python accidentally imports your
  # uncompiled ./python/litert_lm Source Code instead of the actual built wheel we just installed.
  cd "${TEST_VENV}"

  # Execute our standalone checked-in verification script directly
  ${PY_EXE} "${WORKSPACE_ROOT}/python/litert_lm/verify_api.py"

  # Return to workspace root for next step
  cd "${WORKSPACE_ROOT}"
  deactivate
done

# Clean up temporary scratch folders
rm -rf "${TEST_VENV}"
echo "✨ Verification completed successfully for all Python versions!"
