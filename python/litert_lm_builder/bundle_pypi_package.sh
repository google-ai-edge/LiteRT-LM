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

# This script builds the litert-lm-builder Python package into a PyPI-ready
# wheel using Bazelisk, and verifies the built wheel in an isolated environment.

set -ex

# Get the workspace root
WORKSPACE_ROOT=$(bazelisk info workspace)
cd "${WORKSPACE_ROOT}"

# Get the bazel-bin path
BAZEL_BIN=$(bazelisk info bazel-bin)
WHEEL_DIR="${BAZEL_BIN}/python/litert_lm_builder"
rm -rf "${WHEEL_DIR}"

echo "Building wheel using Bazelisk..."
bazelisk build //python/litert_lm_builder:wheel "$@"

# --- Testing / Verification Steps ---
TEST_VENV="${WORKSPACE_ROOT}/python/litert_lm_builder/test_venv"

for PY_VER in "3.10" "3.11" "3.12" "3.13" "3.14"; do
  echo "------------------------------------------------"
  echo "Setting up temporary virtual environment for Python ${PY_VER}..."
  echo "------------------------------------------------"
  rm -rf "${TEST_VENV}"
  # Force uv to use or download the specific Python version
  uv venv --python="${PY_VER}" "${TEST_VENV}"
  # Activate venv and install the wheel
  source "${TEST_VENV}/bin/activate"
  echo "Installing the built wheel..."
  uv pip install --index-url https://pypi.org/simple "${WHEEL_DIR}"/*.whl
  echo "Verifying CLI tools..."
  litert-lm-builder --help
  litert-lm-peek --help
  echo "Verifying Python import..."
  python3 -c "import litert_lm_builder; print(litert_lm_builder.LitertLmFileBuilder)"
  # Clean up for next iteration
  deactivate
done

rm -rf "${TEST_VENV}"
echo "Verification completed successfully for all Python versions!"
