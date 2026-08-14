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

# This script builds the litert-lm CLI Python package into a PyPI-ready
# wheel using Bazelisk, and verifies the built wheel in an isolated environment.

set -ex

# Get the workspace root
WORKSPACE_ROOT=$(bazelisk info workspace)
cd "${WORKSPACE_ROOT}"

# Get the bazel-bin path
BAZEL_BIN=$(bazelisk info bazel-bin)
WHEEL_DIR="${BAZEL_BIN}/python/litert_lm_cli"
rm -rf "${WHEEL_DIR}"

# Determine version early
LITERT_LM_VERSION=$(grep "^VERSION =" version.bzl | cut -d'"' -f2)
TODAY_DATE=$(TZ='America/Los_Angeles' date +%Y%m%d)

# Determine the build/release mode
IS_STABLE_RELEASE=false
if [[ "${PUBLISH_STABLE_RELEASE}" == "1" ]]; then
  IS_STABLE_RELEASE=true
fi

IS_STAGE_RELEASE=false
if [[ "${STAGE_ARTIFACTS_WITH_INTERNAL_SYMBOLS}" == "1" && "${KOKORO_JOB_TYPE:-}" == "RELEASE" ]]; then
  IS_STAGE_RELEASE=true
fi

IS_NIGHTLY_RELEASE=false
if [[ "${PUBLISH_NIGHTLY}" == "1" ]]; then
  IS_NIGHTLY_RELEASE=true
fi

# Set configurations based on mode
# Staging release should behave exactly like stable release (no fallback, stable wheels),
# but it won't be uploaded to PyPI (handled in build.sh).
if [[ "${IS_STABLE_RELEASE}" == "true" || "${IS_STAGE_RELEASE}" == "true" ]]; then
  echo "Mode: Stable Release or Staging Release (Strict)"
  API_WHEELS_GCS_DIR="gs://litert-lm-api/macos/stable_wheels"
  API_PREFIX="litert_lm_api"
  EXPECTED_VERSION="${LITERT_LM_VERSION}"
  ALLOW_FALLBACK=false
  EXTRA_BAZEL_ARGS=""
elif [[ "${IS_NIGHTLY_RELEASE}" == "true" ]]; then
  echo "Mode: Nightly Release (Strict)"
  API_WHEELS_GCS_DIR="gs://litert-lm-api/macos/nightly_wheels"
  API_PREFIX="litert_lm_api_nightly"
  EXPECTED_VERSION="${LITERT_LM_VERSION}.dev${TODAY_DATE}"
  ALLOW_FALLBACK=false
  EXTRA_BAZEL_ARGS="--define=DEV_BUILD=1 --define=DEV_VERSION=${TODAY_DATE}"
else
  echo "Mode: Presubmit or Local Developer Build (Fallback Allowed)"
  API_WHEELS_GCS_DIR="gs://litert-lm-api/macos/nightly_wheels"
  API_PREFIX="litert_lm_api_nightly"
  EXPECTED_VERSION="${LITERT_LM_VERSION}.dev${TODAY_DATE}"
  ALLOW_FALLBACK=true
  EXTRA_BAZEL_ARGS="--define=DEV_BUILD=1 --define=DEV_VERSION=${TODAY_DATE}"
fi

API_WHEELS_DIR="${WORKSPACE_ROOT}/api_wheels"
mkdir -p "${API_WHEELS_DIR}"

# Check if wheels were pre-fetched by Kokoro via gfile_resources into KOKORO_GFILE_DIR
if [[ -n "${KOKORO_GFILE_DIR:-}" ]]; then
  echo "Checking KOKORO_GFILE_DIR (${KOKORO_GFILE_DIR}) for pre-fetched API wheels..."
  MATCHED_WHEEL_LIST=$(find "${KOKORO_GFILE_DIR}" -name "${API_PREFIX}-${EXPECTED_VERSION}-*.whl" 2>/dev/null)
  
  if [[ -z "${MATCHED_WHEEL_LIST}" ]]; then
    if [[ "${ALLOW_FALLBACK}" == "false" ]]; then
      echo "❌ Expected API wheels (${EXPECTED_VERSION}) not found in KOKORO_GFILE_DIR."
      exit 1
    else
      echo "Expected API wheels (${EXPECTED_VERSION}) not found in KOKORO_GFILE_DIR. Trying fallback..."
      LATEST_FALLBACK_WHEEL=$(find "${KOKORO_GFILE_DIR}" -name "${API_PREFIX}-*.whl" 2>/dev/null | sort -V | tail -n 1 || true)
      if [[ -n "${LATEST_FALLBACK_WHEEL}" ]]; then
         ACTUAL_VERSION=$(basename "$LATEST_FALLBACK_WHEEL" | cut -d'-' -f2)
         MATCHED_WHEEL_LIST=$(find "${KOKORO_GFILE_DIR}" -name "${API_PREFIX}-${ACTUAL_VERSION}-*.whl" 2>/dev/null)
         EXPECTED_VERSION="${ACTUAL_VERSION}"
         ACTUAL_DATE=$(echo "$ACTUAL_VERSION" | grep -o 'dev[0-9]*' | sed 's/dev//' || true)
         if [[ -n "${ACTUAL_DATE}" ]]; then
           EXTRA_BAZEL_ARGS="--define=DEV_BUILD=1 --define=DEV_VERSION=${ACTUAL_DATE}"
         fi
      fi
    fi
  fi
  
  if [[ -n "${MATCHED_WHEEL_LIST}" ]]; then
    echo "Copying pre-fetched API wheels..."
    echo "${MATCHED_WHEEL_LIST}" | while read -r f; do
      if [[ -n "$f" ]]; then
        echo "  $(basename "$f")"
        cp "$f" "${API_WHEELS_DIR}/"
      fi
    done
  fi
fi

# Fetch from GCS if not found in GFILE_DIR
if ! ls "${API_WHEELS_DIR}"/*.whl > /dev/null 2>&1; then
  echo "Fetching API wheels from GCS..."
  set +e
  WHEEL_LIST=$(gcloud storage ls "${API_WHEELS_GCS_DIR}/${API_PREFIX}-${EXPECTED_VERSION}-*.whl" 2>/dev/null)
  set -e
  
  if [[ -z "${WHEEL_LIST}" ]]; then
    if [[ "${ALLOW_FALLBACK}" == "false" ]]; then
      echo "❌ Expected API wheels (${EXPECTED_VERSION}) not found in GCS."
      exit 1
    else
      echo "Expected API wheels (${EXPECTED_VERSION}) not found in GCS. Trying fallback..."
      set +e
      LATEST_FALLBACK_PATH=$(gcloud storage ls --sort-by="~updated" --limit=1 "${API_WHEELS_GCS_DIR}/${API_PREFIX}-*.whl" 2>/dev/null | head -n 1 || true)
      set -e
      if [[ -n "${LATEST_FALLBACK_PATH}" ]]; then
        ACTUAL_VERSION=$(basename "$LATEST_FALLBACK_PATH" | cut -d'-' -f2)
        set +e
        WHEEL_LIST=$(gcloud storage ls "${API_WHEELS_GCS_DIR}/${API_PREFIX}-${ACTUAL_VERSION}-*.whl" 2>/dev/null)
        set -e
        EXPECTED_VERSION="${ACTUAL_VERSION}"
        ACTUAL_DATE=$(echo "$ACTUAL_VERSION" | grep -o 'dev[0-9]*' | sed 's/dev//' || true)
        if [[ -n "${ACTUAL_DATE}" ]]; then
          EXTRA_BAZEL_ARGS="--define=DEV_BUILD=1 --define=DEV_VERSION=${ACTUAL_DATE}"
        fi
      fi
    fi
  fi
  
  if [[ -n "${WHEEL_LIST}" ]]; then
    echo "Downloading API wheels..."
    gcloud storage cp "${API_WHEELS_GCS_DIR}/${API_PREFIX}-${EXPECTED_VERSION}-*.whl" "${API_WHEELS_DIR}/"
  else
    echo "❌ Failed to find any matching API wheels in GCS!"
    exit 1
  fi
fi

echo "Building wheel using Bazelisk..."
bazelisk build //python/litert_lm_cli:wheel "$@" ${EXTRA_BAZEL_ARGS}

TEST_VENV="${WORKSPACE_ROOT}/python/litert_lm_cli/test_venv"

for PY_VER in "3.10" "3.11" "3.12" "3.13" "3.14"; do
  echo "------------------------------------------------"
  echo "Setting up temporary virtual environment for Python ${PY_VER}..."
  echo "------------------------------------------------"
  rm -rf "${TEST_VENV}"

  # Force uv to use or download the specific target Python version
  uv venv --python="${PY_VER}" "${TEST_VENV}"

  # Universal Cross-Platform venv Activation & Python binary selection
  if [[ -d "${TEST_VENV}/Scripts" ]]; then
    source "${TEST_VENV}/Scripts/activate"
    PY_EXE="python"
  else
    source "${TEST_VENV}/bin/activate"
    PY_EXE="python3"
  fi

  echo "Installing the freshly built CLI wheel and the GCS API wheels..."
  uv pip install --index-url https://pypi.org/simple --find-links "${API_WHEELS_DIR}" "${WHEEL_DIR}"/*.whl

  cd "${TEST_VENV}"

  # Run CLI tests
  bash "${WORKSPACE_ROOT}/python/litert_lm_cli/cli_tests.sh"

  cd "${WORKSPACE_ROOT}"
  deactivate
done

rm -rf "${TEST_VENV}"
echo "✨ Verification completed successfully for all Python versions!"
