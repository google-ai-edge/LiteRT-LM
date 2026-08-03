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

set -ex

MODEL_FILENAME="gemma-4-E2B-it.litertlm"
GCS_MODEL_PATH="gs://litert-lm-api/models/${MODEL_FILENAME}"
HF_REPO="litert-community/gemma-4-E2B-it-litert-lm"

CACHE_DIR="/tmp/litert_lm_cli_tests_cache"
mkdir -p "${CACHE_DIR}"
MODEL_REF="${CACHE_DIR}/${MODEL_FILENAME}"

if [[ -f "${MODEL_REF}" ]]; then
  echo "✅ Model already downloaded at ${MODEL_REF}"
  EXTRA_RUN_ARGS=""
elif [[ -n "${KOKORO_GFILE_DIR:-}" && -f "${KOKORO_GFILE_DIR}/${MODEL_FILENAME}" ]]; then
  echo "✅ Found pre-fetched model in KOKORO_GFILE_DIR"
  cp "${KOKORO_GFILE_DIR}/${MODEL_FILENAME}" "${MODEL_REF}"
  EXTRA_RUN_ARGS=""
else
  echo "Attempting to fetch Gemma model from GCS..."
  # Temporarily disable exit-on-error to allow fallback
  set +e
  gcloud storage cp "${GCS_MODEL_PATH}" "${MODEL_REF}"
  GCS_STATUS=$?
  set -e

  if [ $GCS_STATUS -eq 0 ]; then
    echo "✅ Successfully downloaded model from GCS."
    EXTRA_RUN_ARGS=""
  else
    echo "⚠️ Failed to download from GCS. Falling back to HuggingFace download on-the-fly..."
    MODEL_REF="${MODEL_FILENAME}"
    EXTRA_RUN_ARGS="--from-huggingface-repo ${HF_REPO}"
  fi
fi

for BACKEND in cpu gpu; do
  echo "Testing litert-lm run with real model on $BACKEND backend..."
  RUN_OUTPUT=$(litert-lm run ${EXTRA_RUN_ARGS} ${MODEL_REF} --prompt "What is the capital of France?" --backend $BACKEND)
  echo "Run Output ($BACKEND): $RUN_OUTPUT"
  if [[ ! "$RUN_OUTPUT" =~ "Paris" ]]; then
    echo "❌ Expected 'Paris' in run output for $BACKEND, but got: $RUN_OUTPUT"
    exit 1
  fi
  echo "✅ litert-lm run passed with real model on $BACKEND!"

  echo "Testing litert-lm serve with real model on $BACKEND backend..."
  # Serve runs a server, so we run it in the background
  litert-lm serve --port 8000 &
  SERVE_PID=$!

  # Give the server a few seconds to start up (and potentially download from HF if it wasn't cached)
  sleep 25

  # Send a chat completion request to the local server
  RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL_REF},${BACKEND}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}]}")

  echo "Serve Response ($BACKEND): $RESPONSE"

  # Kill the background server
  kill -9 $SERVE_PID

  if [[ ! "$RESPONSE" =~ "Paris" ]]; then
    echo "❌ Expected 'Paris' in serve response for $BACKEND, but got: $RESPONSE"
    exit 1
  fi
  echo "✅ litert-lm serve passed with real model on $BACKEND!"
done

echo "✅ All basic and real-model CLI tests passed! The pip package is structurally sound."
