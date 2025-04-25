#!/bin/bash
#
# Example commands:
#
# Run on CPU (GPU is not supported yet) with new LiteRT path:
# ./third_party/odml/litert_lm/runtime/engine/run_litert_lm.sh \
# --models=gemma3_1b_1280_8bit \
# --backend=cpu
#
# Run on GPU with legacy engine:
# ./third_party/odml/litert_lm/runtime/engine/run_litert_lm.sh \
# --models=gemma3_1b_1280_8bit \
# --use_legacy_engine \
# --backend=gpu
#
# Run on CPU with legacy engine:
# ./third_party/odml/litert_lm/runtime/engine/run_litert_lm.sh \
# --models=gemma3_1b_1280_8bit \
# --use_legacy_engine \
# --backend=cpu
#
source gbash.sh || exit

DEFINE_bool skip_model_push false "Skip pushing the model to the device."
DEFINE_bool force_full_protos false "Force full protos."
DEFINE_string backend "cpu" "Backend to use (cpu or gpu)."
DEFINE_string input_prompt "What is the highest building in Paris?" "Input prompt to use for testing LLM execution."
DEFINE_bool use_legacy_engine false "Use legacy engine (i.e. the Executor implementations with the TfLite interpreter.)."
DEFINE_string model_config "third_party/odml/litert_lm/runtime/engine/models.sh" "Model configurations to source from."

# Pre-parse the model_config flag to define the `models` enum.
if [[ "${@}" =~ --model_config[=\ ]([^-]*) ]]; then
  FLAGS_model_config="${BASH_REMATCH[1]%% *}"
fi
# Extract function names using grep
MODEL_CONFIGS=$(grep -Po "\w+(?=\(\))" "${FLAGS_model_config}")
MODEL_ENUM=$(echo "${MODEL_CONFIGS}" | tr -s ' \n' ' ,' | sed 's/,$//')
DEFINE_array models --type=enum --enum="${MODEL_ENUM}" "" "The list of models to run."
: "${ADB_CMD:=adb}"

gbash::init_google "$@"
source ${FLAGS_model_config}

function check_device_file_ready() {
  local MODEL_PATH=$1
  local DEVICE_MODEL_PATH=$2
  if ${ADB_CMD} shell [[ -e "${DEVICE_MODEL_PATH}" ]]; then
    echo "Computing device resource checksum: ${DEVICE_MODEL_PATH}"
    local DEVICE_CHECKSUM=$(${ADB_CMD} shell sha1sum "${DEVICE_MODEL_PATH}" | awk '{print $1}')
  fi

  if command -v fileutil &>/dev/null; then
    if fileutil test -f "${MODEL_PATH}"; then
      echo "Computing remote resource checksum: ${MODEL_PATH}"
      local CNS_CHECKSUM=$(fileutil sha1checksum "${MODEL_PATH}" | awk '{print $2}')
      if [[ "${CNS_CHECKSUM:-unset cns}" == "${DEVICE_CHECKSUM:-unset device}" ]]; then
        return 0  # Considered "true" by if.
      fi
    fi
  fi
  return 1
}

# Compile the binary.
BUILD_COMMAND="bazel build -c opt "
if (( FLAGS_force_full_protos )); then
  BUILD_COMMAND+="--config force_full_protos "
fi
if (( FLAGS_use_legacy_engine )); then
  echo "Using legacy engine."
  BUILD_COMMAND+="--copt=-DUSE_LEGACY_ENGINE "
fi
BUILD_COMMAND+="--android_ndk_min_sdk_version=26 "
# The UNDEBUG flag is required to get ERROR logs.
BUILD_COMMAND+="--copt=-UNDEBUG "
# The following xnnpack flags are required to get reasonable responses using
# CPU.
BUILD_COMMAND+="--define=xnnpack_use_latest_ops=true "
BUILD_COMMAND+="--define=xnnpack_enable_subgraph_reshaping=true "
BUILD_COMMAND+="--config=android_arm64 --copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1 third_party/odml/litert_lm/runtime/engine:litert_lm_main"
BINARY_PATH="bazel-bin/third_party/odml/litert_lm/runtime/engine/litert_lm_main"

# Process each of the specified models.
MODELS_TO_EXECUTE=("${FLAGS_models[@]}")
if [ ${#MODELS_TO_EXECUTE[@]} -eq 0 ]; then
  echo "The provided model list is empty. Please use the --models flag to specify at least one model from the following list: (${MODEL_ENUM}) to run."
  exit 1
fi
for MODEL_SETTING in "${MODELS_TO_EXECUTE[@]}"
do
  eval "$MODEL_SETTING"
  OUTPUT_DIR=/data/local/tmp/$EXP_NAME
  TMP_DIR=/tmp/$EXP_NAME
  CACHE_DIR=$OUTPUT_DIR/cache_${FLAGS_backend}
  echo $'\n====================='
  echo "Experiment setting:"
  echo "  EXP_NAME: ${EXP_NAME}"
  echo "  MODEL_PATH: ${MODEL_PATH}"
  echo "  BACKEND: ${FLAGS_backend}"
  echo "  MODEL_FILENAME: ${MODEL_FILENAME}"
  echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
  echo "  TMP_DIR: ${TMP_DIR}"
  echo "  CACHE_DIR: ${CACHE_DIR}"
  echo "  INPUT_PROMPT: ${FLAGS_input_prompt}"

  if (( ! FLAGS_skip_model_push )); then
    echo "Checking if model is already on device..."
    if check_device_file_ready "${MODEL_PATH}" "${OUTPUT_DIR}/${MODEL_FILENAME}"; then
      echo "Model is already on device."
    else
      echo "Pushing model to device..."
      rm -R -f $TMP_DIR
      mkdir -p $TMP_DIR
      fileutil mirror -dest_is_file -parallelism=128 $MODEL_PATH "${TMP_DIR}/${MODEL_FILENAME}"
      adb shell rm -R -f $OUTPUT_DIR
      adb shell mkdir -p $OUTPUT_DIR
      adb push "${TMP_DIR}/${MODEL_FILENAME}" $OUTPUT_DIR
    fi
  fi

  BINARY_ARGS="--backend=${FLAGS_backend} "
  BINARY_ARGS+="--model_path=${OUTPUT_DIR}/${MODEL_FILENAME} "
  BINARY_ARGS+="--input_prompt=\"${FLAGS_input_prompt}\""
  echo "Running binary with args: ${BINARY_ARGS}"

  $BUILD_COMMAND && \
  adb push $BINARY_PATH $OUTPUT_DIR && \
  adb shell "${OUTPUT_DIR}/litert_lm_main ${BINARY_ARGS}"
done

