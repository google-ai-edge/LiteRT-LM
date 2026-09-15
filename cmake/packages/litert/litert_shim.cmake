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


include("${LITERTLM_MODULES_DIR}/utils.cmake")
include("${LITERTLM_MODULES_DIR}/macros.cmake")
include("${LITERTLM_MODULES_DIR}/generators/generate_protobuf.cmake")

include("${LITERTLM_LITERT_CONFIG_PATH}")

set(LITERTLM_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/download" CACHE PATH "Directory for downloads")
set(LITERTLM_PYTHON_DOWNLOAD_DIR "${LITERTLM_DOWNLOAD_DIR}/python")

option(ENABLE_UV "Enables uv virtual environment for the build" ON)
set(LITERTLM_ENV_WRAPPER "" CACHE INTERNAL
  "LiteRT-LM: Environment variables for uv virtual environment")
set(LITERTLM_UV_ARGS     "" CACHE INTERNAL
  "LiteRT-LM: Arguments for uv virtual environment")

if(ENABLE_UV)
  find_program(UV_EXECUTABLE uv REQUIRED)
  set(LITERTLM_VENV_DIR "${CMAKE_BINARY_DIR}/.venv" CACHE PATH 
    "LiteRT-LM: Path to the uv virtual environment")
    message(STATUS "[LiteRTLM] --- Installing Python 3.13 ---")
    message(STATUS "[LiteRTLM] Python download location: ${LITERTLM_PYTHON_DOWNLOAD_DIR}")

    execute_process(
    COMMAND ${UV_EXECUTABLE} python install 3.13  --managed-python --no-cache --force
        --install-dir "${LITERTLM_PYTHON_DOWNLOAD_DIR}"
    )

    file(GLOB LITERTLM_PYTHON_DIR
        "${LITERTLM_PYTHON_DOWNLOAD_DIR}/*cpython-3.13-*/"
    )
    message(STATUS "[LiteRTLM] LITERTLM_PYTHON_DIR: ${LITERTLM_PYTHON_DIR}")
    set(LITERTLM_PYTHON_DIR ${LITERTLM_PYTHON_DIR} CACHE PATH "")
    set(LITERTLM_PYTHON_INCLUDE_DIR ${LITERTLM_PYTHON_DIR}/include/python3.13 CACHE PATH "")
    set(LITERTLM_PYTHON_BIN_DIR "${LITERTLM_PYTHON_DIR}/bin" CACHE PATH "")
    set(LITERTLM_PYTHON_LIB_DIR "${LITERTLM_PYTHON_DIR}/lib" CACHE PATH "")

    file(GLOB LITERTLM_PYTHON_EXE
        "${LITERTLM_PYTHON_BIN_DIR}/python3${CMAKE_EXECUTABLE_SUFFIX}"
    )
    message(STATUS "[LiteRTLM] LITERTLM_PYTHON_EXE: ${LITERTLM_PYTHON_EXE}")
    set(LITERTLM_PYTHON_EXE ${LITERTLM_PYTHON_EXE} CACHE PATH "")
    if(NOT EXISTS "${LITERTLM_VENV_DIR}")
      message(STATUS "[LiteRTLM] --- Creating uv virtual environment ---")
      message(STATUS "[LiteRTLM] Virtual environment path: ${LITERTLM_VENV_DIR}")
      message(STATUS "[LiteRTLM] Python executable path: ${LITERTLM_PYTHON_EXE}")

      execute_process(
        COMMAND ${UV_EXECUTABLE}
          venv "${LITERTLM_VENV_DIR}" --python "${LITERTLM_PYTHON_EXE}"
        COMMAND_ERROR_IS_FATAL ANY
      )
  endif()

  set(ENV{VIRTUAL_ENV} "${LITERTLM_VENV_DIR}")
  set(Python3_FIND_VIRTUALENV FIRST)

  if(LITERTLM_CMAKE_CROSSCOMPILING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  endif()
  message(STATUS "Using isolated Python: ${Python3_EXECUTABLE}")
  set(LITERTLM_ENV_WRAPPER 
    "VIRTUAL_ENV=${LITERTLM_VENV_DIR}"
    "PATH=${LITERTLM_VENV_DIR}/bin:$ENV{PATH}"
    CACHE INTERNAL "LiteRT-LM: Environment variables for uv virtual environment"
  )
  set(LITERTLM_UV_ARGS 
    "-DPython3_ROOT_DIR=${LITERTLM_VENV_DIR}"
    "-DPython3_FIND_VIRTUALENV=FIRST")
endif()

include_directories("${LITERTLM_PYTHON_INCLUDE_DIR}")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.1.0

)
set(PYBIND11_NOPYTHON OFF CACHE INTERNAL "")
set(PYBIND11_FINDPYTHON ON CACHE INTERNAL "")
set(PYBIND11_INCLUDE_DIR ${LITERTLM_PYTHON_INCLUDE_DIR} CACHE INTERNAL "" FORCE)
set(PYBIND11_PYTHON_EXECUTABLE_LAST "${LITERLM_PYTHON_EXE}" CACHE INTERNAL "" Force)

set(Python3_EXECUTABLE "${LITERTLM_PYTHON_EXE}" CACHE INTERNAL "" FORCE)
set(Python3_ROOT_DIR "${LITERTLM_PYTHON_DIR}" CACHE INTERNAL "")
set(Python3_FIND_STRATEGY "LOCATION" CACHE INTERNAL "")
FetchContent_MakeAvailable(pybind11)

link_libraries(pybind11::headers)


include("${LITERTLM_GTEST_CONFIG_PATH}")

include("${LITERTLM_ABSL_CONFIG_PATH}")
include("${LITERTLM_ABSL_AGGREGATE_PATH}")
generate_absl_aggregate()

include("${LITERTLM_PROTOBUF_CONFIG_PATH}")
include("${LITERTLM_PROTOBUF_AGGREGATE_PATH}")
generate_protobuf_aggregate()

include("${LITERTLM_FLATBUFFERS_CONFIG_PATH}")
include("${LITERTLM_FLATBUFFERS_AGGREGATE_PATH}")
generate_flatbuffers_aggregate()
generate_flatc_aggregate()

include("${LITERTLM_SENTENCEPIECE_CONFIG_PATH}")
include("${LITERTLM_SENTENCEPIECE_AGGREGATE_PATH}")
generate_sentencepiece_aggregate()

include("${LITERTLM_TFLITE_CONFIG_PATH}")
include("${LITERTLM_TFLITE_AGGREGATE_PATH}")
generate_tflite_aggregate()

if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers INTERFACE IMPORTED GLOBAL)
    set_target_properties(flatbuffers::flatbuffers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
)
endif()

if(NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED GLOBAL)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${LITERTLM_HOST_PROTOC}"
    )
endif()

if(NOT TARGET flatc)
    add_executable(flatc IMPORTED GLOBAL)
    set_target_properties(flatc PROPERTIES 
        IMPORTED_LOCATION "${LITERTLM_FLATC_EXECUTABLE}"
    )
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_JSON_SRC_DIR}/include"
    )
endif()

# TFLite configuration
set(TFLITE_SOURCE_DIR "${LITERTLM_TFLITE_SRC_DIR}" CACHE PATH "Path to TFLite source directory" FORCE)
set(TFLITE_BUILD_DIR "${LITERTLM_TFLITE_BUILD_DIR}" CACHE PATH "Path to TFLite build directory" FORCE)

include_directories(
  "${LITERTLM_ABSL_INCLUDE_DIR}"
  "${LITERTLM_FLATBUFFERS_INCLUDE_DIR}"
  "${LITERTLM_PROTOBUF_INCLUDE_DIR}"
  "${LITERTLM_SENTENCEPIECE_INCLUDE_PATHS}"
  "${LITERTLM_TOKENIZERS_INCLUDE_DIR}"
  "${LITERTLM_TOKENIZERS_SRC_DIR}"
  "${LITERTLM_TOKENIZERS_INCLUDE_DIR}"
  "${LITERTLM_MINIAUDIO_SRC_DIR}"
  "${LITERTLM_MINIZIP_SRC_DIR}/minizip"
  "${LITERTLM_MINJA_SRC_DIR}/include"
  "${LITERTLM_JSON_SRC_DIR}/include"
  "${LITERTLM_STB_SRC_DIR}"
  "${LITERTLM_KISSFFT_SRC_DIR}"
  "${LITERTLM_TFLITE_INCLUDE_DIR}"
  "${LITERTLM_TFLITE_SRC_DIR}"
  "${LITERTLM_RUY_INCLUDE_DIR}"
  "${LITERTLM_TFLITE_SRC_DIR}"
  "${LITERTLM_TENSORFLOW_SRC_DIR}"
  "${LITERTLM_TFLITE_BUILD_DIR}"
  "${LITERTLM_TFLITE_BUILD_DIR}/xnnpack/include"
  "${LITERTLM_TFLITE_BUILD_DIR}/FP16-source/include"
  "${LITERTLM_TFLITE_BUILD_DIR}/ruy"
  "${LITERTLM_TFLITE_BUILD_DIR}/gemmlowp"
  "${LITERTLM_TFLITE_BUILD_DIR}/eigen"
  "${LITERTLM_TENSORFLOW_PYBIND11_DIR}"
)

set(TFLITE_ENABLE_GPU OFF CACHE BOOL "[LiteRTLM] litert_shim.cmake" FORCE)
set(LITERT_ENABLE_GPU OFF CACHE BOOL "[LiteRTLM] litert_shim.cmake" FORCE)

if(LITERT_BUILD_CONFIG_DISABLE_GPU_VAL)
  add_compile_definitions(LITERT_DISABLE_GPU)
endif()

add_compile_definitions(
    LITERT_LM_EXTERNAL_CMAKE_BUILD
    LITERT_DISABLE_GPU
)

# [TODO] Refactor into macro for DRY principle.
# --- Toolchain-Specific Linker Flags ---
set(_LITERTLM_LINK_MULTIDEF "")
set(_LITERTLM_LINK_GROUP_START "")
set(_LITERTLM_LINK_GROUP_END "")
set(_LITERTLM_SYSLIBS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(APPLE)
        # AppleClang / Mach-O Linker
        set(_LITERTLM_LINK_MULTIDEF "-Wl,-multiply_defined,suppress")
        set(_LITERTLM_SYSLIBS "-lz -lpthread -ldl")
    elseif(ANDROID)
        # Android / Bionic (NO standalone rt or pthread)
        set(_LITERTLM_LINK_MULTIDEF "-Wl,--allow-multiple-definition")
        set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
        set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
        set(_LITERTLM_SYSLIBS "-lz -ldl -llog")
    else()
        # Linux / ELF Linker (GNU ld or LLD)
        set(_LITERTLM_LINK_MULTIDEF "-Wl,--allow-multiple-definition")
        set(_LITERTLM_LINK_GROUP_START "-Wl,--start-group")
        set(_LITERTLM_LINK_GROUP_END "-Wl,--end-group")
        set(_LITERTLM_SYSLIBS "-lz -lrt -lpthread -ldl")
    endif()
elseif(MSVC)
    # MSVC Linker
    set(_LITERTLM_LINK_MULTIDEF "/FORCE:MULTIPLE")
    set(_LITERTLM_SYSLIBS "")
endif()

set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} ${_LITERTLM_LINK_MULTIDEF} ${_LITERTLM_LINK_GROUP_START} ${_TFLITE_PAYLOAD} ${_SENTENCEPIECE_PAYLOAD} ${_FLATBUFFERS_PAYLOAD} ${_PROTOBUF_PAYLOAD} ${_ABSL_PAYLOAD} ${_LITERTLM_SYSLIBS} ${_LITERTLM_LINK_GROUP_END}"
    CACHE STRING "[LiteRTLM] litert_shim.cmake" FORCE)
