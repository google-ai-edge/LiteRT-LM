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

include(FetchContent)

# --- ANTLR ---
set(LITERTLM_ANTLR_SRC_DIR 
  ${CMAKE_BINARY_DIR}/_deps/antlr_lib-src/runtime/Cpp/runtime/src 
  CACHE PATH "Path to antlr source directory"
)
set(LITERTLM_ANTLR_TAG "7d5770395bb7b02eb56e7c62662cb1d7c08f42a3" CACHE STRING "ANTLR git tag")
FetchContent_Declare(
  antlr_lib
  GIT_REPOSITORY https://github.com/antlr/antlr4.git
  GIT_TAG ${LITERTLM_ANTLR_TAG}
  GIT_SHALLOW true
  SOURCE_SUBDIR runtime/Cpp
  PATCH_COMMAND 
    ${CMAKE_COMMAND} 
    -DTARGET_FILE=${LITERTLM_ANTLR_SRC_DIR}/../../CMakeLists.txt 
    -P${LITERTLM_PATCHES_DIR}/antlr_patch.cmake
)
block()
  set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
  set(ANTLR_BUILD_STATIC ON)
  set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(antlr_lib)
endblock()
if(TARGET antlr4_static)
  add_library(antlr_lib ALIAS antlr4_static)
endif()

FetchContent_Declare(
  antlr_tool
  URL "https://www.antlr.org/download/antlr-4.13.2-complete.jar"
  DOWNLOAD_DIR "${LITERTLM_THIRD_PARTY_DIR}/antlr"
  DOWNLOAD_NO_EXTRACT TRUE
)

FetchContent_GetProperties(antlr_tool)
if(NOT antlr_tool_POPULATED)
  message(STATUS "[LiteRTLM] Fetching ANTLR 4.13.2 Tool...")
  FetchContent_Populate(antlr_tool)
endif()
set(ANTLR_JAR_PATH "${LITERTLM_THIRD_PARTY_DIR}/antlr/antlr-4.13.2-complete.jar")
message(STATUS "[LiteRTLM] ANTLR JAR located at: ${ANTLR_JAR_PATH}")

# --- Corrosion ---
set(LITERTLM_CORROSION_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/corrosion CACHE PATH "Path to corrosion source directory")
set(LITERTLM_CORROSION_INCLUDE_DIR "${CMAKE_BINARY_DIR}/corrosion_generated/cxxbridge/litertlm_cxx_bridge/include")
set(LITERTLM_CORROSION_TAG "v0.6.1" CACHE STRING "Corrosion git tag")
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG ${LITERTLM_CORROSION_TAG}
    SOURCE_DIR ${LITERTLM_CORROSION_SRC_DIR}
)
FetchContent_MakeAvailable(Corrosion)

if(TARGET Corrosion)
  include(Corrosion)
endif()

# --- Llguidance ---
set(LITERTLM_LLGUIDANCE_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/llguidance CACHE PATH "Path to llguidance source directory")
set(LITERTLM_LLGUIDANCE_INCLUDE_DIR "${LITERTLM_THIRD_PARTY_DIR}/llguidance/parser")
set(LITERTLM_LLGUIDANCE_TAG "v1.3.0" CACHE STRING "Llguidance git tag")
FetchContent_Declare(
  llguidance
  GIT_REPOSITORY https://github.com/guidance-ai/llguidance.git
  GIT_TAG ${LITERTLM_LLGUIDANCE_TAG}
  SOURCE_DIR ${LITERTLM_LLGUIDANCE_SRC_DIR}
)
FetchContent_Populate(llguidance)

# --- LibPNG ---
set(LITERTLM_LIBPNG_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/libpng CACHE PATH "Path to libpng source directory")
set(LITERTLM_LIBPNG_TAG "v1.6.40" CACHE STRING "LibPNG git tag")
FetchContent_Declare(
  libpng_lib
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG ${LITERTLM_LIBPNG_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_LIBPNG_SRC_DIR}
)
block()
  set(PNG_SHARED OFF)
  set(PNG_TESTS OFF)
  set(PNG_EXECUTABLES OFF)
  set(SKIP_INSTALL_ALL ON)

  # Handle ZLIB dependency injection for PNG
  if(TARGET zlib)
    get_target_property(ZLIB_INCLUDE_DIR zlib INTERFACE_INCLUDE_DIRECTORIES)
    set(ZLIB_LIBRARY zlib)
  elseif(TARGET zlibstatic)
    get_target_property(ZLIB_INCLUDE_DIR zlibstatic INTERFACE_INCLUDE_DIRECTORIES)
    set(ZLIB_LIBRARY zlibstatic)
  endif()

  FetchContent_MakeAvailable(libpng_lib)
endblock()
if(TARGET png_static)
  add_library(libpng_lib ALIAS png_static)
endif()

# --- KissFFT ---
set(LITERTLM_KISSFFT_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/kissfft CACHE PATH "Path to kissfft source directory")
set(LITERTLM_KISSFFT_TAG "131.2.0" CACHE STRING "KissFFT git tag")
FetchContent_Declare(
  kissfft_lib
  GIT_REPOSITORY https://github.com/mborgerding/kissfft
  GIT_TAG ${LITERTLM_KISSFFT_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_KISSFFT_SRC_DIR}
)
block()
  cmake_policy(SET CMP0077 OLD)
  set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
  set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(kissfft_lib)
endblock()
if(TARGET kissfft)
  add_library(kissfft_lib ALIAS kissfft)
endif()

# --- MiniAudio ---
set(LITERTLM_MINIAUDIO_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/miniaudio CACHE PATH "Path to miniaudio source directory")
set(LITERTLM_MINIAUDIO_TAG "0.11.25" CACHE STRING "MiniAudio git tag")
FetchContent_Declare(
  miniaudio_lib
  GIT_REPOSITORY https://github.com/mackron/miniaudio
  GIT_TAG ${LITERTLM_MINIAUDIO_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_MINIAUDIO_SRC_DIR}
)
FetchContent_MakeAvailable(miniaudio_lib)
if(TARGET miniaudio)
  add_library(miniaudio_lib ALIAS miniaudio)
endif()

# --- MiniZip ---
set(LITERTLM_MINIZIP_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/minizip CACHE PATH "Path to minizip source directory")
set(LITERTLM_MINIZIP_TAG aee7fbddf118d9363575af96310c29fa747d70c0 CACHE STRING "MiniZip git tag")
FetchContent_Declare(
  minizip_lib
  GIT_REPOSITORY https://github.com/domoticz/minizip
  GIT_TAG ${LITERTLM_MINIZIP_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_MINIZIP_SRC_DIR}
)
block()
  FetchContent_MakeAvailable(minizip_lib)
endblock()

if(TARGET minizip)
  add_library(minizip_lib ALIAS minizip)
endif()

# --- Minja ---
set(LITERTLM_MINJA_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/minja CACHE PATH "Path to minja source directory")
set(LITERTLM_MINJA_TAG "021c2293c187789ef13d56c6cfd89c9b134fd80f" CACHE STRING "Minja git tag")
FetchContent_Declare(
  minja_lib
  GIT_REPOSITORY https://github.com/google/minja
  GIT_TAG ${LITERTLM_MINJA_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_MINJA_SRC_DIR}
  PATCH_COMMAND ${CMAKE_COMMAND} -DTARGET_FILE=${LITERTLM_MINJA_SRC_DIR}/CMakeLists.txt -P${LITERTLM_PATCHES_DIR}/minja_patch.cmake
)
  # Turn off tests and examples
set(MINJA_TEST_ENABLED OFF CACHE BOOL "")
set(MINJA_EXAMPLE_ENABLE OFF CACHE BOOL "")
set(MINJA_FUZZTEST_ENABLED OFF CACHE BOOL "")
set(MINJA_FUZZTEST_FUZZING_MODE OFF CACHE BOOL "")

FetchContent_MakeAvailable(minja_lib)

if(TARGET minja)
  add_library(minja_lib ALIAS minja)
endif()

# --- JSON (Header Only - Populated) ---
set(LITERTLM_JSON_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/json CACHE PATH "Path to json headers")
set(LITERTLM_JSON_TAG "v3.12.0" CACHE STRING "JSON git tag")
FetchContent_Declare(
  json_lib
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG ${LITERTLM_JSON_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_JSON_SRC_DIR}
)
FetchContent_Populate(json_lib)
if(NOT TARGET json_lib)
  add_library(json_lib INTERFACE IMPORTED GLOBAL)
  target_include_directories(json_lib INTERFACE ${LITERTLM_JSON_SRC_DIR}/include)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
  add_library(nlohmann_json::nlohmann_json ALIAS json_lib)
endif()

if(NOT TARGET LiteRTLM::nlohmann_json::nlohmann_json)
  add_library(LiteRTLM::nlohmann_json::nlohmann_json ALIAS json_lib)
endif()

# --- STB (Header Only - Populated) ---
set(LITERTLM_STB_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/stb_lib CACHE PATH "Path to libstb headers")
set(LITERTLM_STB_TAG "master" CACHE STRING "STB git tag")
FetchContent_Declare(
  stb_lib
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG ${LITERTLM_STB_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_STB_SRC_DIR}
)
FetchContent_Populate(stb_lib)
if(NOT TARGET stb_lib)
  add_library(stb_lib INTERFACE)
  target_include_directories(stb_lib INTERFACE ${LITERTLM_STB_SRC_DIR})
endif()

# --- ZLIB ---
set(LITERTLM_ZLIB_SRC_DIR ${LITERTLM_THIRD_PARTY_DIR}/zlib CACHE PATH "Path to zlib source directory")
set(LITERTLM_ZLIB_TAG "master" CACHE STRING "ZLIB git tag")
FetchContent_Declare(
  zlib_lib
  GIT_REPOSITORY https://github.com/madler/zlib
  GIT_TAG ${LITERTLM_ZLIB_TAG}
  GIT_SHALLOW true
  SOURCE_DIR ${LITERTLM_ZLIB_SRC_DIR}
)
block()
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(zlib_lib)
endblock()
if(TARGET zlibstatic)
  add_library(zlib_lib ALIAS zlibstatic)
elseif(TARGET zlib)
  add_library(zlib_lib ALIAS zlib)
endif()

# --- Path Exports ---
set(LITERTLM_FETCHCONTENT_MODULE_SRC_DIRS
  ${LITERTLM_ANTLR_SRC_DIR}
  ${LITERTLM_KISSFFT_SRC_DIR}
  ${LITERTLM_LLGUIDANCE_SRC_DIR}/parser
  ${LITERTLM_MINIAUDIO_SRC_DIR}
  ${LITERTLM_MINIZIP_SRC_DIR}
  ${LITERTLM_MINJA_SRC_DIR}
  ${LITERTLM_JSON_SRC_DIR}
  ${LITERTLM_STB_SRC_DIR}
  ${LITERTLM_ZLIB_SRC_DIR}
  ${LITERTLM_CORROSION_SRC_DIR}
)

set(LITERTLM_FETCHCONTENT_MODULE_INCLUDE_DIR
  ${LITERTLM_ANTLR_SRC_DIR}
  ${LITERTLM_KISSFFT_SRC_DIR}
  ${LITERTLM_LLGUIDANCE_SRC_DIR}/parser
  ${LITERTLM_MINIAUDIO_SRC_DIR}
  ${LITERTLM_MINIZIP_SRC_DIR}/minizip
  ${LITERTLM_MINJA_SRC_DIR}/include
  ${LITERTLM_JSON_SRC_DIR}/include
  ${LITERTLM_STB_SRC_DIR}
  ${LITERTLM_ZLIB_SRC_DIR}
  ${LITERTLM_CORROSION_INCLUDE_DIR}
)

add_custom_target(fetch_content_complete)
add_dependencies(fetch_content_complete
    antlr_lib
    libpng_lib
    kissfft_lib
    miniaudio_lib
    minizip_lib
    minja_lib
    json_lib
    stb_lib
    zlib_lib
)