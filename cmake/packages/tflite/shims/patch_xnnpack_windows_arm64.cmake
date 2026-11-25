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

# Patches XNNPACK's CMakeLists.txt right before add_subdirectory(xnnpack) so
# Windows ARM64 matches the Bazel build (third_party/xnnpack/windows_arm64.patch):
#
# 1. Keep XNNPACK_ENABLE_ARM_I8MM ON on Windows ARM64. XNNPACK's CMakeLists.txt
#    sets XNNPACK_TARGET_PROCESSOR_AARCH64_DEFAULTS OFF on Windows and then
#    unconditionally resets XNNPACK_ENABLE_ARM_I8MM to OFF. Without I8MM,
#    qs8_qc4w GEMM on Windows ARM64 falls back to scalar C (~12x slower on
#    Snapdragon X Elite), and qd8_f32_* / qs8_qc8w drop from I8MM to DotProd.
#    KleidiAI, SVE2, SME, SME2 and SME_ACLE remain OFF.
# 2. Gate FP16-output I8MM microkernels (qd8-f16-*__neoni8mm.c) on
#    XNNPACK_ENABLE_ARM_FP16_VECTOR and NEONI8MMBF16 microkernels on
#    XNNPACK_ENABLE_ARM_BF16, so MSVC cl.exe (which lacks FP16/BF16 ACLE
#    intrinsics) compiles the integer/FP32 I8MM kernels cleanly.

if(DEFINED xnnpack_SOURCE_DIR AND EXISTS "${xnnpack_SOURCE_DIR}/CMakeLists.txt")
  set(_litertlm_xnnpack_cmakelists "${xnnpack_SOURCE_DIR}/CMakeLists.txt")
  file(READ "${_litertlm_xnnpack_cmakelists}" _litertlm_xnn_content)
  string(REPLACE "\r\n" "\n" _litertlm_xnn_content "${_litertlm_xnn_content}")
  set(_litertlm_xnn_original "${_litertlm_xnn_content}")

  set(_litertlm_xnn_i8mm_needle
    "IF(NOT XNNPACK_TARGET_PROCESSOR_AARCH64_DEFAULTS)\n  SET(XNNPACK_ENABLE_ARM_I8MM OFF)")
  set(_litertlm_xnn_prod_needle
    "  IF(XNNPACK_ENABLE_ARM_I8MM)\n    LIST(APPEND PROD_MICROKERNEL_SRCS \${PROD_NEONI8MM_MICROKERNEL_SRCS})\n    LIST(APPEND PROD_MICROKERNEL_SRCS \${PROD_NEONI8MMBF16_MICROKERNEL_SRCS})\n  ENDIF()")
  set(_litertlm_xnn_nonprod_needle
    "  IF(XNNPACK_ENABLE_ARM_I8MM)\n    LIST(APPEND NON_PROD_MICROKERNEL_SRCS \${NON_PROD_NEONI8MM_MICROKERNEL_SRCS})\n    LIST(APPEND NON_PROD_MICROKERNEL_SRCS \${NON_PROD_NEONI8MMBF16_MICROKERNEL_SRCS})\n  ENDIF()")

  # Skip if already patched (re-configure on an existing build tree).
  string(FIND "${_litertlm_xnn_content}" "LITERTLM_XNNPACK_WIN_ARM64_PATCHED" _litertlm_xnn_patched)
  if(_litertlm_xnn_patched EQUAL -1)
    set(_litertlm_xnn_missing "")
    foreach(_litertlm_needle_name IN ITEMS
        _litertlm_xnn_i8mm_needle
        _litertlm_xnn_prod_needle
        _litertlm_xnn_nonprod_needle)
      string(FIND "${_litertlm_xnn_content}" "${${_litertlm_needle_name}}" _litertlm_pos)
      if(_litertlm_pos EQUAL -1)
        list(APPEND _litertlm_xnn_missing "${_litertlm_needle_name}")
      endif()
    endforeach()

    if(_litertlm_xnn_missing)
      if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([Aa][Rr][Mm]64|[Aa][Aa][Rr][Cc][Hh]64)$")
        message(FATAL_ERROR
          "[LiteRTLM] Failed to patch XNNPACK CMakeLists.txt for Windows ARM64 I8MM "
          "(patterns not found: ${_litertlm_xnn_missing}). Update patch_xnnpack_windows_arm64.cmake "
          "to match the pinned XNNPACK revision.")
      else()
        message(WARNING
          "[LiteRTLM] XNNPACK CMakeLists.txt did not match Windows ARM64 I8MM patch patterns "
          "(${_litertlm_xnn_missing}); skipping.")
      endif()
    else()
      # 1. Allow XNNPACK_ENABLE_ARM_I8MM to stay ON on Windows ARM64.
      string(REPLACE
        "${_litertlm_xnn_i8mm_needle}"
        "# LITERTLM_XNNPACK_WIN_ARM64_PATCHED\nIF(NOT XNNPACK_TARGET_PROCESSOR_AARCH64_DEFAULTS)\n  IF(NOT (CMAKE_SYSTEM_NAME STREQUAL \"Windows\" AND XNNPACK_TARGET_PROCESSOR MATCHES \"^[Aa][Rr][Mm]64\"))\n    SET(XNNPACK_ENABLE_ARM_I8MM OFF)\n  ENDIF()"
        _litertlm_xnn_content "${_litertlm_xnn_content}"
      )

      # 2. Gate FP16-output I8MM sources on XNNPACK_ENABLE_ARM_FP16_VECTOR and
      #    NEONI8MMBF16 sources on XNNPACK_ENABLE_ARM_BF16.
      string(REPLACE
        "${_litertlm_xnn_prod_needle}"
        "  IF(XNNPACK_ENABLE_ARM_I8MM)\n    IF(NOT XNNPACK_ENABLE_ARM_FP16_VECTOR)\n      LIST(FILTER PROD_NEONI8MM_MICROKERNEL_SRCS EXCLUDE REGEX \"f16\")\n    ENDIF()\n    LIST(APPEND PROD_MICROKERNEL_SRCS \${PROD_NEONI8MM_MICROKERNEL_SRCS})\n    IF(XNNPACK_ENABLE_ARM_BF16)\n      LIST(APPEND PROD_MICROKERNEL_SRCS \${PROD_NEONI8MMBF16_MICROKERNEL_SRCS})\n    ENDIF()\n  ENDIF()"
        _litertlm_xnn_content "${_litertlm_xnn_content}"
      )
      string(REPLACE
        "${_litertlm_xnn_nonprod_needle}"
        "  IF(XNNPACK_ENABLE_ARM_I8MM)\n    IF(NOT XNNPACK_ENABLE_ARM_FP16_VECTOR)\n      LIST(FILTER NON_PROD_NEONI8MM_MICROKERNEL_SRCS EXCLUDE REGEX \"f16\")\n    ENDIF()\n    LIST(APPEND NON_PROD_MICROKERNEL_SRCS \${NON_PROD_NEONI8MM_MICROKERNEL_SRCS})\n    IF(XNNPACK_ENABLE_ARM_BF16)\n      LIST(APPEND NON_PROD_MICROKERNEL_SRCS \${NON_PROD_NEONI8MMBF16_MICROKERNEL_SRCS})\n    ENDIF()\n  ENDIF()"
        _litertlm_xnn_content "${_litertlm_xnn_content}"
      )

      if(NOT _litertlm_xnn_content STREQUAL _litertlm_xnn_original)
        message(STATUS "[LiteRTLM] Patched XNNPACK CMakeLists.txt for Windows ARM64 I8MM support")
        file(WRITE "${_litertlm_xnnpack_cmakelists}" "${_litertlm_xnn_content}")
      endif()
    endif()
  endif()
endif()
