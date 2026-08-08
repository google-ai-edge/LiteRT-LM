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

function(patch_generated_file GENERATED_FILE ORIGINAL_STRING PATCH_STRING)
    if(NOT EXISTS "${GENERATED_FILE}")
        message(AUTHOR_WARNING "[LiteRTLM] ⚠️ Patch skipped (File missing): ${GENERATED_FILE}")
        return()
    endif()

    file(READ "${GENERATED_FILE}" GEN_FILE_CONTENT)

    string(REPLACE 
        "${ORIGINAL_STRING}" 
        "${PATCH_STRING}" 
        GEN_FILE_CONTENT_PATCHED 
        "${GEN_FILE_CONTENT}"
    )

    if("${GEN_FILE_CONTENT}" STREQUAL "${GEN_FILE_CONTENT_PATCHED}")
    else()
        file(WRITE "${GENERATED_FILE}" "${GEN_FILE_CONTENT_PATCHED}")
        message(STATUS "[LiteRTLM] ✅ Patched ${GENERATED_FILE}")
    endif()
endfunction()

function(apply_patches_from_json JSON_CONFIG_FILE)
    if(NOT EXISTS "${JSON_CONFIG_FILE}")
        message(FATAL_ERROR "[LiteRTLM] Patch config not found: ${JSON_CONFIG_FILE}")
    endif()
    file(READ "${JSON_CONFIG_FILE}" JSON_CONTENT)

    string(JSON PATCH_COUNT LENGTH "${JSON_CONTENT}" "patches")
    math(EXPR STOP_IDX "${PATCH_COUNT} - 1")

    foreach(IDX RANGE ${STOP_IDX})
        string(JSON TARGET_REL_PATH     GET "${JSON_CONTENT}" "patches" ${IDX} "src_file")
        string(JSON FIND_STRING         GET "${JSON_CONTENT}" "patches" ${IDX} "find_string")
        string(JSON PATCH_FILE_REL_PATH GET "${JSON_CONTENT}" "patches" ${IDX} "patch_file")

        set(FULL_TARGET_PATH "${LITERTLM_GENERATED_SRC_DIR}/${TARGET_REL_PATH}")
        set(FULL_PATCH_FILE_PATH "${LITERTLM_PROJECT_ROOT}/cmake/patches/${PATCH_FILE_REL_PATH}")

        if(NOT EXISTS "${FULL_PATCH_FILE_PATH}")
            message(FATAL_ERROR "[LiteRTLM] ❌ Shim file missing: ${FULL_PATCH_FILE_PATH}")
        endif()

        file(READ "${FULL_PATCH_FILE_PATH}" REPLACEMENT_CODE)

        patch_generated_file(
            "${FULL_TARGET_PATH}"
            "${FIND_STRING}"
            "${REPLACEMENT_CODE}"
        )
    endforeach()
endfunction()