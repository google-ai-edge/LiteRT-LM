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


file(GLOB_RECURSE FBS_FILES "${SCHEMA_DIR}/*.fbs")

if(NOT FBS_FILES)
    message(WARNING "[LiteRTLM] No .fbs files found in ${SCHEMA_DIR}")
    return()
endif()

foreach(FBS_FILE ${FBS_FILES})
    get_filename_component(FBS_DIR "${FBS_FILE}" DIRECTORY)
    execute_process(
        COMMAND ${FLATC_BIN} --cpp --gen-object-api --reflect-names --gen-mutable -o "${FBS_DIR}" "${FBS_FILE}_generated.h"
        RESULT_VARIABLE RET_CODE
    )

    if(NOT RET_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to compile ${FBS_FILE}")
    endif()
endforeach()