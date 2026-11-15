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

function(generate_flatbuffers TARGET_NAME)
    set(GENERATED_HDRS "")

    foreach(FBS_FILE IN LISTS LITERTLM_FLATBUFFERS_FILES)
        get_filename_component(FBS_DIR "${FBS_FILE}" DIRECTORY)
        get_filename_component(FIL_WE "${FBS_FILE}" NAME_WE)

        set(HDR_FILE "${FBS_DIR}/${FIL_WE}_generated.h")

        add_custom_command(
            OUTPUT "${HDR_FILE}"
            COMMAND LiteRTLM::flatbuffers::flatc
                    --cpp --gen-object-api --reflect-names --gen-mutable 
                    -o "${FBS_DIR}" 
                    "${FBS_FILE}"
            DEPENDS "${FBS_FILE}" LiteRTLM::flatbuffers::flatc
            COMMENT "[LiteRTLM] Generating Flatbuffers header: ${HDR_FILE}"
            VERBATIM
        )
        list(APPEND GENERATED_HDRS "${HDR_FILE}")
    endforeach()

    add_custom_target(litertlm_fbs_header_gen DEPENDS LITERTLM_HOST_FLATC)

    add_dependencies(${TARGET_NAME} litertlm_fbs_header_gen)

    target_sources(${TARGET_NAME} INTERFACE ${GENERATED_HDRS})
endfunction()