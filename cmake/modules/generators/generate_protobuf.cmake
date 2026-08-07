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

# --- Protobuf ---
function(generate_protobuf TARGET_NAME _root_path)
    set(GENERATED_SRCS)
    set(GENERATED_HDRS)

    add_custom_command(
        OUTPUT ""
        COMMAND $<TARGET_FILE:protobuf::protoc>
        ARGS --version
    )

    foreach(PROTO_FILE ${LITERTLM_PROTO_FILES})
        file(RELATIVE_PATH REL_PROTO_PATH "${_root_path}" "${PROTO_FILE}")

        get_filename_component(REL_DIR "${REL_PROTO_PATH}" DIRECTORY)
        get_filename_component(FIL_WE "${REL_PROTO_PATH}" NAME_WE)

        set(OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}")
        set(SRC_FILE "${OUT_DIR}/${FIL_WE}.pb.cc")
        set(HDR_FILE "${OUT_DIR}/${FIL_WE}.pb.h")

        file(MAKE_DIRECTORY "${OUT_DIR}")

        add_custom_command(
            OUTPUT "${SRC_FILE}" "${HDR_FILE}"
            COMMAND $<TARGET_FILE:protobuf::protoc>
            ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
                 -I "${_root_path}"
                 "${PROTO_FILE}"
            DEPENDS "${PROTO_FILE}" protobuf::protoc
            COMMENT "Generating C++ from ${REL_PROTO_PATH}"
            VERBATIM
        )

        list(APPEND GENERATED_SRCS "${SRC_FILE}")
        list(APPEND GENERATED_HDRS "${HDR_FILE}")
    endforeach()
    set_source_files_properties(${GENERATED_SRCS} ${GENERATED_HDRS} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET_NAME} PRIVATE ${GENERATED_SRCS} ${GENERATED_HDRS})
endfunction()
