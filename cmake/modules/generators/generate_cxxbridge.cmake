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

# --- RUST ---
set(LITERTLM_RUST_FILES
    "${LITERTLM_PROJECT_ROOT}/runtime/components/rust/minijinja_template.rs"
    "${LITERTLM_PROJECT_ROOT}/runtime/components/tool_use/rust/parsers.rs"

    CACHE INTERNAL "Rust files to include in the cxx bridge"
)
set(LITERTLM_RUST_BRIDGE_FILES "")

foreach(RS_FILE IN LISTS LITERTLM_RUST_FILES)
    file(RELATIVE_PATH REL_RS "${LITERTLM_PROJECT_ROOT}" "${RS_FILE}")

    list(APPEND LITERTLM_RUST_BRIDGE_FILES "../${REL_RS}")
endforeach()

# --- CORROSION SETUP ---
corrosion_import_crate(MANIFEST_PATH "${LITERTLM_CARGO_TOML}")

if(DEFINED LITERTLM_RUST_LINKER_OVERRIDE)
    if(TARGET litert_lm_deps)
        corrosion_set_linker(litert_lm_deps "${LITERTLM_RUST_LINKER_OVERRIDE}")
        message(STATUS "[LiteRTLM] Hard-wiring litert_lm_deps linker to ${LITERTLM_RUST_LINKER_OVERRIDE}")
    else()
        message(WARNING "[LiteRTLM] Target 'litert_lm_deps' not found. Linker override failed.")
    endif()
endif()

corrosion_add_cxxbridge(
    litertlm_cxx_bridge
    FILES ${LITERTLM_RUST_BRIDGE_FILES}
    CRATE litert_lm_deps
)

set(CORROSION_INC_DIR "${CMAKE_BINARY_DIR}/corrosion_generated/cxxbridge/litertlm_cxx_bridge/include")

foreach(RS_FILE IN LISTS LITERTLM_RUST_FILES)
    get_filename_component(RS_NAME ${RS_FILE} NAME_WE)
    get_filename_component(RS_DIR ${RS_FILE} DIRECTORY)
    file(RELATIVE_PATH REL_PATH "${LITERTLM_PROJECT_ROOT}" "${RS_DIR}")

    set(CORROSION_HEADER "${CORROSION_INC_DIR}/${REL_PATH}/${RS_NAME}.h")
    set(RS_HEADER "${CORROSION_INC_DIR}/${REL_PATH}/${RS_NAME}.rs.h")

    add_custom_command(
        OUTPUT "${RS_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CORROSION_HEADER}" "${RS_HEADER}"
        DEPENDS "${CORROSION_HEADER}"
        COMMENT "Aliasing ${RS_NAME}.h to ${RS_NAME}.rs.h"
    )

    list(APPEND LITERTLM_BRIDGE_ALIASES "${RS_HEADER}")
endforeach()

add_custom_target(litertlm_cxx_bridge_aliases ALL DEPENDS ${LITERTLM_BRIDGE_ALIASES})
add_dependencies(litertlm_cxx_bridge_aliases litertlm_cxx_bridge)
add_custom_command(
    TARGET litertlm_cxx_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
        -P "${LITERTLM_SCRIPTS_DIR}/find_and_copy_cxxbridge.cmake"
    COMMENT "Scanning Cargo build artifacts for libcxxbridge1.a"
)

set(_cxxbridge_paths
    "${CMAKE_BINARY_DIR}/liblitertlm_cxx_bridge.a"
    "${CMAKE_BINARY_DIR}/liblitert_lm_deps.a"
    "${CMAKE_BINARY_DIR}/libcxxbridge1.a"
)

if(NOT TARGET LiteRTLM::CxxBridge::Aggregate)
    message(STATUS "[LiteRTLM] Generating CxxBridge Aggregate...")

    add_library(LiteRTLM::CxxBridge::Aggregate INTERFACE IMPORTED GLOBAL)
    set_target_properties(LiteRTLM::CxxBridge::Aggregate PROPERTIES
        INTERFACE_LINK_LIBRARIES
            "${CMAKE_BINARY_DIR}/liblitertlm_cxx_bridge.a"
            "${CMAKE_BINARY_DIR}/liblitert_lm_deps.a"
            "${CMAKE_BINARY_DIR}/libcxxbridge1.a"
        INTERFACE_INCLUDE_DIRECTORIES
            "${ABSL_INCLUDE_DIR}"
    )
    add_dependencies(LiteRTLM::CxxBridge::Aggregate litertlm_cxx_bridge)
    get_target_property(_CXXBRIDGE_PAYLOAD LiteRTLM::CxxBridge::Aggregate INTERFACE_LINK_LIBRARIES)

    string(REPLACE ";" " " _CXXBRIDGE_LINK_FLAGS "${_CXXBRIDGE_LINK_FLAGS}")

    message(STATUS "[LiteRTLM] CxxBridge Aggregate generated with ${_CXXBRIDGE_PAYLOAD} targets.")
endif()