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



# find_and_copy_bridge.cmake
if(CMAKE_HOST_WIN32)
    # MSVC ABI: the cxx crate's support archive is cxxbridge1.lib. It is also
    # bundled into the Rust static library, so not finding it is not an error.
    file(GLOB_RECURSE BRIDGE_LIB
        "${CMAKE_BINARY_DIR}/cargo/*/cxxbridge1.lib"
        "${CMAKE_BINARY_DIR}/cargo/*/libcxxbridge1.a"
    )
    if(BRIDGE_LIB)
        list(GET BRIDGE_LIB 0 ACTUAL_PATH)
        message(STATUS "Found cxxbridge at: ${ACTUAL_PATH}")
        file(COPY "${ACTUAL_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")
    else()
        message(STATUS "cxxbridge1.lib not found in ${CMAKE_BINARY_DIR}/cargo; using the copy inside the Rust static library")
    endif()
    return()
endif()

file(GLOB_RECURSE BRIDGE_LIB 
    "${CMAKE_BINARY_DIR}/cargo/*/libcxxbridge1.a"
)

if(BRIDGE_LIB)
    # If multiple are found (e.g., debug and release), take the first one
    list(GET BRIDGE_LIB 0 ACTUAL_PATH)
    message(STATUS "Found cxxbridge at: ${ACTUAL_PATH}")
    file(COPY "${ACTUAL_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")
else()
    message(FATAL_ERROR "Could not find libcxxbridge1.a in ${CMAKE_BINARY_DIR}/cargo")
endif()