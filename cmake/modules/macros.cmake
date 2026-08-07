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

macro(import_static_lib target_name lib_full_path)
    if("${lib_full_path}" STREQUAL "")
        message(FATAL_ERROR "Critical Error: Empty path for '${target_name}'")
    endif()

    add_library(${target_name} INTERFACE IMPORTED GLOBAL)
endmacro()

macro(import_absl_lib target_name lib_full_path)
    add_library(${target_name} INTERFACE IMPORTED GLOBAL)
    set_target_properties(${target_name} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ABSL_INCLUDE_DIR}"
    )

    string(REPLACE "imp_" "" clean_name "${target_name}")
    string(REPLACE "absl_" "absl::" ns_path "${clean_name}")

    if(NOT TARGET LiteRTLM::${ns_path})
        add_library(LiteRTLM::${ns_path} ALIAS ${target_name})
    endif()
endmacro()

macro(import_proto_lib target_name lib_path)
    if(NOT TARGET ${target_name})
        add_library(${target_name} INTERFACE IMPORTED GLOBAL)
        set_target_properties(imp_protobuf PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LITERTLM_PROTOBUF_INCLUDE_DIR}")
        add_dependencies(${target_name} protobuf_external)
    endif()
endmacro()

macro(load_package name)
    string(TOUPPER "${name}" upper_name)
    set(USE_SYSTEM_VAR "LITERTLM_USE_SYSTEM_${upper_name}")
    option(${USE_SYSTEM_VAR} "LiteRT-LM: Use system/pre-existing ${name} instead of bundled distribution" OFF)

    set(SHOULD_PROVISION TRUE)

    if(${${USE_SYSTEM_VAR}})
        find_package(${name} QUIET)
        if(TARGET ${name}::${name} OR TARGET ${name})
            message(STATUS "[LiteRTLM] Resolution: Using SYSTEM ${name} (User Override)")
            set(SHOULD_PROVISION FALSE)

            if(TARGET ${name}::${name})
                set(actual_target "${name}::${name}")
            else()
                set(actual_target "${name}")
            endif()

            message(STATUS "[LiteRTLM] Mapping ${name}_external to existing target")
            add_dependencies("${name}_external" ${actual_target})

        else()
            message(FATAL_ERROR "[LiteRTLM] User set ${USE_SYSTEM_VAR}=ON but ${name} was not found in the environment!")
        endif()

    elseif(TARGET ${name}::${name} OR TARGET ${name})
        message(STATUS "[LiteRTLM] Resolution: Using PRE-RESOLVED ${name} (Detected in Namespace)")
        set(SHOULD_PROVISION FALSE)
    endif()

    if(SHOULD_PROVISION)
        message(STATUS "[LiteRTLM] Resolution: Using INTERNAL build for ${name}")
        include("${LITERTLM_CMAKE_PACKAGES_DIR}/${name}/${name}.cmake")
    endif()
    cmake_checkpoint_target("${name}_external" TYPE CUSTOM QUIET)
endmacro()

macro(detect_deps_provider name)
    set(_${name}_user_defined FALSE)
    set(_${name}_predefined FALSE)
    set(_${name}_find_package FALSE)
    set(_${name}_should_provision TRUE)

    string(TOUPPER "${name}" upper_name)
    if(${USE_SYSTEM_VAR} AND DEFINED LITERTLM_SYSTEM_TARGET_${upper_name} AND TARGET ${LITERTLM_SYSTEM_TARGET_${upper_name}})
        set(_${name}_user_defined TRUE)

    elseif(TARGET ${name}::${name})
        set(_${name}_predefined TRUE)

    elseif(${USE_SYSTEM_VAR})
        find_package(${name} QUIET)
        if(${name}_FOUND)
            set(_${name}_find_package TRUE)
        else()
            message(FATAL_ERROR "[LiteRT-LM] User requested system ${name}, but find_package failed.")
        endif()
    endif()
endmacro()

macro(literlm_configure_component_interface prefix main_targets dependency_targets include_dirs)
    set(_target "${prefix}_libs")
    set(_ns_target "LiteRTLM::${prefix}::${prefix}")

    if(NOT TARGET ${_target})
        add_library(${_target} INTERFACE IMPORTED GLOBAL)
    endif()

    if(NOT TARGET ${_ns_target})
        add_library(${_ns_target} ALIAS ${_target})
    endif()

    target_include_directories(${_target} SYSTEM INTERFACE ${include_dirs})

    target_link_libraries(${_target} INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wl,--start-group>
            ${main_targets}
            ${dependency_targets}
            LiteRTLM::absl::absl
            LiteRTLM::protobuf::libprotobuf
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wl,--end-group>
        pthread
    )
endmacro()

macro(add_litertlm_library target_name lib_type)
    file(RELATIVE_PATH _rel_path "${LITERTLM_PROJECT_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}")

    set(_redirected_sources "")

    foreach(_src ${ARGN})
        if(NOT IS_ABSOLUTE "${_src}")
            list(APPEND _redirected_sources "${LITERTLM_GENERATED_SRC_DIR}/${_rel_path}/${_src}")
        else()
            list(APPEND _redirected_sources "${_src}")
        endif()
    endforeach()

    set_source_files_properties(${_redirected_sources} PROPERTIES GENERATED TRUE)

    add_library(${target_name} ${lib_type} ${_redirected_sources})

    if(TARGET generator_complete)
        add_dependencies(${target_name} generator_complete)
    endif()

    if(TARGET litert_external)
        add_dependencies(${target_name} litert_external)
    endif()

    if("${lib_type}" STREQUAL "STATIC")
        set_target_properties(${target_name} PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${LITERTLM_STAGING_DIR}"
        )
        set(_phys_path "${LITERTLM_STAGING_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${target_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")

        set_property(GLOBAL APPEND PROPERTY LITERTLM_ARCHIVE_REGISTRY "${_phys_path}")
        set_property(GLOBAL APPEND PROPERTY LITERTLM_TARGET_REGISTRY "${target_name}")
    endif()
endmacro()

macro(try_generate_aggregate target)
    message(STATUS "[LiteRTLM] Attempting to generate_${target}_aggregate()" )

    set(_config_path "${LITERTLM_CMAKE_PACKAGES_DIR}/${target}/${target}_config.cmake")
    set(_aggregate_path "${LITERTLM_CMAKE_PACKAGES_DIR}/${target}/${target}_aggregate.cmake")
    if(EXISTS ${_aggregate_path})
        include("${_config_path}" OPTIONAL)
        include("${_aggregate_path}" OPTIONAL)        
        cmake_language(EVAL CODE "generate_${target}_aggregate()")
    else()
        message(STATUS "[LiteRTLM] Aggregate filepath not found: ${_aggregate_path}")       
    endif()
endmacro()