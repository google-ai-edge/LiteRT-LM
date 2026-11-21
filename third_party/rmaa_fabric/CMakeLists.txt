# Copyright 2026 RMAA Holdings & The ODML Authors.

cmake_minimum_required(VERSION 3.20)
project(litert_lm_rmaa_fabric CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LITERT_LM_ENABLE_RMAA_FABRIC "Enable RMAA Execution and Memory Fabric for LiteRT-LM" ON)

if (LITERT_LM_ENABLE_RMAA_FABRIC)
    add_library(litert_lm_rmaa_fabric STATIC
        src/litert_lm_rmaa_fabric.cc
    )

    target_include_directories(litert_lm_rmaa_fabric PUBLIC
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../c
    )

    target_compile_definitions(litert_lm_rmaa_fabric PUBLIC
        LITERT_LM_ENABLE_RMAA_FABRIC
    )

    if (BUILD_TESTING)
        add_executable(test_rmaa_integration tests/test_rmaa_integration.cc)
        target_link_libraries(test_rmaa_integration PRIVATE litert_lm_rmaa_fabric)
    endif()
endif()
