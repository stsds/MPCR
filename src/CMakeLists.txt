
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

add_subdirectory(data-units)
add_subdirectory(operations)
add_subdirectory(adapters)
# Add MPR library for Testing CPP using all C/C++ Code

set(
        NEW_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/DataTypeModule.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MPCRTileModule.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RcppExports.cpp
        ${SOURCES}
        ${NEW_SOURCES}

)

add_library(mpcr
        SHARED
        ${NEW_SOURCES}
)

# Clang sanitizer linking libraries
if (NOT DEFINED ENV{SAN_LIBS})
    execute_process(COMMAND ${R_ROOT_PATH}/bin/R CMD config SAN_LIBS OUTPUT_VARIABLE CLANG_SANTIZER)
    if ("${CLANG_SANTIZER}" MATCHES "ERROR")
        set(CLANG_SANTIZER "")
        message(STATUS "No Clang-UBSAN")
    else ()
        string(REGEX REPLACE "\n" "" CLANG_SANTIZER "${CLANG_SANTIZER}")
        set(CLANG_SANTIZER "${CLANG_SANTIZER}")
        message("Clang Sanitizer linking flags :  " ${CLANG_SANTIZER})
    endif ()
else ()
    set(CLANG_SANTIZER $ENV{SAN_LIBS})
    message("Clang Sanitizer linking flags ENV :  " ${CLANG_SANTIZER})
endif ()

target_link_libraries(mpcr ${LIBS} ${CLANG_SANTIZER})


install(TARGETS mpcr DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
target_compile_definitions(mpcr PUBLIC CONFIG_FILES_DIR="${CMAKE_CONFIG_PREFIX}")