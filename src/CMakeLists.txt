
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

add_subdirectory(data-units)
add_subdirectory(operations)
add_subdirectory(adapters)
add_subdirectory(kernels)

set(
        NEW_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/DataTypeModule.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RcppExports.cpp
        ${SOURCES}
        ${NEW_SOURCES}

)

if(${BUILD_MPCR_STATIC})
add_library(mpcr
        STATIC
        ${NEW_SOURCES}
        )
else()
    add_library(mpcr
            SHARED
            ${NEW_SOURCES}
            )
    endif()

target_link_libraries(mpcr ${LIBS})


install(TARGETS mpcr DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
target_compile_definitions(mpcr PUBLIC CONFIG_FILES_DIR="${CMAKE_CONFIG_PREFIX}")