
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

if (${BUILD_MPCR_STATIC})
    add_library(mpcr
            STATIC
            ${NEW_SOURCES}
            )
else ()
    add_library(mpcr
            SHARED
            ${NEW_SOURCES}
            )
endif ()

# Ensure BLAS++ and LAPACK++ are built before mpcr when MPCR_INSTALL is ON
if (${MPCR_INSTALL})
    if (TARGET blaspp)
        add_dependencies(mpcr blaspp)
    endif ()
    if (TARGET lapackpp)
        add_dependencies(mpcr lapackpp)
    endif ()
endif ()

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

if (${MPCR_INSTALL})
    if (APPLE)
        target_link_libraries(mpcr ${BLAS_LIBRARIES} ${LIBS} ${CLANG_SANTIZER})
    else ()
        target_link_libraries(mpcr ${BLAS_LIBRARIES} ${LIBS} ${CLANG_SANTIZER} "-Wl,--exclude-libs=ALL")
    endif ()
else ()
    if (APPLE)
        target_link_libraries(mpcr ${LIBS} ${CLANG_SANTIZER})
    else ()
        target_link_libraries(mpcr ${LIBS} ${CLANG_SANTIZER})
    endif ()
endif ()

set_target_properties(mpcr
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${R_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${R_LIB}"
        IMPORTED_LOCATION ${RCPP_LIB}
        )

install(TARGETS mpcr DESTINATION ${CMAKE_INSTALL_PREFIX}/MPCR/lib)
target_compile_definitions(mpcr PUBLIC CONFIG_FILES_DIR="${CMAKE_CONFIG_PREFIX}")

if (EXISTS "${Backend_Support_Status}")
    # Install the dummy file
    install(FILES ${Backend_Support_Status} DESTINATION "${CMAKE_INSTALL_PREFIX}/MPCR/")
endif ()