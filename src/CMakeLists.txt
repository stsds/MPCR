
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

target_link_libraries(mpcr ${BLAS_LIBRARIES} ${LIBS}  "-Wl,--exclude-libs=ALL" )
set_target_properties(mpcr
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${R_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${R_LIB}"
        IMPORTED_LOCATION ${RCPP_LIB}
        )


install(TARGETS mpcr DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
target_compile_definitions(mpcr PUBLIC CONFIG_FILES_DIR="${CMAKE_CONFIG_PREFIX}")