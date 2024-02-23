
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################
if (${MPCR_INSTALL})
    set(SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/TileLinearAlgebra.cpp
            ${SOURCES}
            )
endif ()

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/BasicOperations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/BinaryOperations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MathematicalOperations.cpp


        ${SOURCES}
        PARENT_SCOPE)