
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

if (${MPCR_INSTALL})
    set(SOURCES
            ${SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/RLinearAlgebra.cpp
            )
endif ()


set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/RBasicUtilities.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RHelpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RBinaryOperations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RMathematicalOperations.cpp

        ${SOURCES}
        PARENT_SCOPE)
