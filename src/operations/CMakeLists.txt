
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

if(${USE_CUDA})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/concrete)

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/BasicOperations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/BinaryOperations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MathematicalOperations.cpp

        ${SOURCES}
        PARENT_SCOPE)