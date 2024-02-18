
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

if (USE_CUDA)
    set(SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/GPULinearAlgebra.cpp
            ${SOURCES})
endif ()


set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/CPULinearAlgebra.cpp
        ${SOURCES}
        PARENT_SCOPE)

