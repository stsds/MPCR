
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/Promoter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ContextManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MemoryHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RunContext.cpp

        ${SOURCES}
        PARENT_SCOPE)