
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

set(TESTFILES

        ${CMAKE_CURRENT_SOURCE_DIR}/TestPrecision.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestPromoter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestMemoryHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestRunContext.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestContextManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestCudaStreams.cpp

        ${TESTFILES}
        PARENT_SCOPE
        )