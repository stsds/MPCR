##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

# Add OpenMP if requested.
option(USE_OPENMP "Use OpenMP, if available" true)
if (NOT USE_OPENMP)
    message(STATUS "User has requested to NOT use OpenMP")
else ()
    if (APPLE)
        include(${PROJECT_SOURCE_DIR}/cmake/ImportOpenMP_MacoOS.cmake)
    else ()
        find_package(OpenMP)
    endif ()

    IF (OPENMP_FOUND)
        message("--------------------------------------------------")
        message("${OpenMP_C_FLAGS}")
        message("${OpenMP_CXX_FLAGS}")
        message("--------------------------------------------------")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(LIBS
                OpenMP::OpenMP_CXX
                ${LIBS}
                )
    ENDIF ()
endif ()
