##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

# search for LAPACK library, if not already included
message("")
message("---------------------------------------- LAPACK")
message(STATUS "Checking for LAPACK")

include(macros/BuildDependency)

if (NOT TARGET LAPACK)
    find_package(LAPACK QUIET)
    if (LAPACK_FOUND)
        message("   Found LAPACK: ${LAPACK_LIBRARIES}")
    else ()
        message(WARNING
                "A complete LAPACK implementation was not found. MPCR requires "
                "single-precision complex LAPACK routines such as clatms. "
                "On macOS, Apple Accelerate does not provide all required routines; "
                "building the bundled OpenBLAS fallback.")
        set(build_tests_save "${build_tests}")
        set(build_tests "false")
        BuildDependency(blas "https://github.com/xianyi/OpenBLAS" "6287a23")
        set(build_tests "${build_tests_save}")
        find_package(LAPACK REQUIRED)
    endif ()
else ()
    message("   LAPACK already included")
endif ()

message(STATUS "LAPACK done")
