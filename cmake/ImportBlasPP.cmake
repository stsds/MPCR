##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

# search for BLAS library, if not already included
message("")
message("---------------------------------------- BLAS++")
message(STATUS "Checking for BLAS++")

if (NOT TARGET blaspp)

    include(ImportBlas)

    set(build_tests_save "${build_tests}")
    set(build_tests "false")
    set(url "https://github.com/icl-utk-edu/blaspp")
    set(tag "v2023.06.00")
    message(STATUS "Fetching BLAS++ ${tag} from ${url}")
    include(FetchContent)
    FetchContent_Declare(
            blaspp GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
    FetchContent_MakeAvailable(blaspp)
    set(build_tests "${build_tests_save}")

else ()
    message("   BLAS++ already included")
endif ()

set(LIBS_LINEAR
        blaspp
        ${LIBS_LINEAR}
        ${LIBS}
        )

message(STATUS "BLAS++ done")
