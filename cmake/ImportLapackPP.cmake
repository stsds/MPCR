##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

# search for LAPACK library, if not already included
message("")
message("---------------------------------------- LAPACK++")
message(STATUS "Checking for LAPACK++")
if (NOT TARGET lapackpp)
    include(ImportLapack)

    set(build_tests_save "${build_tests}")
    set(build_tests "false")

    set(url "https://github.com/icl-utk-edu/lapackpp")
    set(tag "v2023.06.00")
    message(STATUS "Fetching LAPACK++ ${tag} from ${url}")
    include(FetchContent)
    FetchContent_Declare(
            lapackpp GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
    FetchContent_MakeAvailable(lapackpp)

    set(build_tests "${build_tests_save}")

else ()
    message("   LAPACK++ already included")
endif ()

# Add to linking libs.
set(LIBS
        lapackpp
        ${LIBS}
        )

# Add definition indicating version.
if ("${lapackpp_defines}" MATCHES "LAPACK_ILP64")
    set(COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS} -DHCORE_HAVE_LAPACK_WITH_ILP64")
endif ()

message(STATUS "LAPACK++ done")
