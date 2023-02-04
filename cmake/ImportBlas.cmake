# search for BLAS library, if not already included
message("")
message("---------------------------------------- BLAS")
message(STATUS "Checking for BLAS")

include(macros/BuildDependency)

if (NOT TARGET BLAS)

    find_package(BLAS QUIET)

    if (BLAS_FOUND)
        message("   Found BLAS: ${BLAS_LIBRARIES}")
    else ()
        set(build_tests_save "${build_tests}")
        set(build_tests "false")
        BuildDependency(blas "https://github.com/xianyi/OpenBLAS" "v0.3.21")
        set(build_tests "${build_tests_save}")
    endif ()

else ()
    message("   BLAS already included")
endif ()

message(STATUS "BLAS done")
