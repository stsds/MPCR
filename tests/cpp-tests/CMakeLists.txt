
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

set(TESTFILES test_main.cpp)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data-units)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utilities)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operations)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels)


enable_testing()
add_executable(system-tests ${TESTFILES})
target_link_libraries(system-tests mpcr)

target_compile_definitions(system-tests PUBLIC TEST_FILES_DIR="${CMAKE_TEST_PREFIX}")
