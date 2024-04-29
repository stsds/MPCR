##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

# Set minimum cmake version
cmake_minimum_required(VERSION 3.19.0)
cmake_policy(SET CMP0048 NEW)

include(cmake/DetectToolchain.cmake)

# Project Name and Version
project(MPCR VERSION 1.0.0)


if ("${CMAKE_BUILD_TYPE}" STREQUAL "" OR ${CMAKE_BUILD_TYPE} STREQUAL "NOMODE")
    message(STATUS "WORKING ON NO MODE")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    message(STATUS "WORKING ON RELEASE MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "DEBUG" OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "WORKING ON DEBUG MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
else ()
    message(FATAL_ERROR "Unrecognized build type")
endif ()

# Set cmake path module.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(cmake/CheckInstall.cmake)
set(mpcr_path_to_check ${CMAKE_BINARY_DIR})
check_install(${mpcr_path_to_check} mpcr_result)

if (${mpcr_result})
    message("MPCR Install Result : ${mpcr_result}")
    set(MPCR_INSTALL OFF)
else ()
    add_definitions(-DMPCR_INSTALL)
    set(MPCR_INSTALL ON)
    message("MPCR Install Result : ${mpcr_result}")
endif ()

if (NOT APPLE)
    include(ImportGFortran)
endif ()

include(ImportOpenMP)
if (OpenMP_FOUND)
    message("OpenMp Found")
endif ()

if (${BUILD_SHARED_LIBS})
    set(BLA_STATIC OFF)
else ()
    set(BLA_STATIC ON)
endif ()


# Find R and Rcpp using FindR Module
FIND_PACKAGE(R REQUIRED)
if (${R_FOUND})
    message(STATUS "Using R technology")
    list(APPEND LIBS "R")
    set(USE_R ON)
    add_definitions(-DUSING_R)
endif ()


# blaspp and lapackpp will be build with no GPU support.
set(gpu_backend CACHE "none" FORCE)

add_definitions(-DBLAS_ERROR_NDEBUG)
add_definitions(-DNDEBUG)
include(ImportBlasPP)
include(ImportLapack)
include(ImportLapackPP)

if (${MPCR_INSTALL})
    set(LIBS
            ${LIBS_LINEAR}
            ${LIBS}
            )
endif ()


# definitions
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr")
endif ()

message("Install Prefix : " ${CMAKE_INSTALL_PREFIX})

# For Error Handling Class
if (${RUNNING_CPP})
    add_definitions(-DRUNNING_CPP)
endif ()


# Add Include DIRS
include_directories(${CMAKE_SOURCE_DIR}/inst/include)


# Options
option(MPCR_BUILD_TESTS "Option to enable building tests" OFF)

# Add src Directory to expose added libraries
add_subdirectory(src)

message("Installation path : ${CMAKE_INSTALL_PREFIX}")
message("Building MPCR tests : ${MPCR_BUILD_TESTS}")


if (MPCR_BUILD_TESTS)
    include_directories(${CMAKE_SOURCE_DIR}/prerequisites)
    add_subdirectory(tests/cpp-tests)
    # Add the system test.
    include(CTest)
    add_test(NAME System-Tests
            COMMAND system-tests
            )
endif ()
