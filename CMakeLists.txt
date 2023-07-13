
# Set minimum cmake version
cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)

set(CMAKE_CXX_COMPILER g++)

# Project Name and Version
project(MPR VERSION 1.0.0)


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


# Set C++ standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

if (OpenMP_FOUND)
    message("OpenMp Found")
endif ()

if (${USE_OMP})
    get_directory_property(CMAKE_PARENT_DIR PARENT_DIRECTORY)
    if (NOT CMAKE_PARENT_DIR)
        if (${USE_INTEL} STREQUAL "YES" OR ${USE_INTEL} STREQUAL "ON")
            message(STATUS "Building MMPR using INTEL compiler")

            set(CMAKE_C_COMPILER icc)
            set(CMAKE_CXX_COMPILER icpc)

            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -xHost -qopenmp -g -debug -O3 -fp-model fast=2 -no-prec-div -fma -qopt-assume-safe-padding -qopt-report=4 -fPIC")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -xHost -qopenmp -O3 -fp-model fast=2 -no-prec-div -fma -qopt-assume-safe-padding -fPIC")
            add_definitions(-DUSE_INTEL)
        else ()
            set(CMAKE_C_COMPILER gcc)
            set(CMAKE_CXX_COMPILER g++)

            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp -g -ftree-vectorize -O3 -fopt-info-vec-optimized -fPIC")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -ftree-vectorize -O3 -fopt-info-vec-optimized -fPIC")

            message(STATUS "Building MMPR using GCC/G++ compiler")
        endif ()
    endif ()
endif ()



    # definitions
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr")
endif ()

message("Install Prefix : " ${CMAKE_INSTALL_PREFIX})

# Set cmake path module.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Find R and Rcpp using FindR Module
FIND_PACKAGE(R REQUIRED)
if (${R_FOUND})
    message(STATUS "Using R technology")
    list(APPEND LIBS "R")
    set(USE_R ON)
    add_definitions(-DUSING_R)
endif ()

#adding blas and lapack
#add_definitions(-DMKL_ILP64)
#add_definitions(-DBLAS_ILP64)
#add_definitions(-DLAPACK_ILP64)

include(ImportBlasPP)
include(ImportLapack)
include(ImportLapackPP)


# For Error Handling Class
if (${RUNNING_CPP})
    add_definitions(-DRUNNING_CPP)
endif ()


# Add Include DIRS
include_directories(${CMAKE_SOURCE_DIR}/inst/include)
include_directories(${CMAKE_SOURCE_DIR}/prerequisites)


# Options
option(MPR_BUILD_TESTS "Option to enable building tests" OFF)

# Add src Directory to expose added libraries
add_subdirectory(src)

message("Installation path : ${CMAKE_INSTALL_PREFIX}")
message("Building MPR tests : ${MPR_BUILD_TESTS}")


if (MPR_BUILD_TESTS)
    add_subdirectory(tests/cpp-tests)
    # Add the system test.
    include(CTest)
    add_test(NAME System-Tests
            COMMAND system-tests
            )
endif ()



