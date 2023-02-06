
# Set minimum cmake version
cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)

# Project Name and Version
project(MPR VERSION 1.0.0)


# Set C++ standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


message(${CMAKE_INSTALL_PREFIX})

# definitions
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(${CMAKE_INSTALL_PREFIX})
    set(CMAKE_INSTALL_PREFIX "/usr")
endif ()


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
add_definitions(-DMKL_ILP64)
add_definitions(-DBLAS_ILP64)
add_definitions(-DLAPACK_ILP64)
include(ImportBlasPP)
include(ImportLapack)
include(ImportLapackPP)



# For Error Handling Class
if(${RUNNING_CPP})
    add_definitions(-DRUNNING_CPP)
endif()


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



