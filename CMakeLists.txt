
##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################

# Set minimum cmake version
cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    message("-------------------- MPCR 2023-2024 -------------------")
    include(cmake/DetectToolchain.cmake)
endif()

# Project Name and Version
project(MPCR VERSION 1.0.0)


if ("${CMAKE_BUILD_TYPE}" STREQUAL "" OR ${CMAKE_BUILD_TYPE} STREQUAL "NOMODE")
    message(STATUS "WORKING ON NO MODE")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    message(STATUS "WORKING ON RELEASE MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE}")

elseif (${CMAKE_BUILD_TYPE} STREQUAL "DEBUG" OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "WORKING ON DEBUG MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_DEBUG}")
else ()
    message(FATAL_ERROR "Unrecognized build type")
endif ()

include(CheckLanguage)

# Set cmake path module.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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

# This section is to test if there's a CUDA compiler available on the system,
# if so, the code will compile both CPU and GPU backends.
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(USE_CUDA ON)
else()
    message(STATUS "-- No CUDA support --")
endif()

# Find CUDAToolkit and Link to CUSOLVER
if (${USE_CUDA})
    message("-- Build CUDA Support --")
    include(toolchains/CudaToolchain)
    enable_language(CUDA)
    include(ImportCuSolver)
    add_definitions(-DUSE_CUDA)
else ()
    message("-- Build Serial Support -- ")
endif ()

# since the code will compile both CPU and GPU calls, blaspp and lapackpp will be build with no GPU support and all
# the GPU operation would be done directly using cusolver.
set(gpu_backend CACHE "none" FORCE)
include(ImportBlasPP)
include(ImportLapack)
include(ImportLapackPP)

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

install(DIRECTORY inst/include DESTINATION ${CMAKE_INSTALL_PREFIX}/MPCR
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

