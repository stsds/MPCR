# Copyright (C) 2021-2023 by Brightskies inc
#
# This file is part of BS CMake.
#
# BS CMake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BS CMake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GEDLIB. If not, see <http://www.gnu.org/licenses/>.

# - Detect Backend Framework
#
# Brief:
#  This module is used to find the specified technology and use it with the appropriate toolchain if found.
#

if (DEFINED ENV{R_HOME})
    set(R_ROOT_PATH "$ENV{R_HOME}")

else ()
    execute_process(COMMAND R RHOME OUTPUT_VARIABLE R_HOME)
    string(REGEX REPLACE "\n" "" R_HOME "${R_HOME}")
    set(R_ROOT_PATH "${R_HOME}")
endif ()

if (NOT USE_TECH)
    execute_process(COMMAND ${R_ROOT_PATH}/bin/R CMD config CC OUTPUT_VARIABLE USE_TECH)
    string(REGEX REPLACE "\n" "" USE_TECH "${USE_TECH}")
    set(USE_TECH "${USE_TECH}")
    message("C Compiler used for R :  " ${USE_TECH})
endif ()

string(TOLOWER ${USE_TECH} USE_TECH)

if ("${USE_TECH}" STREQUAL "intel" OR "${USE_TECH}" STREQUAL "intel")
    include(toolchains/intel)
    if ("${USE_TECH}" STREQUAL "omp" OR "${USE_TECH}" STREQUAL "cuda")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -qopenmp")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -qopenmp")
    endif ()
elseif ("${USE_TECH}" STREQUAL "clang")
    include(toolchains/clang)
    if ("${USE_TECH}" STREQUAL "omp" OR "${USE_TECH}" STREQUAL "cuda")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    endif ()
else ()
    include(toolchains/gnu)
    if ("${USE_TECH}" STREQUAL "omp" OR "${USE_TECH}" STREQUAL "cuda")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    endif ()
endif ()


if ("${USE_TECH}" STREQUAL "omp" OR "${USE_TECH}" STREQUAL "cuda")
    add_definitions(-DUSE_OMP)
    set(USE_OMP ON)
    message("STATUS OpenMP is enabled")
endif ()

if ("${USE_TECH}" STREQUAL "cuda")
    include(toolchains/cuda)
endif ()

