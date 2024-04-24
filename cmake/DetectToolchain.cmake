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
set(USE_TECH "omp" CACHE STRING "Specify the technology to use(cuda,sycl,omp_offload), default will fallback to omp")
option(USE_INTEL "Use intel compilers as base compilers if possible" OFF)

string(TOLOWER ${USE_TECH} USE_TECH)

if ("${USE_TECH}" STREQUAL "intel")
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

