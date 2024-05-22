# Copyright (C) 2021-2024 by Brightskies inc
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

# - Use GNU compiler
#
# Brief:
#  A modules that sets the variables needed to use the GNU compiler
#
# Usage:
#  It sets the following variables:
#   CMAKE_C_COMPILER
#   CMAKE_CXX_COMPILER
#   CMAKE_CXX_FLAGS_DEBUG
#   CMAKE_CXX_FLAGS_RELEASE
#   CMAKE_CXX_STANDARD
#   CMAKE_CXX_STANDARD_REQUIRED
#   CMAKE_CXX_EXTENSIONS
#  It adds the following definitions to compilation:
#   USE_GNU
#   GCC_VERSION


#execute_process(COMMAND ${R_ROOT_PATH}/bin/R CMD config CC OUTPUT_VARIABLE C_COMPILER_FLAG)
#string(REGEX REPLACE "\n" "" USE_TECH "${C_COMPILER_FLAG}")
#set(C_COMPILER_FLAG "${C_COMPILER_FLAG}")
#
#execute_process(COMMAND ${R_ROOT_PATH}/bin/R CMD config CXX OUTPUT_VARIABLE CXX_COMPILER_FLAG)
#string(REGEX REPLACE "\n" "" USE_TECH "${CXX_COMPILER_FLAG}")
#set(CXX_COMPILER_FLAG "${CXX_COMPILER_FLAG}")


message("----------------------------------------------------------------------")
message(${C_MPCR_COMPILER})
message(${CXX_MPCR_COMPILER})
message("----------------------------------------------------------------------")
set(CMAKE_C_COMPILER ${C_MPCR_COMPILER})
set(CMAKE_CXX_COMPILER ${CXX_MPCR_COMPILER})

#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CXX_COMPILER g++ -std=gnu++11)


add_definitions(-DUSE_GNU)
set(USE_GNU ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -march=native -ftree-vectorize -O0 -fopt-info-vec-optimized -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ftree-vectorize -O3 -fopt-info-vec-optimized -fPIC")

# Obtain GCC compiler version
execute_process(
        COMMAND ${CMAKE_C_COMPILER} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-DGCC_VERSION=${GCC_VERSION})

message(STATUS "--------------- Using GCC Compiler ------------- - [Version ${GCC_VERSION}]")