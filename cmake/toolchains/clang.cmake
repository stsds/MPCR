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

# - Use Clang compiler
#
# Brief:
#  A modules that sets the variables needed to use the Clang compiler
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
#   USE_CLANG
#   CLANG_VERSION

# Set the compilers to clang for C and C++

set(CMAKE_C_COMPILER "${C_MPCR_COMPILER}" "${C_MPCR_FLAGS}")
set(CMAKE_CXX_COMPILER "${CXX_MPCR_COMPILER}" "${CXX_MPCR_FLAGS}")

add_definitions(-DUSE_CLANG)
set(USE_CLANG ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -march=native -g -O0 -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3 -fPIC")

# Obtain Clang compiler version
execute_process(
        COMMAND ${CMAKE_C_COMPILER} --version
        OUTPUT_VARIABLE CLANG_VERSION_FULL
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract the version number from the full version string
string(REGEX MATCH "clang version ([0-9]+\\.[0-9]+\\.[0-9]+)" _ ${CLANG_VERSION_FULL})
set(CLANG_VERSION ${CMAKE_MATCH_1})

# Optionally, print the extracted version
message(STATUS "Detected Clang version: ${CLANG_VERSION}")
add_definitions(-DCLANG_VERSION=${CLANG_VERSION})

message(STATUS "--------------- Using Clang Compiler -------------- [Version ${CLANG_VERSION}]")