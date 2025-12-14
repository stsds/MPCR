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

# - Use Intel compiler
#
# Brief:
#  A modules that sets the variables needed to use the Intel compiler
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
#   USE_INTEL

set(CMAKE_C_COMPILER "${C_MPCR_COMPILER}" "${C_MPCR_FLAGS}")
set(CMAKE_CXX_COMPILER "${CXX_MPCR_COMPILER}" "${CXX_MPCR_FLAGS}")

add_definitions(-DUSE_INTEL)
set(USE_INTEL ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -xHost -g -debug -O0 -fma -qopt-report=max -fPIC -fp-model precise")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}  -xHost -g -debug -O0 -fma -qopt-report=max -fPIC -fp-model precise")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -xHost -O3 -fma -qopt-report=max -fPIC -fp-model precise")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}  -xHost -O3 -fma -qopt-report=max -fPIC -fp-model precise")

message(STATUS "--------------- Using Intel Compiler -------------")