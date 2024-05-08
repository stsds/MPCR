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

# - Use cuda compiler
#
# Brief:
#  A modules that sets the variables needed to use the cuda compiler
#
# Usage:
#  It sets the following variables:
#   CMAKE_CUDA_FLAGS
#   CMAKE_CUDA_STANDARD
#   CMAKE_CUDA_STANDARD_REQUIRED
#  It adds the following definitions to compilation:
#   USE_CUDA

enable_language("CUDA")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_definitions(-DUSE_CUDA)
set(USE_CUDA ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(PROPERTIES CUDA_ARCHITECTURES "35;50;72")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=true -ftz=true -prec-div=false -prec-sqrt=false")
message(STATUS "Using CUDA Compiler")

find_package(CUDAToolkit REQUIRED)
