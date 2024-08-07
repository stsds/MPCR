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

# Define the function SplitString
function(SplitString INPUT_STRING FIRST_PART_VAR REST_PARTS_VAR)
    # Initialize an empty list to hold the parts
    set(PARTS)

    # Initialize the remaining string to process
    set(REMAINING_STRING "${INPUT_STRING}")

    # Find the position of the first space
    string(FIND "${REMAINING_STRING}" " " SPACE_POSITION)

    if (SPACE_POSITION GREATER -1)
        # Extract the first part before the space
        string(SUBSTRING "${REMAINING_STRING}" 0 ${SPACE_POSITION} FIRST_PART)

        # Strip any leading or trailing whitespace
        string(STRIP "${FIRST_PART}" FIRST_PART)

        # Update the remaining string (skip the space)
        math(EXPR START_POS "${SPACE_POSITION} + 1")
        string(SUBSTRING "${REMAINING_STRING}" ${START_POS} -1 REMAINING_STRING)
    else ()
        # If no space was found, the whole string is the first part
        set(FIRST_PART "${REMAINING_STRING}")
        set(REMAINING_STRING "")
    endif ()

    # Loop until there are no more spaces in the remaining string
    while ("${REMAINING_STRING}" MATCHES " ")
        # Find the position of the next space
        string(FIND "${REMAINING_STRING}" " " SPACE_POSITION)

        # Extract the part before the space
        string(SUBSTRING "${REMAINING_STRING}" 0 ${SPACE_POSITION} PART)

        # Strip any leading or trailing whitespace
        string(STRIP "${PART}" PART)

        # Add the part to the list
        list(APPEND PARTS "${PART}")

        # Update the remaining string (skip the space)
        math(EXPR START_POS "${SPACE_POSITION} + 1")
        string(SUBSTRING "${REMAINING_STRING}" ${START_POS} -1 REMAINING_STRING)
    endwhile ()

    # Add the last part (if any) to the list
    string(STRIP "${REMAINING_STRING}" REMAINING_STRING)
    if (NOT "${REMAINING_STRING}" STREQUAL "")
        list(APPEND PARTS "${REMAINING_STRING}")
    endif ()

    # Set the output variables
    set(${FIRST_PART_VAR} "${FIRST_PART}" PARENT_SCOPE)
    set(${REST_PARTS_VAR} "${PARTS}" PARENT_SCOPE)
endfunction()


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


execute_process(COMMAND ${R_ROOT_PATH}/bin/R CMD config CC OUTPUT_VARIABLE R_C_MPCR_COMPILER)
string(REGEX REPLACE "\n" "" R_C_MPCR_COMPILER "${R_C_MPCR_COMPILER}")
set(R_C_MPCR_COMPILER "${R_C_MPCR_COMPILER}")

execute_process(COMMAND ${R_ROOT_PATH}/bin/R CMD config CXX OUTPUT_VARIABLE R_CXX_MPCR_COMPILER)
string(REGEX REPLACE "\n" "" R_CXX_MPCR_COMPILER "${R_CXX_MPCR_COMPILER}")
set(R_CXX_MPCR_COMPILER "${R_CXX_MPCR_COMPILER}")


SplitString("${R_C_MPCR_COMPILER}" C_MPCR_COMPILER C_MPCR_FLAGS)
SplitString("${R_CXX_MPCR_COMPILER}" CXX_MPCR_COMPILER CXX_MPCR_FLAGS)

unset("${C_MPCR_FLAGS}")
unset("${CXX_MPCR_FLAGS}")


string(TOLOWER ${USE_TECH} USE_TECH)

if ("${USE_TECH}" MATCHES "intel" OR "${USE_TECH}" MATCHES "icc" OR "${USE_TECH}" MATCHES "icx")
    include(${CMAKE_SOURCE_DIR}/cmake/toolchains/intel.cmake)
elseif ("${USE_TECH}" MATCHES "clang")
    include(${CMAKE_SOURCE_DIR}/cmake/toolchains/clang.cmake)
else ()
    include(${CMAKE_SOURCE_DIR}/cmake/toolchains/gnu.cmake)
endif ()

