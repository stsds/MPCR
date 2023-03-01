
# - Find the R and Rcpp library
#
# Usage:
#   find_package(R [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   R _FOUND               ... true if R and Rcpp is found on the system
#   R_LIBRARIES            ... full path to R and Rcpp library
#   R_INCLUDE_DIRS         ... R and Rcpp include directory
#
# The following variables will be checked by the function
#   R_ROOT_PATH          ... if set, the R libraries are exclusively searched
#                               under this path
#   R_LIB_PATH          ... if set, the Rcpp libraries are exclusively searched
#                              under this path


if (NOT R_ROOT_PATH)
    execute_process(COMMAND R RHOME OUTPUT_VARIABLE R_HOME)
    string(REGEX REPLACE "\n" "" R_HOME "${R_HOME}")
    set(R_ROOT_PATH "${R_HOME}")
    message("R Home Path :  " ${R_ROOT_PATH})
endif ()


if (NOT R_LIB_PATH)
    if($ENV{R_LIB_PATH})
        set(R_LIB_PATH $ENV{R_LIB_PATH})
        else()
            execute_process(COMMAND Rscript ${CMAKE_MODULE_PATH}/FindRLibraryPath.R OUTPUT_VARIABLE R_LIB_PATH)
            set(R_LIB_PATH ${R_LIB_PATH})
            message("R Lib Path :  " ${R_LIB_PATH})
        endif()
endif ()


if (R_ROOT_PATH)

    #find libs
    find_library(
            R_LIB
            REQUIRED
            NAMES "libR.so"
            PATHS ${R_ROOT_PATH}
            PATH_SUFFIXES "lib" "lib64" "bin"
            NO_DEFAULT_PATH
    )

    # find includes
    find_path(
            R_INCLUDE_DIRS
            REQUIRED
            NAMES "R.h"
            PATHS ${R_ROOT_PATH}
            PATH_SUFFIXES "include"
            NO_DEFAULT_PATH
    )


else ()

    #find libs
    find_library(
            R_LIB
            REQUIRED
            NAMES "libR.so"
            PATHS ${LIB_INSTALL_DIR}
    )


    #find includes
    find_path(
            R_INCLUDE_DIRS
            REQUIRED
            NAMES "R.h"
            PATHS ${INCLUDE_INSTALL_DIR}
    )

endif (R_ROOT_PATH)


if (R_LIB_PATH)

    #find libs
    find_library(
            RCPP_LIB
            REQUIRED
            NAMES "Rcpp.so"
            PATHS ${R_LIB_PATH}
            PATH_SUFFIXES "Rcpp/libs" "Rcpp/lib64" "Rcpp/bin"
            NO_DEFAULT_PATH
    )

    # find includes
    find_path(
            RCPP_INCLUDE_DIRS
            REQUIRED
            NAMES "Rcpp.h"
            PATHS ${R_LIB_PATH}
            PATH_SUFFIXES "Rcpp/include"
            NO_DEFAULT_PATH
    )


else ()

    #find libs
    find_library(
            RCPP_LIB
            REQUIRED
            NAMES "Rcpp.so"
            PATHS ${LIB_INSTALL_DIR}
    )

    #find includes
    find_path(
            RCPP_INCLUDE_DIRS
            REQUIRED
            NAMES "Rcpp.h"
            PATHS ${INCLUDE_INSTALL_DIR}
    )


endif (R_LIB_PATH)

set(R_LIBRARIES
        ${R_LIBRARIES}
        ${R_LIB}
        )


set(R_INCLUDE
        ${R_INCLUDE}
        ${R_INCLUDE_DIRS}
        ${RCPP_INCLUDE_DIRS}
        )

add_library(R INTERFACE IMPORTED)
set_target_properties(R
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${R_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "${R_LIBRARIES}"
        IMPORTED_LOCATION ${RCPP_LIB}
        )


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(R DEFAULT_MSG
        R_INCLUDE R_LIBRARIES)

include_directories(${R_INCLUDE})
mark_as_advanced(R_INCLUDE R_INCLUDE_DIRS RCPP_INCLUDE_DIRS R_LIBRARIES R_LIB RCPP_LIB)
