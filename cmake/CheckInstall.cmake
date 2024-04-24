##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################


function(check_install_dir original)
    string(LENGTH "${original}" len)
    set(result_variable_temp "")

    foreach(i RANGE 0 ${len})
        math(EXPR index "${len} - ${i}")
        string(SUBSTRING "${original}" ${index} 1 char)
        set(result_variable_temp "${result_variable_temp}${char}")
    endforeach()

    set(ENV{MPCR_INSTALL}  ${result_variable_temp})
endfunction()

function(check_install path_to_check result_variable)
    if (DEFINED ENV{_R_CHECK_NATIVE_ROUTINE_REGISTRATION_} OR DEFINED ENV{_R_CHECK_R_ON_PATH_}
            OR DEFINED ENV{_R_CHECK_S3_METHODS_NOT_REGISTERED_})
        set(temp_install TRUE)
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E remove "${PROJECT_SOURCE_DIR}/R/MPCR.R"
                RESULT_VARIABLE delete_result
        )
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/src/R/MPCR.R" "${PROJECT_SOURCE_DIR}/R/"
                RESULT_VARIABLE move_result
        )
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_SOURCE_DIR}/src/R/"
                RESULT_VARIABLE delete_result
        )

    else ()
        message("Inside check directories")
        set(temp_install FALSE)
        string(REPLACE "/" ";" path_parts "${path_to_check}")

        foreach(part IN LISTS path_parts)
            check_install_dir("${part}" dir_name_temp)
            set(dir_temp "kcehcR.RCPM")

            if("$ENV{MPCR_INSTALL}" STREQUAL ${dir_temp})
                # Set the variable to true
                set(temp_install TRUE)
                execute_process(
                        COMMAND ${CMAKE_COMMAND} -E remove "${PROJECT_SOURCE_DIR}/R/MPCR.R"
                        RESULT_VARIABLE delete_result
                )
                execute_process(
                        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/src/R/MPCR.R" "${PROJECT_SOURCE_DIR}/R/"
                        RESULT_VARIABLE move_result
                )
                execute_process(
                        COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_SOURCE_DIR}/src/R/"
                        RESULT_VARIABLE delete_result
                )
                break()
            endif()
        endforeach()
    endif()

    # Pass the result back to the caller
    set(${result_variable} ${temp_install} PARENT_SCOPE)
endfunction()