##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################



function(get_all_environment_variables result)
    if(UNIX)
        set(command "printenv")
    elseif(WIN32)
        set(command "set")
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()

    execute_process(
            COMMAND ${CMAKE_COMMAND} -E env ${command}
            OUTPUT_VARIABLE env_vars
    )
    string(REPLACE "\n" ";" env_vars_list "${env_vars}")
    set(${result} ${env_vars_list} PARENT_SCOPE)
endfunction()



function(find_environment_variable_prefix prefix result)
    get_all_environment_variables(env_vars)
    set(found "FALSE")
    set(count 0)  # Initialize a counter to keep track of matches
    foreach(env_var ${env_vars})
        if(env_var MATCHES "^${prefix}")
            math(EXPR count "${count} + 1")  # Increment the counter
            if(count EQUAL 3)  # Check if at least three matches are found
                set(found "TRUE")
                break()
            endif()
        endif()
    endforeach()
    set(${result} "${found}" PARENT_SCOPE)
endfunction()



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
    set(check_install_var "FALSE")
    find_environment_variable_prefix("_R_CHECK_" check_install_var)

    if (check_install_var STREQUAL "TRUE")
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