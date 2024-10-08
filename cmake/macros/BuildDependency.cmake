macro(BuildDependency raw_name url tag)
    string(TOLOWER ${raw_name} name)
    string(TOUPPER ${raw_name} capital_name)
    message(STATUS "Fetching ${name} ${tag} from ${url}")
    include(FetchContent)
    FetchContent_Declare(${name} GIT_REPOSITORY "${url}" GIT_TAG "${tag}")
    FetchContent_Populate(${name})
    set(${name}_srcpath ${CMAKE_BINARY_DIR}/_deps/${name}-src)
    set(${name}_binpath ${CMAKE_BINARY_DIR}/_deps/${name}-bin)
    set(${name}_installpath ${CMAKE_BINARY_DIR}/_deps/${name}-install)
    file(MAKE_DIRECTORY ${${name}_binpath})
    file(MAKE_DIRECTORY ${${name}_installpath})

    # Configure subproject into <subproject-build-dir>
    execute_process(COMMAND ${CMAKE_COMMAND}
            -DCMAKE_INSTALL_PREFIX=${${name}_installpath}
            -DNOFORTRAN=1
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS}
            ${${name}_srcpath}
            WORKING_DIRECTORY
            ${${name}_binpath})

    # Build and install subproject
    include(ProcessorCount)
    ProcessorCount(N)
    set(N 2)
    if (NOT N EQUAL 0)
        execute_process(COMMAND ${CMAKE_COMMAND} --build ${${name}_binpath} --parallel ${N} --target install ERROR_FILE /dev/null OUTPUT_QUIET)
    else ()
        execute_process(COMMAND ${CMAKE_COMMAND} --build ${${name}_binpath} --parallel 48 --target install ERROR_FILE /dev/null OUTPUT_QUIET)
    endif ()

    set(ENV{LD_LIBRARY_PATH} "${${name}_installpath}/lib:${${name}_installpath}/lib64:$ENV{LD_LIBRARY_PATH}")
    set(ENV{LIBRARY_PATH} "${${name}_installpath}/lib:${${name}_installpath}/lib64:$ENV{LIBRARY_PATH}")
    set(ENV{CPATH} "${${name}_installpath}/include:$ENV{CPATH}")
    set(ENV{PKG_CONFIG_PATH} "${${name}_installpath}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    set(${capital_name}_DIR "${${name}_installpath}/lib")

    include_directories(${${name}_installpath}/include)
    link_directories(${${name}_installpath}/lib)
    link_directories(${${name}_installpath}/lib64)

    install(
            DIRECTORY
            "${${name}_installpath}/lib"
            DESTINATION
            ./
            PATTERN "*"
            PERMISSIONS
            GROUP_READ GROUP_WRITE GROUP_EXECUTE

    )
    install(
            DIRECTORY
            "${${name}_installpath}/include"
            DESTINATION
            ./
            PATTERN "*"
            PERMISSIONS
            GROUP_READ GROUP_WRITE GROUP_EXECUTE
    )
    install(
            DIRECTORY
            "${${name}_installpath}/share"
            DESTINATION
            ./
            PATTERN "*"
            PERMISSIONS
            GROUP_READ GROUP_WRITE GROUP_EXECUTE
    )

endmacro()
