

add_subdirectory(data-units)
add_subdirectory(operations)
add_subdirectory(adapters)
# Add MPR library for Testing CPP using all C/C++ Code
add_library(mpr
        SHARED
        ${SOURCES}
        )

set(
        NEW_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/DataTypeModule.cpp
        ${SOURCES}
        ${NEW_SOURCES}

)

# Add MPR_RCPP library for R Package Creation
# Linking MPR With additional Files for R Package
add_library(mpr_rcpp
        SHARED
        ${NEW_SOURCES}
        )


target_link_libraries(mpr ${LIBS})
target_link_libraries(mpr_rcpp ${LIBS})


install(TARGETS mpr DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS mpr_rcpp DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)


target_compile_definitions(mpr PUBLIC CONFIG_FILES_DIR="${CMAKE_CONFIG_PREFIX}")
target_compile_definitions(mpr_rcpp PUBLIC CONFIG_FILES_DIR="${CMAKE_CONFIG_PREFIX}")

