set(TESTFILES test_main.cpp)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data-units)



enable_testing()
add_executable(system-tests ${TESTFILES})
target_link_libraries(system-tests mpr)

target_compile_definitions(system-tests PUBLIC TEST_FILES_DIR="${CMAKE_TEST_PREFIX}")
