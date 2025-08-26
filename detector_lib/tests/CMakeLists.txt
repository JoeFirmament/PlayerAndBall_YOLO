# CMakeLists.txt for pose analysis tests

cmake_minimum_required(VERSION 3.16)

# Find packages
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
    # If GTest is not found, try to find it via pkg-config
    pkg_check_modules(GTEST REQUIRED gtest>=1.10)
    pkg_check_modules(GTEST_MAIN REQUIRED gtest_main>=1.10)
endif()

# Find jsoncpp
pkg_check_modules(JSONCPP REQUIRED jsoncpp>=1.8)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

# Source files for the library being tested
set(POSE_ANALYSIS_SOURCES
    ../src/height_detector.cpp
    ../src/ball_request_detector.cpp
    ../src/id_priority_manager.cpp
    ../src/pose_analyzer.cpp
)

# Test source files
set(TEST_SOURCES
    test_temporal_buffer.cpp
    test_filter_interface.cpp
    test_height_detector.cpp
    # test_ball_request_detector.cpp
    # test_id_priority_manager.cpp
    # test_pose_analyzer.cpp
)

# Create test executable
add_executable(pose_analysis_tests
    ${TEST_SOURCES}
    ${POSE_ANALYSIS_SOURCES}
)

# Link libraries
target_link_libraries(pose_analysis_tests
    ${OpenCV_LIBS}
    ${GTEST_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    pthread
)

# Compiler flags
target_compile_options(pose_analysis_tests PRIVATE
    ${GTEST_CFLAGS_OTHER}
    ${JSONCPP_CFLAGS_OTHER}
    -std=c++17
    -Wall
    -Wextra
    -O2
)

# Enable testing
enable_testing()

# Add test cases
add_test(NAME TemporalBufferTests COMMAND pose_analysis_tests --gtest_filter="TemporalBufferTest.*")
add_test(NAME FilterInterfaceTests COMMAND pose_analysis_tests --gtest_filter="FilterInterfaceTest.*")
add_test(NAME HeightDetectorTests COMMAND pose_analysis_tests --gtest_filter="HeightDetectorTest.*")

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS pose_analysis_tests
    COMMENT "Running all pose analysis tests"
)

# Performance tests (separate executable for long-running tests)
add_executable(pose_analysis_perf_tests
    # performance test files would go here
    ${POSE_ANALYSIS_SOURCES}
)

target_link_libraries(pose_analysis_perf_tests
    ${OpenCV_LIBS}
    ${GTEST_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    pthread
)

target_compile_options(pose_analysis_perf_tests PRIVATE
    ${GTEST_CFLAGS_OTHER}
    ${JSONCPP_CFLAGS_OTHER}
    -std=c++17
    -Wall
    -Wextra
    -O2
)

# Install test executable (optional)
install(TARGETS pose_analysis_tests
    DESTINATION bin/tests
)

# Generate test coverage (if gcov is available)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(pose_analysis_tests PRIVATE --coverage)
    target_link_libraries(pose_analysis_tests gcov)
    
    add_custom_target(coverage
        COMMAND lcov --directory . --capture --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
        COMMAND lcov --list coverage.info
        COMMAND genhtml coverage.info --output-directory coverage_html
        DEPENDS run_tests
        COMMENT "Generating test coverage report"
    )
endif()

# Memory leak detection with valgrind (if available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(memcheck
        COMMAND ${VALGRIND_EXECUTABLE} 
            --tool=memcheck 
            --leak-check=full 
            --show-leak-kinds=all 
            --track-origins=yes 
            --verbose 
            --log-file=valgrind-out.txt 
            $<TARGET_FILE:pose_analysis_tests>
        DEPENDS pose_analysis_tests
        COMMENT "Running memory leak detection with valgrind"
    )
endif()

# Address sanitizer build option
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
if(ENABLE_ASAN)
    target_compile_options(pose_analysis_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_libraries(pose_analysis_tests -fsanitize=address)
endif()

# Thread sanitizer build option  
option(ENABLE_TSAN "Enable Thread Sanitizer" OFF)
if(ENABLE_TSAN)
    target_compile_options(pose_analysis_tests PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_libraries(pose_analysis_tests -fsanitize=thread)
endif()

# Benchmark target (if Google Benchmark is available)
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_executable(pose_analysis_benchmarks
        # benchmark source files would go here
        ${POSE_ANALYSIS_SOURCES}
    )
    
    target_link_libraries(pose_analysis_benchmarks
        benchmark::benchmark
        ${OpenCV_LIBS}
        ${JSONCPP_LIBRARIES}
        pthread
    )
    
    target_compile_options(pose_analysis_benchmarks PRIVATE
        -std=c++17
        -O3
        -DNDEBUG
    )
endif()

# Documentation generation (if Doxygen is available)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(doc_tests
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Print build information
message(STATUS "Pose Analysis Tests Configuration:")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  GTest found: ${GTest_FOUND}")
message(STATUS "  JsonCpp found: ${JSONCPP_FOUND}")
message(STATUS "  Valgrind found: ${VALGRIND_EXECUTABLE}")
message(STATUS "  Address Sanitizer: ${ENABLE_ASAN}")
message(STATUS "  Thread Sanitizer: ${ENABLE_TSAN}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")

# Test data directory
set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_data)
add_definitions(-DTEST_DATA_DIR="${TEST_DATA_DIR}")

# Create test data directory if it doesn't exist
file(MAKE_DIRECTORY ${TEST_DATA_DIR})

# Copy test configuration files
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_config.json
    ${CMAKE_CURRENT_BINARY_DIR}/test_config.json
    COPYONLY
)

# Helper script to run specific test categories
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_test_category.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_test_category.sh
    COPYONLY
)

# Make the script executable
if(UNIX)
    execute_process(
        COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/run_test_category.sh
    )
endif()