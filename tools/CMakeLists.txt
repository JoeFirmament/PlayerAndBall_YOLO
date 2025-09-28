# CMakeLists.txt for Camera Calibration C++ Tools
cmake_minimum_required(VERSION 3.10)
project(CameraCalibrationTools)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV 4.x or higher.")
endif()

# Find Eigen (optional, for advanced matrix operations)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found - enabling advanced matrix operations")
    add_definitions(-DHAVE_EIGEN)
else()
    message(STATUS "Eigen3 not found - using basic matrix operations")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Source files
set(SOURCES
    cpp_calibration_example.cpp
    cpp_calibration_loader.cpp
)

# Headers
set(HEADERS
    cpp_calibration_loader.h
)

# Create executable
add_executable(calibration_example ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(calibration_example ${OpenCV_LIBS})

# Compiler flags
if(MSVC)
    target_compile_options(calibration_example PRIVATE /W4)
else()
    target_compile_options(calibration_example PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation
install(TARGETS calibration_example
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "=== Camera Calibration Tools Configuration ===")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
if(Eigen3_FOUND)
    message(STATUS "Eigen3: Enabled")
else()
    message(STATUS "Eigen3: Not found (optional)")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "============================================")

# Add test target (optional)
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Add documentation target (optional)
find_package(Doxygen)
if(Doxygen_FOUND)
    add_subdirectory(docs)
endif()
