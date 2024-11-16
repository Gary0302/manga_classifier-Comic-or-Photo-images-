cmake_minimum_required(VERSION 3.10)

project(manga_classifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenCV
set(OpenCV_DIR "C:\\Users\\black\\opencv\\build")   
find_package(OpenCV REQUIRED)

# Check if OpenCV was found
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Please ensure OpenCV is installed and its paths are configured.")
endif()

# Add executable
add_executable(manga_classifier manga_classifier.cpp)

# Link libraries
target_include_directories(manga_classifier PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(manga_classifier PRIVATE ${OpenCV_LIBS})

# Print configuration
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
