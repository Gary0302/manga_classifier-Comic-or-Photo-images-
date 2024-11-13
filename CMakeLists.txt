cmake_minimum_required(VERSION 3.10)
project(manga_classifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force ARM64 architecture
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Set build type
set(CMAKE_BUILD_TYPE Release)

# Add Homebrew path
include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add executable
add_executable(manga_classifier manga_classifier.cpp)

# Link libraries
target_link_libraries(manga_classifier PRIVATE ${OpenCV_LIBS})

# Print configuration
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")