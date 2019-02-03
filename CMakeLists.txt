cmake_minimum_required(VERSION 3.11.4)
project( LINKSVision2019 )

# C++ standards
set(CMAKE_CXX_STANDARD 17)
add_definitions(-O3 -Wall -Wextra -pedantic)

# Libraries
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# C++ files
file(GLOB SOURCES "src/*.cc")
add_executable(LINKSVision2019 ${SOURCES})

# Links Libraries
target_link_libraries( LINKSVision2019 ${OpenCV_LIBS} )
