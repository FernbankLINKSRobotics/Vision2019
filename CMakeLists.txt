cmake_minimum_required(VERSION 3.11.4)
project( LINKSVision2019 )

# C++ standards
set(CMAKE_CXX_STANDARD 17)
add_definitions(-O3 -Wall -Wextra -pedantic -DBOOST_ERROR_CODE_HEADER_ONLY )

# Libraries
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /usr/local/include )
include_directories( /usr/local/lib )
find_package( OpenCV REQUIRED )
#find_package( cscore REQUIRED )

### BOOST and its craziness
find_package(Boost)
if(Boost_FOUND)
    set(Boost_USE_MULTITHREADED true)
    set(Boost_USE_STATIC_LIBS true)
    message(STATUS "Boost is ${BOOST_ROOT}")
    message(STATUS "Boost directories found at ${Boost_INCLUDE_DIR}")
    message(STATUS "Boost libraries found at ${Boost_LIBRARY_DIR}")
    set(Boost_LIBRARY_DIR "${BOOST_ROOT}/stage64/lib")
    include_directories(${Boost_INCLUDE_DIR})
elseif()
    message("Boost NOT Found!")
endif()

# Headers
include_directories(include)

# C++ files
file(GLOB SOURCES "src/*.cc")
add_executable(LINKSVision2019 ${SOURCES})

# Links Libraries
target_link_libraries( LINKSVision2019 ${OpenCV_LIBS} )
target_link_libraries( LINKSVision2019 cscore )
target_link_libraries( LINKSVision2019 ntcore )
#target_link_libraries( LINKSVision2019 harfbuzz )
