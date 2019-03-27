cmake_minimum_required(VERSION 3.7)
project( LINKSVision2019 )

# C++ standards
set(CMAKE_CXX_STANDARD 14)
add_definitions(-O3 -Wall -Wextra -pedantic )

# Libraries
include_directories( /usr/local/frc/include/opencv2 )
include_directories( /usr/local/frc/include )
include_directories( /usr/local/frc/lib )
link_directories( /usr/local/frc/lib )

# Headers
include_directories(include)

# C++ files
file(GLOB SOURCES "src/*.cc")
add_executable(LINKSVision2019 ${SOURCES})

# Links Libraries
target_link_libraries( LINKSVision2019 opencv_dnn 
                                       opencv_ml 
                                       opencv_objdetect 
                                       opencv_shape 
                                       opencv_stitching 
                                       opencv_superres 
                                       opencv_videostab
                                       opencv_calib3d 
                                       opencv_features2d 
                                       opencv_highgui 
                                       opencv_videoio 
                                       opencv_imgcodecs 
                                       opencv_video 
                                       opencv_photo
                                       opencv_imgproc 
                                       opencv_flann 
                                       opencv_core ) 

target_link_libraries( LINKSVision2019 pthread 
                                       cscore 
                                       ntcore
                                       wpilibc
                                       cameraserver )
