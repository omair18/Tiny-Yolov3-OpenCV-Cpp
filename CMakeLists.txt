cmake_minimum_required(VERSION 2.8)
project( Vid2Img )
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "-g -std=c++11")

add_executable( detector object_detection_yolo.cpp )
target_link_libraries(detector  ${OpenCV_LIBS} )

