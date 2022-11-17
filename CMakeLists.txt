cmake_minimum_required(VERSION 3.5)
project(dnn)

set(getdata_inc "${CMAKE_CURRENT_BINARY_DIR}/../utils/getdata.h")
set(getdata_src "${CMAKE_CURRENT_BINARY_DIR}/../utils/getdata.cc")

add_executable(dnn main.cc ${getdata_inc} ${getdata_src} network.h network.cc)