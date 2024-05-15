cmake_minimum_required(VERSION 3.28)
project(gstreamer_filter_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(gstreamer_filter_project main.cpp
        image.hh
        utils.hh)
