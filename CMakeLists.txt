cmake_minimum_required(VERSION 3.5)
project(track)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

file(GLOB SRC
        "src/*.h"
        "src/*.cpp"
        "src/*.hpp"
        )

add_executable(track ${SRC})

target_link_libraries(track ${OpenCV_LIBS} )