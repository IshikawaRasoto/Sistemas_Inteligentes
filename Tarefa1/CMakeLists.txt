cmake_minimum_required(VERSION 3.15)
project(saleman)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE) # don't build the supplied example games

FetchContent_Declare(
        raylib
        GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
        GIT_TAG 5.5
        GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(raylib)

add_executable(saleman main.cpp
     map.h
     genetic.h
     annealing.h "logger.h")

target_link_libraries(saleman PRIVATE raylib)