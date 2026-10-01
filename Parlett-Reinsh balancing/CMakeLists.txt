cmake_minimum_required(VERSION 3.14)
project(f22_linalg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(linalg linalg.cpp)

add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE linalg)
