cmake_minimum_required(VERSION 3.20)
project(Tensoris VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS src/*.hpp)

add_executable(tensoris_demo ${SOURCES} ${HEADERS} "src/tensor.cpp")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES} ${HEADERS})

option(TENSORIS_ENABLE_PROFILER "Enable built-in profiler" ON)
if (TENSORIS_ENABLE_PROFILER)
	target_compile_definitions(tensoris_demo PUBLIC TENSORIS_ENABLE_PROFILER=1)
endif()