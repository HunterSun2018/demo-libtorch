cmake_minimum_required(VERSION 3.10)

project(demo)

set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED true)
set(CAFFE2_USE_CUDNN true)

find_package(PythonInterp REQUIRED)

set(CMAKE_PREFIX_PATH ~/libtorch2.0.1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

aux_source_directory(src SRC)
add_executable(demo ${SRC})

target_link_libraries(demo "${TORCH_LIBRARIES}")