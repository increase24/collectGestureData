cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project( keypoint_detection)

set(CMAKE_BUILD_TYPE "Release")

set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
include_directories("D:/envs/onnxruntime-win-x64-1.1.0/include")
include_directories("./include")

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable( palm-detection palm-detection.cpp)
target_link_libraries( palm-detection ${OpenCV_LIBS} )
target_link_libraries( palm-detection ${TORCH_LIBRARIES})
target_link_libraries( palm-detection "D:\\envs\\onnxruntime-win-x64-1.1.0\\lib\\onnxruntime.lib")
set_target_properties( palm-detection PROPERTIES COMPILE_PDB_NAME speed-bench)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread -fPIC -ltvm_runtime")


