cmake_minimum_required(VERSION 3.10)  
  
# 设置项目名称  
project(InnerDNN)  
  
# 设置包含目录  
include_directories(${PROJECT_SOURCE_DIR}/include)  
  
# 查找src目录中的所有.c文件  
file(GLOB_RECURSE SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.c")  
  
# 查找tests目录中的所有.c文件  
file(GLOB_RECURSE TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.c")  
  
# 创建静态库libInnerDNN.a  
add_library(InnerDNN STATIC ${SRC_FILES})  
  
# 对于tests目录中的每个.c文件，创建一个可执行文件，并链接到libInnerDNN.a库  
foreach(TEST_FILE ${TEST_FILES})  
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)  
    add_executable(${TEST_NAME} ${TEST_FILE})  
    target_link_libraries(${TEST_NAME} InnerDNN m GLESv2 EGL)  
endforeach()