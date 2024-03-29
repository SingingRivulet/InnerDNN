cmake_minimum_required(VERSION 3.10)  
  
# 设置项目名称  
project(InnerDNN)  
  
# 设置包含目录  
include_directories(${PROJECT_SOURCE_DIR}/include)  
  
# 查找src目录中的所有.c文件  
file(GLOB_RECURSE SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.c")  
  
# 创建静态库libInnerDNN.a  
add_library(InnerDNN STATIC ${SRC_FILES})  

# 编译可执行文件
file(GLOB_RECURSE TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.c")  
foreach(TEST_FILE ${TEST_FILES})  
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)  
    set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/tests)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -fsanitize=address -fno-omit-frame-pointer -Wall -DDEBUG -DDEBUG_DUMP -rdynamic")
    add_executable(${TEST_NAME} ${TEST_FILE})  
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    target_link_libraries(${TEST_NAME} InnerDNN m GLESv2 EGL)  
endforeach()

file(GLOB_RECURSE TOOL_FILES "${PROJECT_SOURCE_DIR}/tools/*.c")  
foreach(TOOL_FILE ${TOOL_FILES})  
    get_filename_component(TOOL_NAME ${TOOL_FILE} NAME_WE)  
    set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/tools)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -fsanitize=address -fno-omit-frame-pointer")
    add_executable(${TOOL_NAME} ${TOOL_FILE})  
    set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    target_link_libraries(${TOOL_NAME} InnerDNN m GLESv2 EGL)  
endforeach()

file(GLOB_RECURSE INFERENCE_FILES "${PROJECT_SOURCE_DIR}/inference/*.c")  
foreach(INFERENCE_FILE ${INFERENCE_FILES})  
    get_filename_component(INFERENCE_NAME ${INFERENCE_FILE} NAME_WE)  
    set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/inference)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -fsanitize=address -fno-omit-frame-pointer")
    add_executable(${INFERENCE_NAME} ${INFERENCE_FILE})  
    set_target_properties(${INFERENCE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    target_link_libraries(${INFERENCE_NAME} InnerDNN m GLESv2 EGL)  
endforeach()