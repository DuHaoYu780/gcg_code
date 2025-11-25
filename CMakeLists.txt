cmake_minimum_required(VERSION 2.8)
project(test)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4 REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.0")
    # OpenCV 4将某些模块移到了opencv_contrib
    find_package(OpenCV REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_features2d)
endif()
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}

)

add_library(${PROJECT_NAME} SHARED
src/ORBextractor.cc

include/ORBextractor.h
)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}

)

add_executable(test_app src/test.cpp)  # 添加后缀区分
target_link_libraries(test_app ${PROJECT_NAME})  # 保持链接关系
