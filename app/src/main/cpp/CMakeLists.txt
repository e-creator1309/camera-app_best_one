cmake_minimum_required(VERSION 3.22.1)
project("camimg" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV via Prefab (org.opencv:opencv:4.10.0 AAR shipped with the app)
find_package(opencv REQUIRED)

# jni_bridge.c #includes all C translation units.
# doc_scan.cpp is listed separately — it needs C++ for OpenCV.
add_library(camimg SHARED jni_bridge.c doc_scan.cpp)
target_compile_options(camimg PRIVATE -O2 -Wall -Wno-unused-parameter)

find_library(log-lib         log)
find_library(jnigraphics-lib jnigraphics)
find_library(math-lib        m)

target_link_libraries(camimg
    ${log-lib}
    ${jnigraphics-lib}
    ${math-lib}
    opencv::opencv_java4
)
