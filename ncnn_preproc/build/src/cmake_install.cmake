# Install script for directory: /root/studyML/ncnn_preproc/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/root/studyML/ncnn_preproc/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/root/studyML/ncnn_preproc/build/src/libncnn.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/root/studyML/ncnn_preproc/src/allocator.h"
    "/root/studyML/ncnn_preproc/src/benchmark.h"
    "/root/studyML/ncnn_preproc/src/blob.h"
    "/root/studyML/ncnn_preproc/src/c_api.h"
    "/root/studyML/ncnn_preproc/src/command.h"
    "/root/studyML/ncnn_preproc/src/cpu.h"
    "/root/studyML/ncnn_preproc/src/datareader.h"
    "/root/studyML/ncnn_preproc/src/gpu.h"
    "/root/studyML/ncnn_preproc/src/layer.h"
    "/root/studyML/ncnn_preproc/src/layer_shader_type.h"
    "/root/studyML/ncnn_preproc/src/layer_type.h"
    "/root/studyML/ncnn_preproc/src/mat.h"
    "/root/studyML/ncnn_preproc/src/modelbin.h"
    "/root/studyML/ncnn_preproc/src/net.h"
    "/root/studyML/ncnn_preproc/src/option.h"
    "/root/studyML/ncnn_preproc/src/paramdict.h"
    "/root/studyML/ncnn_preproc/src/pipeline.h"
    "/root/studyML/ncnn_preproc/src/pipelinecache.h"
    "/root/studyML/ncnn_preproc/src/simpleocv.h"
    "/root/studyML/ncnn_preproc/src/simpleomp.h"
    "/root/studyML/ncnn_preproc/src/simplestl.h"
    "/root/studyML/ncnn_preproc/src/vulkan_header_fix.h"
    "/root/studyML/ncnn_preproc/build/src/ncnn_export.h"
    "/root/studyML/ncnn_preproc/build/src/layer_shader_type_enum.h"
    "/root/studyML/ncnn_preproc/build/src/layer_type_enum.h"
    "/root/studyML/ncnn_preproc/build/src/platform.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/root/studyML/ncnn_preproc/build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/root/studyML/ncnn_preproc/build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/root/studyML/ncnn_preproc/build/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/root/studyML/ncnn_preproc/build/src/ncnnConfig.cmake")
endif()

