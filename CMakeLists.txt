cmake_minimum_required(VERSION 3.15)
project(pacbio_methylation_hmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

find_package(htslib QUIET)
if(NOT htslib_FOUND)
    find_path(HTSLIB_INCLUDE_DIR htslib/sam.h
        PATHS
        $ENV{HTSLIB_ROOT}/include
        /usr/include
        /usr/local/include
        /opt/htslib/include
    )
    find_library(HTSLIB_LIBRARY
        NAMES hts
        PATHS
        $ENV{HTSLIB_ROOT}/lib
        /usr/lib
        /usr/local/lib
        /opt/htslib/lib
    )
    if(HTSLIB_INCLUDE_DIR AND HTSLIB_LIBRARY)
        set(HTSLIB_FOUND TRUE)
        set(HTSLIB_INCLUDE_DIRS ${HTSLIB_INCLUDE_DIR})
        set(HTSLIB_LIBRARIES ${HTSLIB_LIBRARY})
    else()
        message(FATAL_ERROR "htslib not found. Please install htslib or set HTSLIB_ROOT.")
    endif()
else()
    set(HTSLIB_INCLUDE_DIRS ${HTSLIB_INCLUDE_DIR})
    set(HTSLIB_LIBRARIES ${HTSLIB_LIBRARY})
endif()

find_package(pybind11 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${HTSLIB_INCLUDE_DIRS}
)

add_subdirectory(cpp)
