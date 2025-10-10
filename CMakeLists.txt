cmake_minimum_required(VERSION 3.25)
project(kmeans_mpi)

set(CMAKE_CXX_STANDARD 23)

find_package(MPI REQUIRED)

#cmake_policy(SET CMP0144 NEW)
#cmake_policy(SET CMP0167 OLD)
set(BOOST_ROOT /boost/boost_1_82_0)
find_package(Boost 1.82.0 COMPONENTS mpi serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_executable(kmeans_mpi
        src/main.cpp)

target_link_libraries(kmeans_mpi PRIVATE MPI::MPI_CXX ${Boost_LIBRARIES})
