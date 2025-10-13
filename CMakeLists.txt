cmake_minimum_required(VERSION 3.25)
project(kmeans_mpi)

set(CMAKE_CXX_STANDARD 23)

find_package(MPI REQUIRED)

# For GCC or Clang: -O3 enables auto-vectorization (implies -ftree-vectorize),
# -march=native uses all available CPU extensions (e.g., AVX, SSE) on the build machine.
add_compile_options(-O3 -march=native -ftree-vectorize -flto -std=c++23)

#cmake_policy(SET CMP0144 NEW)
#cmake_policy(SET CMP0167 OLD)
set(BOOST_INCLUDE_LIBRARIES mpi serialization random)
set(BOOST_ENABLE_CMAKE ON)

include(FetchContent)
FetchContent_Declare(
        Boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.89.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Boost)

add_executable(kmeans_mpi
        src/main.cpp
        src/shared/Point.cpp
        src/shared/Point.hpp
        src/shared/DataSet.cpp
        src/shared/DataSet.hpp)

target_link_libraries(kmeans_mpi PRIVATE MPI::MPI_CXX Boost::serialization Boost::mpi Boost::mpi)
