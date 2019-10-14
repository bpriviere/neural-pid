cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)
project (Orca)

add_subdirectory(RVO2)

# find_package (Eigen3 REQUIRED)
# find_package(stdc++fs)

include_directories(
  RVO2/src
)

add_executable(orca
  orca.cpp
)
target_link_libraries(orca
  RVO
)
