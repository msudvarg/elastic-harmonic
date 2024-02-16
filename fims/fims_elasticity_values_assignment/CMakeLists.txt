cmake_minimum_required(VERSION 3.17)
project(fims)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

# set(OpenCV_DIR "/home/pi/fims/opencv/build_install/lib/cmake/opencv4") 
find_package(OpenCV REQUIRED)
# find_package(Boost COMPONENTS system filesystem REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/header) 
include_directories(${PROJECT_SOURCE_DIR}/eigen)

# Get all the source files
file(GLOB SOURCES "source/*.cpp" "main.cpp")  

# Specify the executable target to be built
add_executable(fims ${SOURCES})  

target_link_libraries(fims ${OpenCV_LIBS} rt pthread atomic stdc++fs)



# cmake_minimum_required(VERSION 3.17)
# project(myfims)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope")
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp")

# find_package(OpenCV REQUIRED)
# find_package(Boost COMPONENTS system filesystem REQUIRED)

# # Specify OpenMP_CXX here
# find_package(OpenMP REQUIRED)
# # if(OpenMP_CXX_FOUND)
# #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# # endif()


# set(ZeroMQ_INCLUDE_DIR "/usr/local/include")
# set(ZeroMQ_LIBRARY "/usr/local/lib/libzmq.dylib")
# # set(OPENMP_INCLUDE "/usr/local/opt/llvm/include")
# # set(OPENMP_LIB_DIR "/usr/local/opt/llvm/lib")
# # set(OpenMP_C "/usr/local/opt/llvm")
# # set(OpenMP_CXX "/usr/local/opt/llvm")


# # include_directories("/usr/local/opt/libomp/include")
# # link_directories("/usr/local/opt/libomp/lib")

# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${ZeroMQ_INCLUDE_DIR})
# include_directories(${PROJECT_SOURCE_DIR}/header)  
# include_directories(${PROJECT_SOURCE_DIR}/eigen)
# include_directories(${PROJECT_SOURCE_DIR}/json/include)  


# link_directories(/usr/local/lib)

# file(GLOB SOURCES "source/*.cpp" "main.cpp")  # Get all .cpp files from source directory and main.cpp

# # Specify the executable target to be built
# add_executable(myfims ${SOURCES})  # Note that SOURCES is added here

# # Link the libraries to the executable target
# # target_link_libraries(myfims ${OpenCV_LIBS} ${ZeroMQ_LIBRARY} omp )

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(myfims ${OpenCV_LIBS} ${ZeroMQ_LIBRARY} OpenMP::OpenMP_CXX)
# else()
#     target_link_libraries(myfims ${OpenCV_LIBS} ${ZeroMQ_LIBRARY})
# endif()
