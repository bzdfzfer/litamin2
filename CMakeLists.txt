cmake_minimum_required(VERSION 3.10.0)
project(litamin2)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread") 
set(CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED)

find_package(Ceres REQUIRED QUIET)


find_package(Eigen3 REQUIRED)

find_package(OpenMP)

if (OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#	set(CMAKE_C_FLAGS "$(CMAKE_C_FLAGS) $(OpenMP_C_FLAGS)")
#	set(CMAKE_CXX_FLAGS "$(CMAKE_CXX_FLAGS) $(OpenMP_CXX_FLAGS)")
endif()


find_package(catkin)

###################################
## catkin specific configuration ##
###################################
if(catkin_FOUND)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES litamin
  )
endif()

include_directories(
	include
	${PCL_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

add_library(litamin SHARED
	src/litamin2/litamin2.cpp
	src/fast_gicp/gicp/lsq_registration.cpp
	src/litamin2/litamin2point2voxel.cpp
	src/litamin2/litamin2point2voxelnewton.cpp
)
target_link_libraries(litamin
	${PCL_LIBRARIES}
 	${CERES_LIBRARIES}	
)
target_include_directories(litamin PUBLIC
	include
	${PCL_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)


########### TEST 
#######################################################

# this is voxel to voxel matching.
add_executable(litamin2_align test/test_align_with_litamin2.cpp)
target_link_libraries(litamin2_align
${catkin_LIBRARIES}
${CERES_LIBRARIES}
${PCL_LIBRARIES}
litamin
)


add_executable(litamin2point2voxel_align test/test_align_with_litamin2point2voxel.cpp)
target_link_libraries(litamin2point2voxel_align
${catkin_LIBRARIES}
${CERES_LIBRARIES}
${PCL_LIBRARIES}
litamin
)

add_executable(litamin2newton_align test/test_align_with_litamin2point2voxelnewton.cpp)
target_link_libraries(litamin2newton_align
${catkin_LIBRARIES}
${CERES_LIBRARIES}
${PCL_LIBRARIES}
litamin
)

add_executable(test_dataloader test/test_dataloader.cpp
	src/dataloader/dataset.cpp
	)
target_link_libraries(test_dataloader 
${catkin_LIBRARIES}
${PCL_LIBRARIES}
)