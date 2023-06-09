list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.8.0)
    include(CMakeFindDependencyMacro)
    find_dependency(zeus 1.0.1 REQUIRED)
    find_dependency(OpenCV 4.7.0 REQUIRED)
    find_dependency(TBB 2021.8.0 REQUIRED)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/radTargets.cmake)

