cmake_minimum_required(VERSION 3.15)
project(SpecBRDF_PROJ)

set(CMAKE_CXX_STANDARD 14)

if(BUILD_PYTHON)
    set(Python_ADDITIONAL_VERSIONS 3.6)
endif()

set (CMAKE_CXX_STANDARD 11)
add_subdirectory(pybind11)
pybind11_add_module(SpecBRDF src/SpecBRDF.cpp)



# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(SpecBRDF
        PUBLIC
        ${PROJECT_SOURCE_DIR}/inc
        )

# Add an executable with the above sources
#add_executable(SpecBRDF_bin
#        src/test.cpp
#        )
#
#target_link_libraries( SpecBRDF_bin
#        PRIVATE
#        SpecBRDF
#        )