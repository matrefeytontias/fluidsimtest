cmake_minimum_required(VERSION 3.13)

### Project setup

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(FluidSim DESCRIPTION "Fluid simulation playground" LANGUAGES C CXX)

# Main executable

set(SOURCES
    Source/main.cpp
    Source/gui.cpp
    Source/gui.h
    Source/Context.h
    Source/Camera.h
    Source/fields.hpp
    Source/fluid.hpp
    Source/render.hpp
    Source/render.cpp
    Source/solver.hpp
    Source/solver.cpp)

add_executable(FluidSimTest ${SOURCES})

target_compile_features(FluidSimTest PRIVATE cxx_std_17)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT FluidSimTest)

# Resources

set(RESOURCES
    # Fluid sim shaders
    shaders/sim/entry_point.glsl
    shaders/sim/advection.glsl
    shaders/sim/divergence.glsl
    shaders/sim/forces.glsl
    shaders/sim/grid_scroll.glsl
    shaders/sim/jacobi.glsl
    shaders/sim/projection.glsl
    # Drawing shaders
    shaders/draw/debug_vertex.glsl
    shaders/draw/debug_fragment.glsl
    shaders/draw/fluid_vertex.glsl
    shaders/draw/fluid_fragment.glsl
    shaders/draw/grid_vertex.glsl
    shaders/draw/grid_fragment.glsl)
target_sources(FluidSimTest PRIVATE ${RESOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/shaders PREFIX Shaders FILES ${RESOURCES})

foreach(RES ${RESOURCES})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${RES}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${RES} ${CMAKE_CURRENT_BINARY_DIR}/${RES}
        MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${RES})
endforeach()

### Third-party libraries

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_USE_HYBRID_HPG ON CACHE BOOL "" FORCE)
add_subdirectory(ThirdParty/glfw)
set_target_properties(glfw PROPERTIES FOLDER "ThirdParty/GLFW3")
set_target_properties(update_mappings PROPERTIES FOLDER "ThirdParty/GLFW3")

# imgui
# ThirdParty/imgui/CMakeLists.txt is configured with GLFW and OpenGL3
add_subdirectory(ThirdParty/imgui)
set_target_properties(imgui PROPERTIES FOLDER "ThirdParty")

# Empty
set(EMPTY_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
add_subdirectory(ThirdParty/Empty)
set_target_properties(Empty PROPERTIES FOLDER "ThirdParty")

# Link everything
target_link_libraries(FluidSimTest PUBLIC glfw imgui imgui-glfw imgui-opengl3 Empty)
