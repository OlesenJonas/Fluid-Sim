# variable to collect include directories in
# set(NEW_INCLUDE_DIRS)

# todo: macro
add_subdirectory(thirdparty/stb)
# get_target_property(INCLUDE_DIRS stb INCLUDE_DIRS)
# list(APPEND NEW_INCLUDE_DIRS ${INCLUDE_DIRS})

add_subdirectory(thirdparty/ImGui)
# get_target_property(INCLUDE_DIRS ImGui INCLUDE_DIRS)
# list(APPEND NEW_INCLUDE_DIRS ${INCLUDE_DIRS})

add_subdirectory(thirdparty/glad)
# get_target_property(INCLUDE_DIRS glad INCLUDE_DIRS)
# list(APPEND NEW_INCLUDE_DIRS ${INCLUDE_DIRS})

add_subdirectory(intern)
# get_target_property(INCLUDE_DIRS intern INCLUDE_DIRS)
# list(APPEND NEW_INCLUDE_DIRS ${INCLUDE_DIRS})

# otherwise glad.h complains that it already has been included
target_compile_definitions(glfw INTERFACE "-DGLFW_INCLUDE_NONE" )

# set(INCLUDE_DIRS ${NEW_INCLUDE_DIRS} PARENT_SCOPE)
# create list of all libraries that executables should link to
set(LIBS OpenGL::GL glfw glm::glm glad ImGui stb intern PARENT_SCOPE)