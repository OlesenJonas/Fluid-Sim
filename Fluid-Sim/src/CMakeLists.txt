include(FindOpenGL)
# get libraries from vcpk
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# create libraries from local files
add_subdirectory("libraries/")

# create executables
add_subdirectory("executables/")
# tests?