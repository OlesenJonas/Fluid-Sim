cmake_minimum_required(VERSION 3.2)
include(DefaultLibrary)
# ImGui needs glfw!
target_link_libraries(ImGui PRIVATE glfw)