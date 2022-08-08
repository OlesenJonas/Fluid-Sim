get_filename_component(FOLDER_VAR ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project("${FOLDER_VAR}")

file(GLOB_RECURSE SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/*.c
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

add_compile_definitions(EXECUTABLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_definitions(EXECUTABLE_NAME="${FOLDER_VAR}")

# Define the executable
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

# IF(CMAKE_BUILD_TYPE MATCHES Release)
# 	# For release builds we want to put the shader sources and misc files in the executable folder
# 	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD  
# 		COMMAND ${CMAKE_COMMAND} -E copy_directory 		
# 		"${CMAKE_SOURCE_DIR}/misc/"     
# 		"$<TARGET_FILE_DIR:${PROJECT_NAME}>/misc/")
# 	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD  
# 		COMMAND ${CMAKE_COMMAND} -E copy_directory 		
# 		"${CMAKE_SOURCE_DIR}/src/Shaders"     
# 		"$<TARGET_FILE_DIR:${PROJECT_NAME}>/Shaders/") 
# ENDIF()