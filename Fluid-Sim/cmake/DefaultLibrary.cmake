get_filename_component(FOLDER ${CMAKE_CURRENT_SOURCE_DIR} NAME)

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)
file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

add_library(${FOLDER} STATIC ${SOURCES})
message(STATUS "Added Library: ${FOLDER}")
target_include_directories(${FOLDER} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(DIR_ONE_ABOVE ../ ABSOLUTE)
target_include_directories(${FOLDER} PUBLIC ${DIR_ONE_ABOVE})


# add a folder above as include dir, so that folder name can be included in #include path
# get_filename_component(DIR_ONE_ABOVE ../ ABSOLUTE)
# set(DIRS ${DIR_ONE_ABOVE} ${CMAKE_CURRENT_SOURCE_DIR})
# set_property(TARGET ${FOLDER} PROPERTY INCLUDE_DIRS ${DIRS})