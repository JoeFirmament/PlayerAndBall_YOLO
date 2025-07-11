cmake_minimum_required(VERSION 3.15)

project(rknn_model_zoo_utils)

# 设置RGA库路径
set(LIBRGA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rga/include)
# 强制使用aarch64版本的librga.so
set(LIBRGA ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rga/libs/Linux/gcc-aarch64/librga.so)

# 设置STB库路径
set(STB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/stb)

add_library(fileutils STATIC
    file_utils.c
)
target_include_directories(fileutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(imagedrawing STATIC
    image_drawing.c
)
target_include_directories(imagedrawing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(DISABLE_RGA AND NOT (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103" OR TARGET_SOC STREQUAL "rv1103b"))
    add_definitions(-DDISABLE_RGA)
endif ()

# only RGA on rv1106 and rk3588 support handle
if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rk3588")
    add_definitions(-DLIBRGA_IM2D_HANDLE)
endif()

add_library(imageutils STATIC
    image_utils.c
)

target_include_directories(imageutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_INCLUDES}
    ${LIBRGA_INCLUDES}
)

target_link_libraries(imageutils
    ${LIBRGA}
)

if (DISABLE_LIBJPEG)
    add_definitions(-DDISABLE_LIBJPEG)
else()
    target_link_libraries(imageutils
        ${LIBJPEG}
    )
    target_include_directories(imageutils PUBLIC
        ${LIBJPEG_INCLUDES}
    )
endif()

# 根据DISABLE_AUDIO_UTILS变量决定是否编译audioutils库
if (NOT DISABLE_AUDIO_UTILS)
    add_library(audioutils STATIC
        audio_utils.c
    )

    target_link_libraries(audioutils
        ${LIBSNDFILE}
    )

    target_include_directories(audioutils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBSNDFILE_INCLUDES}
    )
endif()