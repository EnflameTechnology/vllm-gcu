set(PCALS_PATH module_package/pcals)
set(PCALS_COMMITID "3a2ca3e")
set(PCALS_BRANCH "master")
set(PCALS_DOWN_MODE FILE)
set(PCALS_SEMI_NAME "")
unset(FETCH_OPTIONS)
unset(PACKAGE_CMDS)
unset(PACKAGE_FILES)

set(PACKAGE_CMDS "mkdir -p ${CMAKE_FPKG_LIBDIR}; mv /FILE/ -t ${CMAKE_FPKG_LIBDIR}")
set(PACKAGE_FILES "${CMAKE_FPKG_LIBDIR}//FILE/")
set(FETCH_OPTIONS PKG_ONLY ON)

set(PCALS_GDRC_PACKAGE_VERSION "2.3")
set(PCALS_ECCL_PACKAGE_VERSION "3.5.20250913")

if(NOT ECCL_DDEB_LINK)
    set(ECCL_DDEB_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb)
endif()
fetchFromArtifactory(pcals_eccl_ddeb
    FILE ${ECCL_DDEB_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    ${FETCH_OPTIONS}
)


if(NOT ECCL_LINK)
    set(ECCL_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb)
endif()
fetchFromArtifactory(pcals_eccl_deb
    FILE ${ECCL_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    EXTRACT ON
)

add_library(pcals_eccl_includes INTERFACE)
if(EXISTS ${pcals_eccl_deb_SOURCE_DIR}/usr)
    set(pcals_eccl_deb_SOURCE_DIR ${pcals_eccl_deb_SOURCE_DIR}/usr CACHE INTERNAL "" FORCE)
endif()
target_include_directories(pcals_eccl_includes INTERFACE ${pcals_eccl_deb_SOURCE_DIR}/include/)
add_library(eccl SHARED IMPORTED GLOBAL)
set_target_properties(eccl PROPERTIES IMPORTED_LOCATION ${pcals_eccl_deb_SOURCE_DIR}/lib/libeccl.so)
target_link_libraries(eccl INTERFACE pcals_eccl_includes)

if(NOT ECCL_TESTS_LINK)
    set(ECCL_TESTS_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl-tests${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb)
endif()
fetchFromArtifactory(pcals_eccl_tests_deb
    FILE ${ECCL_TESTS_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
)
