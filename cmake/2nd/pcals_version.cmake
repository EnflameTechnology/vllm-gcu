set(PCALS_PATH module_package/pcals)
set(PCALS_COMMITID "de9154a")
set(PCALS_BRANCH "master")
set(PCALS_DOWN_MODE FILE)
set(PCALS_SEMI_NAME "")
unset(FETCH_OPTIONS)
unset(PACKAGE_CMDS)
unset(PACKAGE_FILES)

set(PACKAGE_CMDS "mkdir -p ${CMAKE_FPKG_LIBDIR}; mv /FILE/ -t ${CMAKE_FPKG_LIBDIR}")
set(PACKAGE_FILES "${CMAKE_FPKG_LIBDIR}//FILE/")
set(FETCH_OPTIONS PKG_ONLY ON)

#set(PCALS_XNAS_LINK "http://10.12.110.200:8080/release/pcals-release/152/integration/aff9f8a")
if(PCALS_XNAS_LINK)
    link_pattern_var("${PCALS_XNAS_LINK}"
        VARS
            ECCL_LINK
            ECCL_DDEB_LINK
            ECCL_INTERNAL_DDEB_LINK
            ECCL_TESTS_LINK
            ECCL_TESTS_DDEB_LINK
            ECCL_TESTS_INTERNAL_DDEB_LINK
        PATTERNS
            "eccl_[0-9].*_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
            "eccl_[0-9].*_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb"
            "eccl_[0-9].*_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym-internal.ddeb"
            "eccl-tests_[0-9].*_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
            "eccl-tests_[0-9].*_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb"
            "eccl-tests_[0-9].*_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym-internal.ddeb"
    )
endif()

set(PCALS_GDRC_PACKAGE_VERSION "2.3")
set(PCALS_ECCL_PACKAGE_VERSION "3.4.20250506")
fetchFromArtifactory(pcals_gdrc_deb
    FILE ${PCALS_PATH}/${PCALS_COMMITID}/pcals_tops_gdrc${PCALS_SEMI_NAME}_${PCALS_GDRC_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    ${FETCH_OPTIONS}
)
if(NOT DEB_ONLY)
    fetchFromArtifactory(pcals_gdrc_rpm
        FILE ${PCALS_PATH}/${PCALS_COMMITID}/pcals_tops_gdrc${PCALS_SEMI_NAME}-${PCALS_GDRC_PACKAGE_VERSION}-1.${_RPM_PACKAGE_ARCHITECTURE}.rpm
        PKG_COMMAND ${PACKAGE_CMDS}
        PKG_FILES ${PACKAGE_FILES}
        ${FETCH_OPTIONS}
    )
endif()

fetchFromArtifactory(pcals_gdrc_run
    FILE ${PCALS_PATH}/${PCALS_COMMITID}/enflame_pcals_gdrdrv-${CMAKE_SYSTEM_PROCESSOR}-gcc-${PCALS_GDRC_PACKAGE_VERSION}.run
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    ${FETCH_OPTIONS}
)

if(NOT ECCL_DDEB_LINK)
    set(ECCL_DDEB_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb)
endif()
fetchFromArtifactory(pcals_eccl_ddeb
    FILE ${ECCL_DDEB_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    ${FETCH_OPTIONS}
)

if(NOT ECCL_INTERNAL_DDEB_LINK)
    set(ECCL_INTERNAL_DDEB_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym-internal.ddeb)
endif()
fetchFromArtifactory(pcals_eccl_internal_ddeb
    FILE ${ECCL_INTERNAL_DDEB_LINK}
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

if(NOT DEB_ONLY)
    fetchFromArtifactory(pcals_eccl_rpm
        FILE ${PCALS_PATH}/${PCALS_COMMITID}/eccl${PCALS_SEMI_NAME}-${PCALS_ECCL_PACKAGE_VERSION}-1.${_RPM_PACKAGE_ARCHITECTURE}.rpm
        PKG_COMMAND ${PACKAGE_CMDS}
        PKG_FILES ${PACKAGE_FILES}
        ${FETCH_OPTIONS}
    )
endif()

if(NOT ECCL_TESTS_DDEB_LINK)
    set(ECCL_TESTS_DDEB_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl-tests${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb)
endif()
fetchFromArtifactory(pcals_eccl_tests_ddeb
    FILE ${ECCL_TESTS_DDEB_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    ${FETCH_OPTIONS}
)

if(NOT ECCL_TESTS_INTERNAL_DDEB_LINK)
    set(ECCL_TESTS_INTERNAL_DDEB_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl-tests${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym-internal.ddeb)
endif()
fetchFromArtifactory(pcals_eccl_tests_internal_ddeb
    FILE ${ECCL_TESTS_INTERNAL_DDEB_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
    ${FETCH_OPTIONS}
)

if(NOT ECCL_TESTS_LINK)
    set(ECCL_TESTS_LINK ${PCALS_PATH}/${PCALS_COMMITID}/eccl-tests${PCALS_SEMI_NAME}_${PCALS_ECCL_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb)
endif()
fetchFromArtifactory(pcals_eccl_tests_deb
    FILE ${ECCL_TESTS_LINK}
    PKG_COMMAND ${PACKAGE_CMDS}
    PKG_FILES ${PACKAGE_FILES}
)

if(NOT DEB_ONLY)
    fetchFromArtifactory(pcals_eccl_tests_rpm
        FILE ${PCALS_PATH}/${PCALS_COMMITID}/eccl-tests${PCALS_SEMI_NAME}-${PCALS_ECCL_PACKAGE_VERSION}-1.${_RPM_PACKAGE_ARCHITECTURE}.rpm
        PKG_COMMAND ${PACKAGE_CMDS}
        PKG_FILES ${PACKAGE_FILES}
        ${FETCH_OPTIONS}
    )
endif()