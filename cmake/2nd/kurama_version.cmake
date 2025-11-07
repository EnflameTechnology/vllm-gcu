set(KURAMA_COMMITID 6f84863)
set(KURAMA_PACKAGE_VERSION 3.3.1+1.0.20251103)
set(KURAMA_BRANCH "main")

# ######################################################
# ################  triton-gcu  ########################
# ######################################################

set(KURAMA_PATH module_package/kurama)
set(KURAMA_DOWN_MODE FILE)
set(KURAMA_SEMI_NAME "")

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64" AND NOT PROJECT_GIT_URL)
    # set(KURAMA_XNAS_LINK "http://10.12.110.200:8080/release/kurama/kurama-release/22/integration/f2fc090/")
    if(KURAMA_XNAS_LINK)
        link_pattern_var("${KURAMA_XNAS_LINK}"
            VARS
            KURAMA_DEB_LINK
            KURAMA_DDEB_LINK
            KURAMA_INTERNAL_DDEB_LINK
            KURAMA_WHL_LINK
            PATTERNS
            "triton-gcu${KURAMA_SEMI_NAME}_[0-9].*${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
            "triton-gcu${KURAMA_SEMI_NAME}_[0-9].*${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb"
            "triton-gcu${KURAMA_SEMI_NAME}_[0-9].*${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym-internal.ddeb"
            "triton-gcu${KURAMA_SEMI_NAME}-[0-9].*-py3.10-none-any.whl"
        )
        message(STATUS "KURAMA_DEB_LINK: ${KURAMA_DEB_LINK}[triton-gcu${KURAMA_SEMI_NAME}_[0-9].*${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb]")

        if(NOT KURAMA_DEB_LINK)
            message(WARNING "Can not find some links from ${KURAMA_XNAS_LINK}")
        endif()
    else()
        set(KURAMA_DEB_LINK ${KURAMA_PATH}/${KURAMA_COMMITID}/triton-gcu${KURAMA_SEMI_NAME}_${KURAMA_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb)
        set(KURAMA_DDEB_LINK ${KURAMA_PATH}/${KURAMA_COMMITID}/triton-gcu${KURAMA_SEMI_NAME}_${KURAMA_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym.ddeb)
        set(KURAMA_INTERNAL_DDEB_LINK ${KURAMA_PATH}/${KURAMA_COMMITID}/triton-gcu${KURAMA_SEMI_NAME}_${KURAMA_PACKAGE_VERSION}-1_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-dbgsym-internal.ddeb)
        set(KURAMA_WHL_LINK ${KURAMA_PATH}/${KURAMA_COMMITID}/triton_gcu${KURAMA_SEMI_NAME}-${KURAMA_PACKAGE_VERSION}-py3-none-any.whl)
    endif()

    set(PACKAGE_CMDS "mkdir -p ${CMAKE_FPKG_LIBDIR}; mv /FILE/ -t ${CMAKE_FPKG_LIBDIR}")
    set(PACKAGE_WHL_CMDS "mkdir -p ${CMAKE_FPKG_PYTHON_PACKAGES}; mv /FILE/ -t ${CMAKE_FPKG_PYTHON_PACKAGES}")
    set(PACKAGE_FILES "${CMAKE_FPKG_LIBDIR}//FILE/")

    if(KURAMA_DDEB_LINK)
        fetchFromArtifactory(fetch_triton_gcu_ddeb
            FILE ${KURAMA_DDEB_LINK}
            PKG_COMMAND ${PACKAGE_CMDS}
            SOURCE_DIR ${CMAKE_BINARY_DIR}
            PKG_FILES ${PACKAGE_FILES}
            PKG_ONLY ON
            BRANCH ${KURAMA_BRANCH}
            VERSION ${KURAMA_PACKAGE_VERSION}
        )
    endif()

    if(KURAMA_INTERNAL_DDEB_LINK)
        fetchFromArtifactory(fetch_triton_gcu_internal_ddeb
            FILE ${KURAMA_INTERNAL_DDEB_LINK}
            PKG_COMMAND ${PACKAGE_CMDS}
            SOURCE_DIR ${CMAKE_BINARY_DIR}
            PKG_FILES ${PACKAGE_FILES}
            PKG_ONLY ON
            BRANCH ${KURAMA_BRANCH}
            VERSION ${KURAMA_PACKAGE_VERSION}
        )
    endif()

    if(KURAMA_DEB_LINK)
        fetchFromArtifactory(fetch_triton_gcu_deb
            FILE ${KURAMA_DEB_LINK}
            PKG_COMMAND ${PACKAGE_CMDS}
            PKG_FILES ${PACKAGE_FILES}
            SOURCE_DIR ${CMAKE_BINARY_DIR}
            PKG_ONLY ON
            BRANCH ${KURAMA_BRANCH}
            VERSION ${KURAMA_PACKAGE_VERSION}
        )
    endif()

    if(KURAMA_WHL_LINK)
        fetchFromArtifactory(fetch_triton_gcu_whl
            FILE ${KURAMA_WHL_LINK}
            PKG_COMMAND ${PACKAGE_WHL_CMDS}
            PKG_FILES ${PACKAGE_FILES}
            SOURCE_DIR ${CMAKE_BINARY_DIR}
            PKG_ONLY ON
            BRANCH ${KURAMA_BRANCH}
            VERSION ${KURAMA_PACKAGE_VERSION}
        )
    endif()

endif()
