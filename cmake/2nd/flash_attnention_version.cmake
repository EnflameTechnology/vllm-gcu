if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set(PACKAGE_CMDS "mkdir -p ${CMAKE_FPKG_PYTHON_PACKAGES}; mv /FILE/ -t ${CMAKE_FPKG_PYTHON_PACKAGES}")
    set(PACKAGE_FILES "${CMAKE_FPKG_PYTHON_PACKAGES}//FILE/")
    set(_COMMIT_ID 0576e10)
    set(_VERSION 2.6.3+torch.2.8.0.gcu.3.4.20251011)
    set(_MNAME      flash_attn)
    set(_SEMI_NAME  "")
    set(_BRANCH master)
    set(_pv 310)
    set(_link module_package/flash-attention/${_COMMIT_ID}/${_MNAME}${_SEMI_NAME}-${_VERSION}-cp${_pv}-cp${_pv}-linux_x86_64.whl)
    fetchFromArtifactory(fetch_${_MNAME}_${_pv}_28
        FILE ${_link}
        PKG_COMMAND ${PACKAGE_CMDS}
        PKG_FILES ${PACKAGE_FILES}
        PKG_ONLY ON
        BRANCH ${_BRANCH}
        VERSION ${_VERSION}
    )
endif ()
