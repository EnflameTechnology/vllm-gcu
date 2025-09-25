if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set(PACKAGE_CMDS "mkdir -p ${CMAKE_FPKG_PYTHON_PACKAGES}; mv /FILE/ -t ${CMAKE_FPKG_PYTHON_PACKAGES}")
    set(PACKAGE_FILES "${CMAKE_FPKG_PYTHON_PACKAGES}//FILE/")
    set(_COMMIT_ID f80309a)
    set(_VERSION 2.6.3+torch.2.5.1.gcu.3.4.20250912)
    string(REPLACE "torch.2.5.1" "torch.2.8.0" _VERSION "${_VERSION}")
    set(_MNAME      flash_attn)
    set(_SEMI_NAME  "")
    set(_BRANCH master)

    set(fa_torch_version_list 2.8.0)
    function(get_version _version _torch_version)
        set(_v "${_VERSION}")
        foreach(_tv IN LISTS fa_torch_version_list)
            string(REPLACE "torch.${_tv}" "torch.${_torch_version}" _v "${_v}")
        endforeach()
        set(${_version} ${_v} PARENT_SCOPE)
    endfunction()

    get_version(_TORCH28_VERSION 2.8.0)
    set(_PY_VERSIONS 310)
    foreach(_pv ${_PY_VERSIONS})
        set(_link module_package/flash-attention/${_COMMIT_ID}/${_MNAME}${_SEMI_NAME}-${_VERSION}-cp${_pv}-cp${_pv}-linux_x86_64.whl)
        fetchFromArtifactory(fetch_${_MNAME}_${_pv}_28
            FILE ${_link}
            PKG_COMMAND ${PACKAGE_CMDS}
            PKG_FILES ${PACKAGE_FILES}
            PKG_ONLY ON
            BRANCH ${_BRANCH}
            VERSION ${_TORCH28_VERSION}
        )
    endforeach()
endif ()
