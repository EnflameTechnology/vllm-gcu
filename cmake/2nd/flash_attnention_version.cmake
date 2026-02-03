
function(get_version _version _torch_version)
    set(_v "${_VERSION}")
    if(NOT _torch_version IN_LIST fa_torch_version_list)
        set(${_version} "" PARENT_SCOPE)
        return()
    endif()
    foreach(_tv IN LISTS fa_torch_version_list)
        string(REPLACE "torch.${_tv}" "torch.${_torch_version}" _v "${_v}")
    endforeach()
    set(${_version} ${_v} PARENT_SCOPE)
endfunction()

set(fa_torch_version_list 2.3.0 2.5.1 2.6.0 2.7.0 2.8.0 2.9.0)


if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set(PACKAGE_CMDS "mkdir -p ${CMAKE_FPKG_PYTHON_PACKAGES}; mv /FILE/ -t ${CMAKE_FPKG_PYTHON_PACKAGES}")
    set(PACKAGE_FILES "${CMAKE_FPKG_PYTHON_PACKAGES}//FILE/")
    set(_COMMIT_ID dbd343b)
    set(_VERSION 2.7.2+torch.2.7.0.gcu.3.4.20260130)
    set(_MNAME      flash_attn)
    set(_SEMI_NAME  "")
    set(_BRANCH master)
    set(_pvs "310" "312")
    get_version(_TORCH28_VERSION 2.8.0)
    foreach(_pv IN LISTS _pvs)
        set(_link module_package/flash-attention/${_COMMIT_ID}/${_MNAME}${_SEMI_NAME}-${_TORCH28_VERSION}-cp${_pv}-cp${_pv}-linux_x86_64.whl)
        fetchFromArtifactory(fetch_${_MNAME}_${_pv}_28
            FILE ${_link}
            PKG_COMMAND ${PACKAGE_CMDS}
            PKG_FILES ${PACKAGE_FILES}
            PKG_ONLY ON
            BRANCH ${_BRANCH}
            VERSION ${_VERSION}
        )
    endforeach()
endif ()
