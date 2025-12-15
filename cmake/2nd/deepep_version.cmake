
set(DEEPEP_COMMITID "9d7158e")
set(DEEPEP_PATH module_package/pcals)
set(DEEPEP_BRANCH "master")
set(DEEPEP_PACKAGE_VERSION "1.0.20251120")

############################################################
##########           DeepEP from DEEPEP           ###########
############################################################
set(deepep_py_versions "3.10" "3.12")
set(deepep_torch_versions "2.8.0")
foreach(torch_version IN LISTS deepep_torch_versions)
    string(REPLACE "." "" torch_version_ext "${torch_version}")
    foreach(py_version IN LISTS deepep_py_versions)
        string(REPLACE "." "" py_version_ext "${py_version}")
        unset(DEEPEP_LINK)
        set(DEEPEP_LINK "${DEEPEP_PATH}/${DEEPEP_COMMITID}/deep_ep-1.2.1+torch.${torch_version}.gcu.${DEEPEP_PACKAGE_VERSION}-cp${py_version_ext}-cp${py_version_ext}-linux_x86_64.whl")
        fetchFromArtifactory(deepep_whl_t${torch_version}_${py_version}
            FILE ${DEEPEP_LINK}
            PKG_COMMAND ${PACKAGE_PYTHON_CMDS}
            PKG_FILES ${PACKAGE_PYTHON_FILES}
            BRANCH ${DEEPEP_BRANCH}
            VERSION ${DEEPEP_PACKAGE_VERSION}
            PKG_ONLY ON
        )
    endforeach()
endforeach()
