# ######################################################
# #########  2nd MODULE PACKAGES GLOBAL SET  ###########
# ######################################################
set(MODULE_PACKAGE_PATH module_package)
set(MODULE_PACKAGE_DOWN_MODE FILE)
set (ARTIFACTS_LOCAL_CACHE_LOCATION "/home/.cache/tops")
# set (ARTIFACTS_LOCAL_CACHE_LOCATION "/home/.cache/") # test
set (ARTIFACTORY_TOPS_SERVER "http://artifact.enflame.cn/artifactory")
set (ARTIFACTORY_TOPS_PROJECT "tops")
set (ARTIFACTORY_BASE_URL "${ARTIFACTORY_TOPS_SERVER}/${ARTIFACTORY_TOPS_PROJECT}")
set (ARTIFACTORY_BROWSE_URL "${ARTIFACTORY_TOPS_SERVER}/webapp/#/artifacts/browse/tree/General/${ARTIFACTORY_TOPS_PROJECT}")

# ######################################################
# ##############  USE PREBUILD PKG  ####################
# ######################################################
# set(PREBUILD_TOPS_EXTENSION_XNAS_BASE "http://xnas.enflame.cn/release/topsop_release_build/310/integration/939f2e7")

# ######################################################
# ############  GET PREBUILD PKG NAME ##################
# ######################################################
macro(getNASPackageName)
    set (oneValueArgs SOURCE_URL PKG_NAME PKG_TYPE)
    set (options)
    set (multiValueArgs)
    cmake_parse_arguments (""
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    execute_process(
        COMMAND curl -L ${_SOURCE_URL}/
        COMMAND grep -oP "${_PKG_NAME}_[0-9].*?.${_PKG_TYPE}"
        COMMAND sort -u
        OUTPUT_VARIABLE ${_PKG_NAME}_pkg_name
        RESULT_VARIABLE error_code
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    unset(_SOURCE_URL)
    unset(_PKG_TYPE)
endmacro()

##########################################
########## fetchFromArtifactory ##########
##########################################

# Server for fetchFromArtifactory
set (ARTIFACTORY_SERVER ${ARTIFACTORY_TOPS_SERVER})

#[[
    ARTIFACTORY_TOKEN_FOLDER, default is ${CMAKE_SOURCE_DIR}/.token/
        Store all token files, which is same as uri second root folder name. Case sensitive. Example:
            uri: http://artifact.enflame.cn:80/artifactory/module_package/  -->  module_package
    ARTIFACTORY_FATAL_TOKEN, default is TRUE.
        Treat token file error as FATAL_ERROR or WARNING.
        You can disable it if anonymous asscess is available,
    ARTIFACTORY_FETCH_CACHE_DIR, project cache path, default is ${CMAKE_BINARY_DIR}/.cache.
        Download file for incremental build, if host cache misses.
]]
if(NOT ARTIFACTORY_TOKEN_FOLDER)
    set(ARTIFACTORY_TOKEN_FOLDER "${CMAKE_SOURCE_DIR}/.token/")
endif()
option(ARTIFACTORY_FATAL_TOKEN "FATAL ERROR IF TOKEN is illegal" TRUE)

# Read token files from the folder, which have no suffix.
if(NOT ARTIFACTORY_FETCH_CACHE_DIR)
    set(ARTIFACTORY_FETCH_CACHE_DIR ${CMAKE_BINARY_DIR}/.cache)
endif()

set(PREBUILD_CAPS_COMMIT 6b94f53)
set(PREBUILD_CAPS_DATE 20250306)
set(PREBUILD_CAPS_VERSION_BIG 1.3.5.7)
set(CAPS_BRANCH integration_20250220)
include(caps_binary)
set(TOPSRT_HOME "${runtime_install_usr_dir_for_run}")
message(STATUS "TOPSRT_HOME : ${TOPSRT_HOME}")

set(BUILD_TORCH_VERSION "2.5.1")
# ######################################################
# ###################  TOPS_EXTENSION  #########################
# ######################################################

set(TOPS_EXTENSION_PATH ${MODULE_PACKAGE_PATH}/tops_extension)
set(TOPS_EXTENSION_COMMITID 6819677)
set(TOPS_EXTENSION_BRANCH master)
set(TOPS_EXTENSION_DAILY_TAG 3.2.20250310)
set(TOPS_EXTENSION_PY_VER 310)
set(TOPS_EXTENSION_SEMI_NAME "")

unset(TOPS_EXTENSION_LINK)
set(tops_extension_link "${TOPS_EXTENSION_COMMITID}/tops_extension${TOPS_EXTENSION_SEMI_NAME}-${TOPS_EXTENSION_DAILY_TAG}+torch.${BUILD_TORCH_VERSION}-cp${TOPS_EXTENSION_PY_VER}-cp${TOPS_EXTENSION_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl")

set(CMAKE_FPKG_PYTHON_PACKAGES python_packages)
set(CMAKE_FPKG_LIBDIR lib)
set(CMAKE_INSTALL_PYTHON_PACKAGES ${CMAKE_INSTALL_PREFIX}/${CMAKE_FPKG_PYTHON_PACKAGES})
set(CMAKE_INSTALL_LIB ${CMAKE_INSTALL_PREFIX}/${CMAKE_FPKG_LIBDIR})
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PYTHON_PACKAGES} ${CMAKE_INSTALL_LIB})

set(PACKAGE_PYTHON_CMDS "mkdir -p ${CMAKE_FPKG_PYTHON_PACKAGES}; mv /FILE/ -t ${CMAKE_FPKG_PYTHON_PACKAGES}")
set(PACKAGE_PYTHON_FILES "${CMAKE_FPKG_PYTHON_PACKAGES}//FILE/")
set(PACKAGE_LIB_CMDS "mkdir -p ${CMAKE_FPKG_LIBDIR}; mv /FILE/ -t ${CMAKE_FPKG_LIBDIR}")
set(PACKAGE_LIB_FILES "${CMAKE_FPKG_LIBDIR}//FILE/")

fetchFromArtifactory(tops_extension_whl
    FILE ${TOPS_EXTENSION_PATH}/${tops_extension_link}
    PKG_COMMNAD ${PACKAGE_PYTHON_CMDS}
    PKG_FILES ${PACKAGE_PYTHON_FILES}
    BRANCH ${TOPS_EXTENSION_BRANCH}
    VERSION ${TOPS_EXTENSION_TORCH_DAILY_TAG}
)

set(TOPSOP_PATH module_package/topsop)
set(TOPSOP_DOWN_MODE FILE)
set(TOPSOP_SEMI_NAME "")
set(TOPSOP_BRANCH "master")
set(TOPSOP_COMMITID "145ce0c")
set(TOPSOP_PACKAGE_VERSION "3.3.20250310")
# set(TOPSOP_XNAS_LINK "http://10.12.110.200:8080/release/topsop_release_build/390/integration/f008b06/")

if (NOT DEFINED TOPSATEN_INSTALL_PREFIX)
    if(TOPSOP_XNAS_LINK)
        link_pattern_var("${TOPSOP_XNAS_LINK}"
            VARS
                TOPSATEN_LINK
            PATTERNS
                "topsaten${TOPSOP_SEMI_NAME}_[0-9].*.deb"
        )
        if(NOT TOPSATEN_LINK)
            message(WARNING "Can not find some links from ${TOPSOP_XNAS_LINK}")
        endif()
    endif()

    if(TOPSATEN_LINK)
        set(TOPSATEN_LINK_CMD URI ${TOPSATEN_LINK})
    else()
        set(TOPSATEN_LINK_CMD ${TOPSOP_DOWN_MODE} ${TOPSOP_PATH}/${TOPSOP_COMMITID}/topsaten${TOPSOP_SEMI_NAME}_${TOPSOP_PACKAGE_VERSION}-1_${_DEB_PACKAGE_ARCHITECTURE}.deb)
    endif()
    fetchFromArtifactory(fetch_topsaten_deb
        ${TOPSATEN_LINK_CMD}
        PKG_COMMAND ${PACKAGE_LIB_CMDS}
        PKG_FILES ${PACKAGE_LIB_FILES}
        BRANCH ${TOPSOP_BRANCH}
        VERSION ${TOPSOP_PACKAGE_VERSION}
        EXTRACT ON
    )
    set(TOPSATEN_HOME "${fetch_topsaten_deb_SOURCE_DIR}/usr")
    message(STATUS "TOPSATEN_HOME : ${TOPSATEN_HOME}")
endif()
####################################################
###############    torch_gcu     ###################
####################################################

set(TORCH_GCU_PATH ${MODULE_PACKAGE_PATH}/torch_gcu)
set(TORCH_GCU_COMMITID dc3a3ed)
set(TORCH_GCU_BRANCH 2.x)
set(TORCH_GCU_DAILY_TAG 3.3.1.1)
set(TORCH_GCU_PY_VER 310)
set(TORCH_GCU_SEMI_NAME "")

# set(TORCH_GCU_XNAS_LINK "http://10.12.110.200:8080/release/torch-gcu-release/57/integration/efda744/")

unset(TORCH_GCU_LINK)
if(TORCH_GCU_XNAS_LINK)
    link_pattern_var("${TORCH_GCU_XNAS_LINK}"
        VARS
            TORCH_GCU_LINK
        PATTERNS
            "torch_gcu${TORCH_GCU_SEMI_NAME}-${BUILD_TORCH_VERSION}.*-cp${TORCH_GCU_PY_VER}-cp${TORCH_GCU_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl"
    )
    message(STATUS "TORCH_GCU_LINK: ${TORCH_GCU_LINK}")
    if(NOT TORCH_GCU_LINK)
        message(WARNING "Can not find some links from ${TORCH_GCU_XNAS_LINK}")
    endif()
endif()
set(torch_gcu_link "${TORCH_GCU_COMMITID}/torch_gcu${TORCH_GCU_SEMI_NAME}-${BUILD_TORCH_VERSION}+${TORCH_GCU_DAILY_TAG}-cp${TORCH_GCU_PY_VER}-cp${TORCH_GCU_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl")
if(NOT TORCH_GCU_LINK)
    set(TORCH_GCU_LINK ${TORCH_GCU_PATH}/${torch_gcu_link})
endif()
fetchFromArtifactory(torch_gcu_whl
    FILE ${TORCH_GCU_LINK}
    PKG_COMMAND ${PACKAGE_PYTHON_CMDS}
    PKG_FILES ${PACKAGE_PYTHON_FILES}
    BRANCH ${TORCH_GCU_BRANCH}
    VERSION ${TORCH_GCU_DAILY_TAG}
    EXTRACT ON
)

# ######################################################
# ###################  XFORMERS  #########################
# ######################################################

set(XFORMERS_PATH ${MODULE_PACKAGE_PATH}/xformers)
set(XFORMERS_COMMITID 773583e)
set(XFORMERS_BRANCH 0.0.28.post3)
set(XFORMERS_DAILY_TAG 0.0.28.post3+torch.2.5.1.gcu.3.2.20250225)
set(XFORMERS_PY_VER 310)
set(XFORMERS_SEMI_NAME "")

if (PROJECT_GIT_URL)
    set(xformers_git_name "xformers_binary")
    download_tx_git_project(${xformers_git_name})
endif()
unset(XFORMERS_LINK)
if(PREBUILD_XFORMERS_XNAS_BASE)
    link_pattern_var("${PREBUILD_XFORMERS_XNAS_BASE}"
        VARS
            XFORMERS_LINK
        PATTERNS
            "xformers-.*-cp${XFORMERS_PY_VER}-cp${XFORMERS_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl"
    )
    message(STATUS "XFORMERS_LINK: ${XFORMERS_LINK}, XFORMERS_TEST_LINK: ${XFORMERS_TEST_LINK}")
    if(NOT XFORMERS_LINK)
        message(WARNING "Can not find some links from ${PREBUILD_XFORMERS_XNAS_BASE}")
    endif()
endif()
set(xformers_link "${XFORMERS_COMMITID}/xformers${XFORMERS_SEMI_NAME}-${XFORMERS_DAILY_TAG}-cp${XFORMERS_PY_VER}-cp${XFORMERS_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl")
if(NOT XFORMERS_LINK)
    set(XFORMERS_LINK ${XFORMERS_PATH}/${xformers_link})
endif()

if (NOT PROJECT_GIT_URL)
    fetchFromArtifactory(xformers_whl
        FILE ${XFORMERS_LINK}
        PKG_COMMNAD ${PACKAGE_PYTHON_CMDS}
        PKG_FILES ${PACKAGE_PYTHON_FILES}
        BRANCH ${XFORMERS_BRANCH}
        VERSION ${XFORMERS_DAILY_TAG}
        PKG_ONLY ON
    )
else()
    set(git_name "xformers_binary")
    string(REGEX REPLACE "(.*)/(.*)" "\\2" xformers_binary_file_name "${XFORMERS_${BUILD_TORCH_VERSION}_LINK}")
    set(fetch_file_name ${XFORMERS_COMMITID}/${xformers_binary_file_name})

    download_tx_whl(${xformers_git_name} ${fetch_file_name})

    set(xformers_${XFORMERS_PY_VER}_whl_${CMAKE_SYSTEM_PROCESSOR}_FILE ${CMAKE_CURRENT_BINARY_DIR}/${xformers_git_name}/${XFORMERS_COMMITID}/${xformers_binary_file_name})

endif()
