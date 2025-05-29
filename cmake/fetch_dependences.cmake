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

function(download_tx_project repo_name fetch_file_name)
    execute_process(
        COMMAND bash -c "cd ${repo_name} && git lfs pull --include=${fetch_file_name}"
        RESULT_VARIABLE res_val
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    if(res_val)
        message(FATAL_ERROR "can't get lfs: cd ${repo_name} && git lfs pull --include=${fetch_file_name}")
    endif()
endfunction()

# function(download_tx_tar repo_name fetch_file_name)
#     message("---begin download ${repo_name}: load file ${fetch_file_name}---")
#     download_tx_project(${repo_name} ${fetch_file_name})
#     set(tar_name ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}/${fetch_file_name})
#     file(ARCHIVE_EXTRACT INPUT "${tar_name}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${git_name}")
#     message("---${repo_name} source dir: ${CMAKE_CURRENT_BINARY_DIR}/${git_name}---")
# endfunction()

function(download_tx_whl repo_name fetch_file_name)
    message("---begin download ${repo_name}: load file ${fetch_file_name}---")
    download_tx_project(${repo_name} ${fetch_file_name})
    if (fetch_file_name MATCHES "_cape")
        string(REGEX REPLACE "_cape" "" fetch_output_file_name "${fetch_file_name}")
        if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${repo_name}/${fetch_output_file_name}")
            execute_process(
                COMMAND mv "${fetch_file_name}" "${fetch_output_file_name}"
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}
            )
            message("---Renamed ${fetch_file_name} to ${fetch_output_file_name}---")
            set(fetch_file_name ${fetch_output_file_name})
        endif()
    else()
        message("---No '_cape' found in ${fetch_file_name}, skipping rename.---")
    endif()
    
    execute_process(
        COMMAND unzip -o ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}/${fetch_file_name} -d ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}
        RESULT_VARIABLE res_val
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    if(res_val)
        message(FATAL_ERROR "FAILED: unzip -o ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}/${fetch_file_name} -d ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}")
    endif()

endfunction()

function(download_tx_deb  repo_name fetch_file_name)
    message("---begin download ${repo_name}: load file ${fetch_file_name}---")
    download_tx_project(${repo_name} ${fetch_file_name})
    set(deb_name ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}/${fetch_file_name})

    execute_process(
        COMMAND dpkg-deb --extract ${deb_name} ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}
        RESULT_VARIABLE res_val
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    if(res_val)
        message(FATAL_ERROR "FAILED: dpkg-deb --extract ${deb_name} ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}")
    endif()

    message("---${repo_name} source dir: ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}---")
endfunction()

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

include(${PROJECT_SOURCE_DIR}/cmake/2nd/caps_version.cmake)
include(caps_binary)
set(TOPSRT_HOME "${runtime_install_usr_dir_for_run}")
message(STATUS "TOPSRT_HOME : ${TOPSRT_HOME}")

if( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(BUILD_TORCH_VERSION "2.6.0")
else()
    set(BUILD_TORCH_VERSION "2.5.1")
endif()

set(CMAKE_FPKG_PYTHON_PACKAGES python_packages)
set(CMAKE_FPKG_LIBDIR lib)
set(CMAKE_INSTALL_PYTHON_PACKAGES ${CMAKE_INSTALL_PREFIX}/${CMAKE_FPKG_PYTHON_PACKAGES})
set(CMAKE_INSTALL_LIB ${CMAKE_INSTALL_PREFIX}/${CMAKE_FPKG_LIBDIR})
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PYTHON_PACKAGES} ${CMAKE_INSTALL_LIB})

set(PACKAGE_PYTHON_CMDS "mkdir -p ${CMAKE_FPKG_PYTHON_PACKAGES}; mv /FILE/ -t ${CMAKE_FPKG_PYTHON_PACKAGES}")
set(PACKAGE_PYTHON_FILES "${CMAKE_FPKG_PYTHON_PACKAGES}//FILE/")
set(PACKAGE_LIB_CMDS "mkdir -p ${CMAKE_FPKG_LIBDIR}; mv /FILE/ -t ${CMAKE_FPKG_LIBDIR}")
set(PACKAGE_LIB_FILES "${CMAKE_FPKG_LIBDIR}//FILE/")
# ######################################################
# ###################  TOPS_EXTENSION  #################
# ######################################################
set(TOPS_EXTENSION_PATH ${MODULE_PACKAGE_PATH}/tops_extension)
set(TOPS_EXTENSION_COMMITID 886d6d6)
set(TOPS_EXTENSION_BRANCH master)
set(TOPS_EXTENSION_DAILY_TAG 3.2.20250507)
set(TOPS_EXTENSION_PY_VERS 310 312)
set(TOPS_EXTENSION_SEMI_NAME "")
foreach(TOPS_EXTENSION_PY_VER IN LISTS TOPS_EXTENSION_PY_VERS)
    unset(TOPS_EXTENSION_${TOPS_EXTENSION_PY_VER}_LINK)
    set(tops_extension_${TOPS_EXTENSION_PY_VER}_link "${TOPS_EXTENSION_COMMITID}/tops_extension${TOPS_EXTENSION_SEMI_NAME}-${TOPS_EXTENSION_DAILY_TAG}+torch.${BUILD_TORCH_VERSION}-cp${TOPS_EXTENSION_PY_VER}-cp${TOPS_EXTENSION_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl")

    if(NOT PROJECT_GIT_URL)
        fetchFromArtifactory(tops_extension_${TOPS_EXTENSION_PY_VER}_whl
            FILE ${TOPS_EXTENSION_PATH}/${tops_extension_${TOPS_EXTENSION_PY_VER}_link}
            PKG_COMMNAD ${PACKAGE_PYTHON_CMDS}
            PKG_FILES ${PACKAGE_PYTHON_FILES}
            BRANCH ${TOPS_EXTENSION_BRANCH}
            VERSION ${TOPS_EXTENSION_TORCH_DAILY_TAG}
        )
    else()
        set(tops_extension_git_name "tops_extension_binary")
        download_tx_whl(${tops_extension_git_name} ${tops_extension_${TOPS_EXTENSION_PY_VER}_link})
        set(tops_extension_${TOPS_EXTENSION_PY_VER}_whl_FILE ${CMAKE_CURRENT_BINARY_DIR}/${tops_extension_git_name}/${tops_extension_${TOPS_EXTENSION_PY_VER}_link})
        if (tops_extension_${TOPS_EXTENSION_PY_VER}_whl_FILE MATCHES "_cape")
            string(REGEX REPLACE "_cape" "" tops_extension_${TOPS_EXTENSION_PY_VER}_whl_FILE "${tops_extension_${TOPS_EXTENSION_PY_VER}_whl_FILE}")
        endif()
        set(tops_extension_${TOPS_EXTENSION_PY_VER}_whl_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${tops_extension_git_name})
    endif()
endforeach()
# ######################################################
# ###################  TOPSATEN  #######################
# ######################################################
set(TOPSOP_PATH module_package/topsop)
set(TOPSOP_DOWN_MODE FILE)
set(TOPSOP_SEMI_NAME "")
include(${PROJECT_SOURCE_DIR}/cmake/2nd/topsop_version.cmake)
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
    if (NOT PROJECT_GIT_URL)
        fetchFromArtifactory(fetch_topsaten_deb
            ${TOPSATEN_LINK_CMD}
            PKG_COMMAND ${PACKAGE_LIB_CMDS}
            PKG_FILES ${PACKAGE_LIB_FILES}
            BRANCH ${TOPSOP_BRANCH}
            VERSION ${TOPSOP_PACKAGE_VERSION}
            EXTRACT ON
        )
        set(TOPSATEN_HOME "${fetch_topsaten_deb_SOURCE_DIR}/usr")
    else()
        set(topsop_git_name "topsaten_binary")
        string(REGEX REPLACE "(.*)/(.*)" "\\2" topsaten_binary_file_name "${TOPSATEN_LINK_CMD}")
        set(fetch_file_name ${TOPSOP_COMMITID}/${topsaten_binary_file_name})

        download_tx_deb(${topsop_git_name} ${fetch_file_name})

        set(fetch_topsaten_deb_FILE ${CMAKE_CURRENT_BINARY_DIR}/${topsop_git_name}/${TOPSOP_COMMITID}/${topsaten_binary_file_name})
        set(TOPSATEN_HOME ${CMAKE_CURRENT_BINARY_DIR}/${topsop_git_name}/usr)
    endif()
    message(STATUS "TOPSATEN_HOME : ${TOPSATEN_HOME}")
endif()
####################################################
###############    torch_gcu     ###################
####################################################

set(TORCH_GCU_PATH ${MODULE_PACKAGE_PATH}/torch_gcu)
include(${PROJECT_SOURCE_DIR}/cmake/2nd/torch_gcu_version.cmake)
set(TORCH_GCU_BRANCH 2.x)
set(TORCH_GCU_PY_VERS 310 312)

# set(TORCH_GCU_XNAS_LINK "http://10.12.110.200:8080/release/torch-gcu-release/57/integration/efda744/")
foreach(TORCH_GCU_PY_VER IN LISTS TORCH_GCU_PY_VERS)
    unset(TORCH_GCU_${TORCH_GCU_PY_VER}_LINK)
    if(TORCH_GCU_XNAS_LINK)
        link_pattern_var("${TORCH_GCU_XNAS_LINK}"
            VARS
                TORCH_GCU_${TORCH_GCU_PY_VER}_LINK
            PATTERNS
                "torch_gcu-${BUILD_TORCH_VERSION}.*-cp${TORCH_GCU_PY_VER}-cp${TORCH_GCU_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl"
        )
        message(STATUS "TORCH_GCU_${TORCH_GCU_PY_VER}_LINK: ${TORCH_GCU_${TORCH_GCU_PY_VER}_LINK}")
        if(NOT TORCH_GCU_${TORCH_GCU_PY_VER}_LINK)
            message(WARNING "Can not find some links from ${TORCH_GCU_XNAS_LINK}")
        endif()
    endif()
    set(torch_gcu_${TORCH_GCU_PY_VER}_link "${TORCH_GCU_COMMITID}/torch_gcu-${TORCH_GCU_DAILY_TAG}-cp${TORCH_GCU_PY_VER}-cp${TORCH_GCU_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl")
    if(NOT TORCH_GCU_${TORCH_GCU_PY_VER}_LINK)
        set(TORCH_GCU_${TORCH_GCU_PY_VER}_LINK ${TORCH_GCU_PATH}/${torch_gcu_${TORCH_GCU_PY_VER}_link})
    endif()
    if(NOT PROJECT_GIT_URL)
        fetchFromArtifactory(torch_gcu_${TORCH_GCU_PY_VER}_whl
            FILE ${TORCH_GCU_${TORCH_GCU_PY_VER}_LINK}
            PKG_COMMAND ${PACKAGE_PYTHON_CMDS}
            PKG_FILES ${PACKAGE_PYTHON_FILES}
            BRANCH ${TORCH_GCU_BRANCH}
            VERSION ${TORCH_GCU_DAILY_TAG}
            EXTRACT ON
        )
    else()
        set(torch_gcu_git_name "torch_gcu_binary")

        download_tx_whl(${torch_gcu_git_name} ${torch_gcu_${TORCH_GCU_PY_VER}_link})

        set(torch_gcu_${TORCH_GCU_PY_VER}_whl_FILE ${CMAKE_CURRENT_BINARY_DIR}/${torch_gcu_git_name}/${torch_gcu_${TORCH_GCU_PY_VER}_link})
        set(torch_gcu_${TORCH_GCU_PY_VER}_whl_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${torch_gcu_git_name})
    endif()
endforeach()

# ######################################################
# ###################  XFORMERS  #######################
# ######################################################
set(XFORMERS_TORCH_0.0.29.post2.2.6.0_COMMITID 0736a34)
set(XFORMERS_PATH ${MODULE_PACKAGE_PATH}/xformers)
set(XFORMERS_COMMITID 0736a34)
set(XFORMERS_BRANCH 0.0.29.post2)
set(XFORMERS_DAILY_TAG 0.0.29.post2+torch.2.6.0.gcu.3.2.20250427)
set(XFORMERS_PY_VERS 310 312)
set(XFORMERS_SEMI_NAME "")
if( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  foreach(XFORMERS_PY_VER IN LISTS XFORMERS_PY_VERS)
      unset(XFORMERS_${XFORMERS_PY_VER}_LINK)
      if(PREBUILD_XFORMERS_XNAS_BASE)
          link_pattern_var("${PREBUILD_XFORMERS_XNAS_BASE}"
              VARS
                  XFORMERS_${XFORMERS_PY_VER}_LINK
              PATTERNS
                  "xformers-.*-cp${XFORMERS_PY_VER}-cp${XFORMERS_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl"
          )
          message(STATUS "XFORMERS_${XFORMERS_PY_VER}_LINK: ${XFORMERS_${XFORMERS_PY_VER}_LINK}, XFORMERS_TEST_LINK: ${XFORMERS_TEST_LINK}")
          if(NOT XFORMERS_${XFORMERS_PY_VER}_LINK)
              message(WARNING "Can not find some links from ${PREBUILD_XFORMERS_XNAS_BASE}")
          endif()
      endif()
      set(xformers_${XFORMERS_PY_VER}_link "${XFORMERS_COMMITID}/xformers${XFORMERS_SEMI_NAME}-${XFORMERS_DAILY_TAG}-cp${XFORMERS_PY_VER}-cp${XFORMERS_PY_VER}-linux_${CMAKE_SYSTEM_PROCESSOR}.whl")
      if(NOT XFORMERS_${XFORMERS_PY_VER}_LINK)
          set(XFORMERS_${XFORMERS_PY_VER}_LINK ${XFORMERS_PATH}/${xformers_${XFORMERS_PY_VER}_link})
      endif()

      if (NOT PROJECT_GIT_URL)
          fetchFromArtifactory(xformers_${XFORMERS_PY_VER}_whl
              FILE ${XFORMERS_${XFORMERS_PY_VER}_LINK}
              PKG_COMMNAD ${PACKAGE_PYTHON_CMDS}
              PKG_FILES ${PACKAGE_PYTHON_FILES}
              BRANCH ${XFORMERS_BRANCH}
              VERSION ${XFORMERS_DAILY_TAG}
              PKG_ONLY ON
          )
      else()
          message("--- don't download xformers for zx build---")
      endif()
  endforeach()
endif()

include(${PROJECT_SOURCE_DIR}/cmake/2nd/topsgraph_version.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/2nd/pcals_version.cmake)

# set(PREBUILD_XNAS_SDK_BASE "http://artifact.enflame.cn/artifactory/module_package/topsfactor/13ee374/")
set(PREBUILD_FACTOR_COMMIT a2916e9)
set(PREBUILD_FACTOR_VERSION 3.4.20250506)
include(factor_binary)

# set(PREBUILD_XNAS_SDK_BASE "http://artifact.enflame.cn/artifactory/module_package/topsfactor/13ee374/")
set(PREBUILD_SDK_COMMIT a2916e9)
set(PREBUILD_SDK_VERSION 3.4.20250506)
include(sdk_binary)
