cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

#download i-cmake
include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)
set(icmake_REPOSITORY "git@git.enflame.cn:i")
if(PROJECT_GIT_URL)
    set(icmake_REPOSITORY ${PROJECT_GIT_URL})
endif()
set(icmake_GIT_REVISION 1.5.73)
FetchContent_Declare(
    I_CMAKE
    GIT_REPOSITORY ${icmake_REPOSITORY}/cmake.git
    GIT_TAG ${icmake_GIT_REVISION}
)
FetchContent_MakeAvailable(I_CMAKE)
set(CMAKE_MODULE_DIR "${i_cmake_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_MODULE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(DISABLE_GIT_SUBMODULE_CHECK ON)
include(module_paths)

project(vllm_gcu CXX C)
include(fetchArtifactory)
include(artifacts)
include(global)
include(ictest)

# set(LOGGING_TAG 4.0.25)
# FetchContent_Declare(sw_enflame.open_logging
#     GIT_REPOSITORY git@git.enflame.cn:sw/enflame.open/logging.git
#     GIT_TAG ${LOGGING_TAG}
#     EXCLUDE_FROM_ALL)
# FetchContent_MakeAvailable(sw_enflame.open_logging)

set(CMAKE_FPKG_PYTHON_PACKAGES "python_packages")
set(CMAKE_INSTALL_PYTHON_PACKAGES "${CMAKE_INSTALL_PREFIX}/${CMAKE_FPKG_PYTHON_PACKAGES}")
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PYTHON_PACKAGES})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}ci/module_packages.json" "{}")
include(package_version)
option(NEED_DAILY_TEST_CASE "enable daily test case" FALSE)
# init vllm version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt" FILE_CONTENT)
string(REGEX MATCH "vllm==([0-9]+\\.[0-9]+\\.[0-9]+[^ ]*)" VERSION_MATCH "${FILE_CONTENT}")
set(VLLM_ORIGIN_VERSION ${CMAKE_MATCH_1})
string(REPLACE "\n" "" VLLM_ORIGIN_VERSION "${VLLM_ORIGIN_VERSION}")
message(STATUS "Found VLLM_ORIGIN VERSION: ${VLLM_ORIGIN_VERSION}")

set(PACKAGE_VERSION ${VLLM_ORIGIN_VERSION}+${PACKAGE_VERSION})

include(fetch_dependences)

if (NOT TARGET package_all)
    add_custom_target(package_all)
endif()
if(NOT PROJECT_GIT_URL)
    set(py_ver "3.12")
    set(cp_ver "312")
else()
    set(py_ver "3.10")
    set(cp_ver "310")
endif()
set(py_ext "39")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/dist/vllm_gcu-${PACKAGE_VERSION}-cp${py_ext}-abi3-linux_${CMAKE_SYSTEM_PROCESSOR}.whl
    COMMAND sudo python${py_ver} -m pip install "${tops_extension_${cp_ver}_whl_FILE}" pyyaml
    COMMAND export PY_PACKAGE_VERSION=${PACKAGE_VERSION} && export TOPSRT_HOME=${TOPSRT_HOME} && export TOPSATEN_HOME=${TOPSATEN_HOME} && export TORCH_GCU_PATH=${torch_gcu_${cp_ver}_whl_SOURCE_DIR}/torch_gcu && python${py_ver} setup.py bdist_wheel --py-limited-api=cp${py_ext}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    )

add_custom_target(
    package_vllm_gcu_whl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dist/vllm_gcu-${PACKAGE_VERSION}-cp${py_ext}-abi3-linux_${CMAKE_SYSTEM_PROCESSOR}.whl
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/dist/vllm_gcu-${PACKAGE_VERSION}-cp${py_ext}-abi3-linux_${CMAKE_SYSTEM_PROCESSOR}.whl
    ${CMAKE_INSTALL_PYTHON_PACKAGES}
    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/dist ${CMAKE_CURRENT_SOURCE_DIR}/build ${CMAKE_CURRENT_SOURCE_DIR}/vllm_gcu.egg-info
)
add_dependencies(package_all package_vllm_gcu_whl)

install(DIRECTORY .epkg DESTINATION ${CMAKE_BINARY_DIR})
install(DIRECTORY tests DESTINATION .)

install(FILES cmake/env_config.json DESTINATION .)
install(FILES cmake/package_info.json DESTINATION .)

install (PROGRAMS ${i_cmake_SOURCE_DIR}/tools/setup_platform.py DESTINATION .)
install (PROGRAMS ${i_cmake_SOURCE_DIR}/tools/setup_platform.sh DESTINATION .)
install (DIRECTORY ${i_cmake_SOURCE_DIR}/tools/ DESTINATION . USE_SOURCE_PERMISSIONS)
configure_file(ctest_install.cmake.in ctest_install.cmake @ONLY)
install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/ctest_install.cmake)
install(FILES "${CMAKE_INSTALL_PREFIX}/CTestTestfile.cmake" DESTINATION .)

file(GLOB test_files_script CONFIGURE_DEPENDS ${CMAKE_MODULE_DIR}/run_test/*)

install(PROGRAMS ${test_files_script} DESTINATION .)
install_module_package()

install(CODE "
        if (EXISTS ${CMAKE_BINARY_DIR}/DartConfiguration.tcl)
            execute_process (COMMAND cp -af ${CMAKE_BINARY_DIR}/DartConfiguration.tcl ${CMAKE_BINARY_DIR}/.DartConfiguration.tcl)
            execute_process (COMMAND sed -i s!${CMAKE_SOURCE_DIR}!.!g ${CMAKE_BINARY_DIR}/DartConfiguration.tcl)
            execute_process (COMMAND sed -i s!${CMAKE_BINARY_DIR}!.!g ${CMAKE_BINARY_DIR}/DartConfiguration.tcl)
            execute_process (COMMAND cp -af ${CMAKE_BINARY_DIR}/DartConfiguration.tcl ${CMAKE_INSTALL_PREFIX}/DartConfiguration.tcl)
        endif ()
        ")
add_subdirectory(tests)

if(NEED_DAILY_TEST_CASE)
    add_subdirectory(tests/daily_test/vllm/kernels)
endif()
