#########################################################################
#################             common ctest             ##################
#########################################################################
option (LABEL_CTEST "Generating CTest with Labels" OFF)
include (add_py_test_n)
include (add_cc_test_n)


# #############################################################################
# # Remove a compiler flag from a specific build target
# #  _target - The target to remove the compile flag from
# #  _flag   - The compile flag to remove
# #############################################################################
macro(remove_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()

function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()

# #############################################################################
# # Append the 'value' to all variables in ARGN list, if condition is true
# #############################################################################
function(append_if condition value)
    if (${condition})
        foreach(variable ${ARGN})
            set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
            message("variable = ${variable} ${value}")
        endforeach(variable)
    endif()
endfunction()

# #############################################################################
# # Add flag to both CMAKE_[C|C++]_FLAGS
# #############################################################################
macro(add_flag_if_supported flag name)
    check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
    append_if("C_SUPPORTS_${name}" "${flag}" CMAKE_C_FLAGS)
    check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
    append_if("CXX_SUPPORTS_${name}" "${flag}" CMAKE_CXX_FLAGS)
endmacro()

# #############################################################################
# # Add flag to both CMAKE_[C|C++]_FLAGS, if not support, will print a warn msg
# #############################################################################
function(add_flag_or_print_warning flag name)
    check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
    check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
    if (C_SUPPORTS_${name} AND CXX_SUPPORTS_${name})
        message(STATUS "Building with ${flag}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE)
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${flag}" PARENT_SCOPE)
    else()
        message(WARNING "${flag} is not supported.")
    endif()
endfunction()

#########################################################################
#################         SANITIZER: For Tests         ##################
#########################################################################
unset(SANITIZER_ENVS)                   # Common sanitizer envs
unset(SANITIZER_ENVS_EXPORT)            # Common sanitizer export commands, Add for all add_*_test
unset(SANITIZER_ENVS_PY)                # Sanitizer envs for pytest
unset(SANITIZER_ENVS_PY_EXPORT)         # Sanitizer export commands only for pytest, Add for add_py_test when ENABLE_SANITIZER_PY_ENV is on
unset(SANITIZER_ENVS_CC)                # Sanitizer envs for pytest
unset(SANITIZER_ENVS_CC_EXPORT)         # Sanitizer export commands only for gtest, Add for add_cc_test when ENABLE_SANITIZER_CC_ENV is on
set(_SANITIZER_ENVS_LIST_ SANITIZER_ENVS SANITIZER_ENVS_PY SANITIZER_ENVS_CC)

if ("${SANITIZER}" MATCHES "address*")
    set (DTU_CLANG9_TOOLCHAIN_PATH ${default_toolchain_folder})
    set (ASAN_CLANG9_LIB_DIR "${DTU_CLANG9_TOOLCHAIN_PATH}/lib")
    set (ASAN_CLANG9_BIN_DIR "${DTU_CLANG9_TOOLCHAIN_PATH}/bin")
    set (ASAN_CLANG9_LINUX_LIB_DIR "${ASAN_CLANG9_LIB_DIR}/clang/9.0.0/lib/linux")

    if((CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.4) OR (CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.5))
        if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "ppc64le")
            set(lib_path_prefix "powerpc64le")
        else()
            set(lib_path_prefix ${CMAKE_SYSTEM_PROCESSOR})
        endif()
        set(_asan_library /usr/lib/${lib_path_prefix}-linux-gnu/libasan.so.2)
        set(SANITIZER_ENVS_PY "LD_PRELOAD=${_asan_library}" CACHE STRING "")
    endif()
    list(APPEND SANITIZER_ENVS "ASAN_SYMBOLIZER_PATH=`pwd`/bin/llvm-symbolizer")
endif ()

macro (fresh_sanitizer)
    foreach (envs IN LISTS _SANITIZER_ENVS_LIST_)
        if (DEFINED ${envs} AND NOT ("${${envs}}" STREQUAL ""))
            set (_sanitizer_export_ ${${envs}})
            list (TRANSFORM _sanitizer_export_ PREPEND "export ")
            list (JOIN _sanitizer_export_ " && " ${envs}_EXPORT)
            string (APPEND ${envs}_EXPORT " &&")
            set (${envs}_EXPORT "${${envs}_EXPORT}" CACHE STRING "")
            unset (_sanitizer_export_)
        endif ()
    endforeach ()
endmacro ()
fresh_sanitizer()

set (EXPECTED_ARCH_LIST default x86 aarch64)
set (EXPECTED_CATEGORY_LIST func perf convergence stability)
set (EXPECTED_PLATFORM_LIST vdk vdk1x edk silicon distrib cpu null kvm s6 s30 s60 s90)
set (EXPECTED_PROJECT_LIST leo vela pavo dorado scorpio galaxy pavo_galaxy)
set (EXPECTED_REGRESSION_LIST ci daily weekly release sanity preci modelzoo null)
set (EXPECTED_OS_LIST ubuntu tlinux redhat centos ubuntuhost ubuntu1604 ubuntu1804 ubuntu2004 ubuntu2204 kylin uos euler anolis redhat9)

list (SORT EXPECTED_ARCH_LIST)
list (SORT EXPECTED_CATEGORY_LIST)
list (SORT EXPECTED_PLATFORM_LIST)
list (SORT EXPECTED_PROJECT_LIST)
list (SORT EXPECTED_REGRESSION_LIST)
list (SORT EXPECTED_OS_LIST)

set (EXPECTED_ARCH_LIST "${EXPECTED_ARCH_LIST}" CACHE STRING "")
set (EXPECTED_CATEGORY_LIST "${EXPECTED_CATEGORY_LIST}" CACHE STRING "")
set (EXPECTED_PLATFORM_LIST "${EXPECTED_PLATFORM_LIST}" CACHE STRING "")
set (EXPECTED_PROJECT_LIST "${EXPECTED_PROJECT_LIST}" CACHE STRING "")
set (EXPECTED_REGRESSION_LIST "${EXPECTED_REGRESSION_LIST}" CACHE STRING "")
set (EXPECTED_OS_LIST "${EXPECTED_OS_LIST}" CACHE STRING "")

function(_list_to_regex_ regex)
    list(SORT ARGN)
    string(REPLACE ";" ".*" _proj_regrex "${ARGN}")
    string(APPEND _proj_regrex ".*")
    string(PREPEND _proj_regrex ".*")
    set(${regex} "${_proj_regrex}" PARENT_SCOPE)
endfunction()

function (__add_test_)
    set(options OPTIONAL OVERRIDE WILLFAIL NO_MONITOR NO_TIMEOUT_CHECK TEST_W_LABEL DISABLE_HOST_COMPILE)
    set(oneValueArgs ID NAME TIMEOUT VDK_TIMEOUT EDK_TIMEOUT ARM_TIMEOUT PASS_EXPRESSION SETUP COMMAND CLEANUP WORKING_DIRECTORY ZEBU_DATABASE ENABLE_PROFILER)
    set(multiValueArgs CATEGORY PROJECT PLATFORM REGRESSION OS ARCH PY_VER ENVIRONMENT MODULE LABELS)
    cmake_parse_arguments(X_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string (TOLOWER "${X_TEST_MODULE}" X_TEST_MODULE)
    string (TOLOWER "${X_TEST_NAME}" X_TEST_NAME)

    # Adding this list will result in inflexibility, so don't add it
    #set (EXCLUDE_MUDULE_LIST torch_dtu pytorch op backend20 op_conv op_dot op_dma op_bn op_elementwise op_others xla factor model_zoo_c4b1 mlir op_hlir inference_operator inference_whole_network profile runtime scheduler tops inference_mlir inference_python_api util tfop vg cape_xapp_ocr inference-cppapi inference dtupp logging topsprof  )

    if (NOT X_TEST_PROJECT
         OR "${X_TEST_PROJECT}"  STREQUAL "all")
        set (X_TEST_PROJECT ${EXPECTED_PROJECT_LIST})
    else()
        _list_to_regex_(_regex ${X_TEST_PROJECT})
        if(NOT "${EXPECTED_PROJECT_LIST}" MATCHES "${_regex}")
            message(FATAL_ERROR "${X_TEST_PROJECT} SHOULD IN ${EXPECTED_PROJECT_LIST}")
        endif()
    endif ()

    if (NOT X_TEST_PLATFORM
         OR "${X_TEST_PLATFORM}" STREQUAL "dtu")
        set (X_TEST_PLATFORM vdk vdk1x edk silicon)
    else()
        _list_to_regex_(_regex ${X_TEST_PLATFORM})
        if(NOT "${EXPECTED_PLATFORM_LIST}" MATCHES "${_regex}")
            message(FATAL_ERROR "${X_TEST_PLATFORM} SHOULD IN ${EXPECTED_PLATFORM_LIST}")
        endif()
    endif ()

    if (NOT X_TEST_REGRESSION)
        set (X_TEST_REGRESSION null)
    # elseif (CI_TEST_ONLY)
    #     if("ci" IN_LIST X_TEST_REGRESSION)
    #         set(X_TEST_REGRESSION "ci")
    #     endif()
    else()
        _list_to_regex_(_regex ${X_TEST_REGRESSION})
        if(NOT "${EXPECTED_REGRESSION_LIST}" MATCHES "${_regex}")
            message(FATAL_ERROR "${X_TEST_REGRESSION} SHOULD IN ${EXPECTED_REGRESSION_LIST}")
        endif()
    endif ()

    if (NOT X_TEST_CATEGORY)
        set (X_TEST_CATEGORY func)
    else()
        _list_to_regex_(_regex ${X_TEST_CATEGORY})
        if(NOT "${EXPECTED_CATEGORY_LIST}" MATCHES "${_regex}")
            message(FATAL_ERROR "${X_TEST_CATEGORY} SHOULD IN ${EXPECTED_CATEGORY_LIST}")
        endif()
    endif ()

    if (NOT X_TEST_OS)
        set (X_TEST_OS ubuntu)
    else()
        _list_to_regex_(_regex ${X_TEST_OS})
        if(NOT "${EXPECTED_OS_LIST}" MATCHES "${_regex}")
            message(FATAL_ERROR "${X_TEST_OS} SHOULD IN ${EXPECTED_OS_LIST}")
        endif()
    endif ()

    if (NOT X_TEST_ARCH)
        set (X_TEST_ARCH default)
    else ()
        list (APPEND X_TEST_ARCH default)
        _list_to_regex_(_regex ${X_TEST_ARCH})
        if(NOT "${EXPECTED_ARCH_LIST}" MATCHES "${_regex}")
            message(FATAL_ERROR "${X_TEST_ARCH} SHOULD IN ${EXPECTED_ARCH_LIST}")
        endif()
    endif ()

    if (NOT X_TEST_COMMAND)
        set (X_TEST_COMMAND test)
    endif ()
    if(X_TEST_TIMEOUT)
        set(_TIME_OUT_DEFAULT ${X_TEST_TIMEOUT})
    endif()
    if(NOT X_TEST_EDK_TIMEOUT)
        set(X_TEST_EDK_TIMEOUT 240000)
    endif()
    if(NOT X_TEST_VDK_TIMEOUT)
        set(X_TEST_VDK_TIMEOUT 240000)
    endif()
    if(NOT X_TEST_ARM_TIMEOUT)
        set(X_TEST_ARM_TIMEOUT 3600)
    endif()

    # WORKING_DIRECTORY select condition branch
    if (X_TEST_WORKING_DIRECTORY)
        set (workdir ${CMAKE_INSTALL_PREFIX}/${X_TEST_WORKING_DIRECTORY})
    else ()
        set (workdir ${CMAKE_INSTALL_PREFIX})
    endif ()
    
    # joint Zebu database upload fixture
    if (X_TEST_ZEBU_DATABASE)
        set (X_TEST_COMMAND "pmon_start ${X_TEST_ZEBU_DATABASE}\
                            ; ${X_TEST_COMMAND}\
                            ; ret_value=\$?\
                            ; pmon_stop\
                            ; sync\
                            ; pmon_report ${X_TEST_ZEBU_DATABASE}")
        set(bash_cmd /bin/bash -c -i)
    else()
        set(bash_cmd /bin/bash -c)
    endif ()
    
    unset (ENV)
    # support runtime 3.0
    # list(APPEND X_TEST_ENVIRONMENT "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/tops/lib")
    foreach (env IN LISTS X_TEST_ENVIRONMENT)
        string(APPEND ENV "${env};")
    endforeach ()

    if (X_TEST_SETUP AND X_TEST_CLEANUP)
        set(_FIXTURES_REQUIRED "${X_TEST_SETUP};${X_TEST_CLEANUP}")
    elseif (X_TEST_SETUP)
        set(_FIXTURES_REQUIRED "${X_TEST_SETUP}")
    elseif (X_TEST_CLEANUP)
        set(_FIXTURES_REQUIRED "${X_TEST_CLEANUP}")
    endif ()    
    
    unset(X_TEST_ORGIN_COMMAND)
    set(X_TEST_ORGIN_COMMAND "${X_TEST_COMMAND}")

    foreach (proj IN LISTS X_TEST_PROJECT)
        # enable profiler
        if (X_TEST_ENABLE_PROFILER)
            if ("${proj}" STREQUAL "pavo")
                set(enable_engine "cqm")
            else()
                set(enable_engine "cdma,cqm,sdma,odma,ts,sip")
            endif ()
            if ("${proj}" STREQUAL "scorpio")
                set(X_TEST_COMMAND "topsprof --force-overwrite --print-app-log --enable-activities all --buffer device --export-rawdata ${X_TEST_NAME}.data --export-visual-profiler ./${X_TEST_NAME} bash -c '${X_TEST_ORGIN_COMMAND}'")
            else ()
                set(X_TEST_COMMAND "efsmt -dpm level=80\
                                ; efsmi --ppo off\
                                ; efsmi --ppo status\
                                ; efsmt -clock list \
                                ; topsprof --reset\
                                ; topsprof --force-overwrite --print-app-log --enable-activities '*/TS,ODMA/*|*/*/CDMA,CQM,SIP/*|*/general/operator' --buffer host --export-rawdata ${X_TEST_NAME}.data --export-visual-profiler ./${X_TEST_NAME} bash -c '${X_TEST_ORGIN_COMMAND}'\
                                ; ret_value=\$?\
                                ; /bin/bash -c \"for bdf in `lspci -d 1ea0:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"\
                                ; /bin/bash -c \"for bdf in `lspci -d 1e36:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"")
            endif ()
        endif ()
        #return saved return value if profiler or zebu databse enabled
        if (X_TEST_ENABLE_PROFILER OR X_TEST_ZEBU_DATABASE)
            set (X_TEST_COMMAND "${X_TEST_COMMAND}\
                                ; exit \$ret_value")
        endif ()

        foreach (plat IN LISTS X_TEST_PLATFORM)
            if ("${plat}" STREQUAL "edk")
                set(_timeout_plat ${X_TEST_EDK_TIMEOUT})
            elseif ("${plat}" MATCHES "vdk")
                set(_timeout_plat ${X_TEST_VDK_TIMEOUT})
            else()
                set(_timeout_plat ${_TIME_OUT_DEFAULT})
            endif ()

            foreach (regress IN LISTS X_TEST_REGRESSION)
                foreach (arch IN LISTS X_TEST_ARCH)
                    if ("${arch}" STREQUAL "aarch64")
                        set(_timeout ${X_TEST_ARM_TIMEOUT})
                    else()
                        set(_timeout ${_timeout_plat})
                    endif ()

                    foreach (os IN LISTS X_TEST_OS)
                        foreach (category IN LISTS X_TEST_CATEGORY)
                            foreach(module IN LISTS X_TEST_MODULE)
                                if ("${arch}" STREQUAL "default")
                                    # default name not contain ${arch} for CI
                                    set (TEST_NAME ${proj}_${plat}_${regress}_${category}_${os}_${module}_${X_TEST_ID}_${X_TEST_NAME})
                                else ()
                                    set (TEST_NAME ${proj}_${plat}_${regress}_${category}_${os}_${arch}_${module}_${X_TEST_ID}_${X_TEST_NAME})
                                endif ()
                                add_test (NAME ${TEST_NAME} COMMAND ${bash_cmd} "${X_TEST_COMMAND}" WORKING_DIRECTORY ${workdir})
                                file (APPEND ${CMAKE_BINARY_DIR}/sqlite.txt "\"${proj}\",\"${plat}\", \"${regress}\", \",${category}\", \"${os}\", \"${arch}\", \"${module}\", \"${X_TEST_ID}\", \"${X_TEST_NAME}\", \"${TEST_NAME}\", \"${X_TEST_COMMAND}\", \"${workdir}\", \"${_timeout}\"\n")
                                if(ENV)
                                    set_tests_properties (${TEST_NAME} PROPERTIES ENVIRONMENT "${ENV}")
                                endif()
                                if (X_TEST_PASS_EXPRESSION)
                                    set_tests_properties (${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION "${X_TEST_PASS_EXPRESSION}")
                                #else ()
                                #    set_tests_properties (${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 2)
                                endif ()
                                if (_timeout)
                                    set_tests_properties (${TEST_NAME} PROPERTIES TIMEOUT ${_timeout})
                                endif ()
                                if (X_TEST_WILLFAIL)
                                    set_tests_properties (${TEST_NAME} PROPERTIES WILL_FAIL true)
                                endif ()
                                if(_FIXTURES_REQUIRED)
                                    set_tests_properties(${proj}${suite}_${X_TEST_ID}${X_TEST_NAME} PROPERTIES FIXTURES_REQUIRED "${_FIXTURES_REQUIRED}")
                                endif()

                                if (X_TEST_UNPARSED_ARGUMENTS)
                                    message (FATAL_ERROR "${X_TEST_UNPARSED_ARGUMENTS} - parameters are not supported!")
                                endif ()
                                if (X_TEST_LABELS)
                                    unset(LABELS_SET)
                                    foreach (label IN LISTS X_TEST_LABELS)
                                        string(APPEND LABELS_SET "${label};")
                                    endforeach ()
                                    set_tests_properties (${TEST_NAME} PROPERTIES LABELS "${LABELS_SET}")
                                endif ()
                            endforeach ()
                        endforeach ()
                    endforeach ()
                endforeach ()
            endforeach ()
        endforeach ()
    endforeach ()
endfunction ()



#
# Add CTest Function
#
# ADD_SW_TEST PARAM:
# add_sw_test(
#     ID                 test_ID, e.g. GDMA.1.1
#     NAME               test_name
#     [PROJECT]          leo|vela|pavo|dorado|scorpio|all, test could run on which project, default is all
#     [PLATFORM]         vdk|vdk1x|edk|silicon|distri|cpu|dtu, test could run on which platform, dtu=vdkedksilicon, default is dtu
#     [REGRESSION]       ci|daily|weekly|release|preci|modelzoo, test will run in which regression set, default is daily
#     [CATEGORY]         func|robust|perf, test belongs to which category, default is func
#     [OS]               u16|u20|c7, test will run with which OS, u16=ubuntu16.04, c7=centOS 7
#     COMMAND            test command line
#     [PY_VER]           py27|py35|py36|py37, test will run with which version of python, py27=python2.7
#     [EXTRA_PARAM]      additional parameter for efvs e.g. "-v debug"
#     [TIMEOUT]          time_in_second, set timeout to override default value(1200s)
#     [VDK_TIMEOUT]      time_in_second, set timeout to override default value(240000s)
#     [EDK_TIMEOUT]      time_in_second, set timeout to override default value(240000s)
#     [PASS_EXPRESSION]  "pass regular expression in log", change the pass/fail criteria to parse log
#     [SETUP]            set up fixtures
#     [CLEANUP]          clean up fixtures
#     [OVERRIDE]         override default parameter with EXTRA_PARAM
#     [WILLFAIL]         test ctest expect result is fail (return code > 0)
#     [NO_MONITOR]       per test request do not enable any system monitor in parallel
#     [NO_TIMEOUT_CHECK] normally slt test need to be less than 120s, this option is used for special case which over 120s
# )
#

function (add_py_test)
    # set(options OPTIONAL OVERRIDE WILLFAIL NO_MONITOR NO_TIMEOUT_CHECK TEST_W_LABEL)
    # set(oneValueArgs ID NAME TIMEOUT VDK_TIMEOUT EDK_TIMEOUT ARM_TIMEOUT PASS_EXPRESSION SETUP COMMAND CLEANUP WORKING_DIRECTORY ZEBU_DATABASE ENABLE_PROFILER)
    # set(multiValueArgs CATEGORY PROJECT PLATFORM REGRESSION OS ARCH PY_VER ENVIRONMENT MODULE LABELS)

    # cmake_parse_arguments(X_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cmake_parse_arguments(X_TEST "" "COMMAND" "" ${ARGN})

    # for runtime 3.0
    list(APPEND X_TEST_ENVIRONMENT "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/tops/lib")

    # generating ctest with label instead of combo
    if (X_TEST_TEST_W_LABEL OR TEST_W_LABEL)
        add_py_test_n(${ARGN})
        return()
    endif ()

    # if (ENABLE_SANITIZER_PY_ENV)
    #     set (X_TEST_COMMAND "${SANITIZER_ENVS_PY_EXPORT} ${X_TEST_COMMAND}")
    # endif ()
    set(py_module_reg [[python[0-9\.]*[ \t\r\n\]+-m[ \t\r\n]+pytest]])
    set(pytest_reg [[pytest]])

    string(REGEX MATCH ${py_module_reg} _match_cmd "${X_TEST_COMMAND}")
    if(NOT _match_cmd)
        string(REGEX MATCH ${pytest_reg} _match_cmd "${X_TEST_COMMAND}")
    endif()
    if(_match_cmd)
        set(_sanitizer_env "${SANITIZER_ENVS_EXPORT} ${SANITIZER_ENVS_PY_EXPORT}")
        string(REPLACE "`pwd`" "$_test_prefix" _sanitizer_env "${_sanitizer_env}")
        string(REPLACE "${_match_cmd}" "${_sanitizer_env} ${_match_cmd}" X_TEST_COMMAND "${X_TEST_COMMAND}")
        set (X_TEST_COMMAND "_test_prefix=`pwd` && ${X_TEST_COMMAND}")
    else()
        set (X_TEST_COMMAND "${SANITIZER_ENVS_PY_EXPORT} ${X_TEST_COMMAND}")
        set (X_TEST_COMMAND "${SANITIZER_ENVS_EXPORT} ${X_TEST_COMMAND}")
    endif()

    __add_test_(${ARGN} COMMAND ${X_TEST_COMMAND})
endfunction ()

function (add_cc_test)
    # set(options OPTIONAL OVERRIDE WILLFAIL NO_MONITOR NO_TIMEOUT_CHECK TEST_W_LABEL)
    # set(oneValueArgs ID NAME TIMEOUT VDK_TIMEOUT EDK_TIMEOUT ARM_TIMEOUT PASS_EXPRESSION SETUP COMMAND CLEANUP WORKING_DIRECTORY ZEBU_DATABASE ENABLE_PROFILER)
    # set(multiValueArgs CATEGORY PROJECT PLATFORM REGRESSION OS ARCH PY_VER ENVIRONMENT MODULE LABELS)

    # cmake_parse_arguments(X_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cmake_parse_arguments(X_TEST "" "COMMAND" "" ${ARGN})

    # for runtime 3.0
    list(APPEND X_TEST_ENVIRONMENT "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/tops/lib")
    # generating ctest with label instead of combo
    if (X_TEST_TEST_W_LABEL OR TEST_W_LABEL)
        add_cc_test_n(${ARGN})
        return()
    endif ()

    if (ENABLE_SANITIZER_CC_ENV)
        set (X_TEST_COMMAND "${SANITIZER_ENVS_CC_EXPORT} ${X_TEST_COMMAND}")
    endif ()
    set (X_TEST_COMMAND "${SANITIZER_ENVS_EXPORT} ${X_TEST_COMMAND}")

    __add_test_(${ARGN} COMMAND ${X_TEST_COMMAND})
endfunction ()

###
### This function is to add a submodule folder into CMake Build System.
### It will check if submodule init or not before calling add_subdirectory()
### Param1: submodule foler name
### Param2: tagfile to check if submodule exists
###
function (add_submodule_directory dir)
    if(${ARGC} GREATER 1)
        set (file_to_check ${ARGV1})
    else ()
        set (file_to_check CMakeLists.txt)
    endif ()

    set(tagfile ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${file_to_check})
    if (NOT EXISTS ${tagfile})
        message (WARNING "tagfile ${tagfile} doesn't exist, it seems submodule not init yet")
        message (FATAL_ERROR "Seems submodule '${dir}' not init, you MAY run following command to init:\n 'cd ${CMAKE_SOURCE_DIR} && git submodule update --init --recursive'\n")
    else ()
        add_subdirectory(${dir})
    endif ()
endfunction ()

option (DISABLE_PIP_CONFIG_CHECK "disable the ~/.pip/pip.conf file check" OFF)

macro (python_pip_config)
    if (NOT DISABLE_PIP_CONFIG_CHECK)
        if (NOT EXISTS $ENV{HOME}/.pip/pip.conf)
            message (WARNING "There is no local pip config file for better build performance, please setup as following:
######################################################################################
$ENV{HOME}/.pip/pip.conf:
\[global\]
  index-url = http://artifact.enflame.cn/artifactory/api/pypi/pypi-remote/simple
\[install\]
  trusted-host = artifact.enflame.cn
######################################################################################"
        )
            message (FATAL_ERROR "please setup python pip.conf for build performance")
        endif ()
    endif ()
endmacro()
