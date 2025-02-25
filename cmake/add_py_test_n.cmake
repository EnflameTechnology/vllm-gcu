function (add_py_test_n)
    set(options OPTIONAL OVERRIDE WILLFAIL NO_MONITOR NO_TIMEOUT_CHECK)
    set(oneValueArgs ID
                     NAME
                     TIMEOUT
                     VDK_TIMEOUT
                     EDK_TIMEOUT
                     ARM_TIMEOUT
                     PASS_EXPRESSION
                     SETUP
                     COMMAND
                     CLEANUP
                     WORKING_DIRECTORY
                     ZEBU_DATABASE
                     ENABLE_PROFILER)

    set(multiValueArgs CATEGORY PROJECT
                                PLATFORM
                                REGRESSION
                                OS
                                ARCH
                                PY_VER
                                ENVIRONMENT
                                MODULE)

    cmake_parse_arguments(PARAM "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    #message ("PARAM_OVERRIDE  = ${PARAM_OVERRIDE}")
    #message ("PARAM_WILLFAIL  = ${PARAM_WILLFAIL}")
    #message ("PARAM_CATEGORY  = ${PARAM_CATEGORY}")
    #message ("PARAM_MODULE    = ${PARAM_MODULE}")
    #message ("PARAM_ID        = ${PARAM_ID}")
    #message ("PARAM_NAME      = ${PARAM_NAME}")
    #message ("PARAM_COMMAND   = ${PARAM_COMMAND}")
    #message ("PARAM_PROJECT   = ${PARAM_PROJECT}")
    #message ("PARAM_PLATFORM  = ${PARAM_PLATFORM}")
    ##message ("PARAM_EXTRA_PARAM = ${PARAM_EXTRA_PARAM}")
    ##message ("PARAM_TIMEOUT = ${PARAM_TIMEOUT}")
    #message ("PARAM_REGRESSION= ${PARAM_REGRESSION}")
    #message ("PARAM_OS        = ${PARAM_OS}")
    #message ("PARAM_PY_VER    = ${PARAM_PY_VER}")
    ##message ("PARAM_UNPARSED_ARGUMENTS" = ${PARAM_UNPARSED_ARGUMENTS})
    ##message ("PARAM_TIMEOUT" = ${PARAM_TIMEOUT})

    string (TOLOWER "${PARAM_PROJECT}" PARAM_PROJECT)
    string (TOLOWER "${PARAM_PLATFORM}" PARAM_PLATFORM)
    string (TOLOWER "${PARAM_REGRESSION}" PARAM_REGRESSION)
    string (TOLOWER "${PARAM_CATEGORY}" PARAM_CATEGORY)
    string (TOLOWER "${PARAM_OS}" PARAM_OS)
    string (TOLOWER "${PARAM_ARCH}" PARAM_ARCH)
    string (TOLOWER "${PARAM_MODULE}" PARAM_MODULE)
    string (TOLOWER "${PARAM_ID}" PARAM_ID)
    string (TOLOWER "${PARAM_NAME}" PARAM_NAME)
    string (TOLOWER "${PARAM_PY_VER}" PARAM_PY_VER)

    set (EXPECTED_PROJECT_LIST leo vela pavo dorado scorpio galaxy)
    set (EXPECTED_PLATFORM_LIST vdk vdk1x edk silicon distrib cpu null kvm)
    set (EXPECTED_REGRESSION_LIST ci daily weekly release sanity preci modelzoo null)
    set (EXPECTED_CATEGORY_LIST func perf convergence stability)
    set (EXPECTED_OS_LIST ubuntu tlinux redhat centos ubuntuhost ubuntu1604 ubuntu1804 ubuntu2004 kylin uos euler anolis)
    set (EXPECTED_ARCH_LIST x86_64 aarch64)
    set (EXPECTED_MODULE_LIST
        advanced_settings_topsprof
        cape_of_topsprof
        cape_xapp_bert
        cape_xapp_ocr
        ci
        compatibility_of_topsprof
        control_flow_topsprof
        cpu_profiler_topsprof
        distribution_8card
        driver
        dtu_profiler_topsprof
        dtu_training_for_dorado_topsprof
        dumphlir
        dumphlo
        eccl_16card
        eccl_24card
        eccl_2card
        eccl_48card
        eccl_80card
        eccl_8card
        fmwpy_framework
        fmwpy_kernel
        fmwpy_layers
        fmwpy_ops
        gcu_profiler_pytopsprof
        gcu_profiler_topsprof
        hlir_builder
        hloopv
        inference
        inference_module_fusion
        inference_operator
        inference_optimizer
        inference_python_api
        inference_sample_code
        inference_whole_network
        inference_whole_network_int8
        input_output_topsprof
        internal_of_topsprof_distribution
        internal_of_topsprof_double_16card_gpt2
        internal_only_topsprof
        kvm4c
        mindspore
        mindspore_pytopsprof
        mindspore_topsprof
        model
        model_1280card
        model_160card
        model_16card
        model_16card_esl_direct_transport
        model_24card
        model_2card
        model_2card
        model_320card
        model_32card
        model_40card
        model_48card
        model_640card
        model_80card
        model_8card
        model_idc
        model_int8
        model_king_8card
        model_one_card
        model_pavoking
        model_xring_8card
        model_zoo_c1b1
        model_zoo_c4b1
        model_zoo_c4b16
        model_zoo_c4b2
        model_zoo_c4b4
        network_hlo_add
        network_hlo_bitcast
        network_hlo_bitcast-convert
        network_hlo_constant
        network_hlo_cosine
        network_hlo_get-dimension-size
        network_hlo_log-plus-one
        network_hlo_multiply
        network_hlo_reshape
        network_hlo_sign
        network_hlo_slice
        network_hlo_transpose
        network_hlo_transpose_bacth1
        network_hlo_transpose_bacth2
        network_hlo_transpose_bacth3
        network_hlo_transpose_bacth4
        network_hlo_tuple
        onnx_acc_runtime_int8
        onnx_runtime
        onnx_runtime_int8
        onnxruntime_gcu
        op
        op_benchmark
        op_benchmark_branch1
        op_benchmark_branch2
        op_benchmark_branch3
        op_benchmark_branch4
        op_benchmark_branch5
        op_benchmark_branch6
        op_bn
        op_conv
        op_dma
        op_dma_branch1
        op_dma_branch2
        op_dot
        op_elementwise
        op_elementwise1
        op_elementwise2
        op_others
        op_pengfeng
        op_perf
        opv
        opv_pavoking
        paddle
        paddle_8card
        paddle_native
        paddle_pytopsprof
        paddle_topsprof
        print_info_topsprof
        pytopsprof
        pytorch
        pytorch11
        pytorch11_built-in_autograd
        pytorch11_built-in_module_init
        pytorch11_built-in_modules
        pytorch11_built-in_nn
        pytorch11_built-in_ops
        pytorch11_built-in_reductions
        pytorch11_built-in_shape_ops
        pytorch11_built-in_sort_and_select
        pytorch11_built-in_torch
        pytorch11_built-in_type_promotion
        pytorch11_team5
        pytorch11_fmwpy
        pytorch11_team1
        pytorch11_team2
        pytorch11_team3
        pytorch11_team4
        pytorch_built-in_autograd
        pytorch_built-in_binary_ufuncs
        pytorch_built-in_indexing
        pytorch_built-in_linalg
        pytorch_built-in_module_init
        pytorch_built-in_modules
        pytorch_built-in_nn
        pytorch_built-in_numpy_interop
        pytorch_built-in_ops
        pytorch_built-in_reductions
        pytorch_built-in_reference_numerics
        pytorch_built-in_segment_reductions
        pytorch_built-in_serialization
        pytorch_built-in_shape_ops
        pytorch_built-in_sort_and_select
        pytorch_built-in_spectral_ops
        pytorch_built-in_tensor_creation_ops
        pytorch_built-in_testing
        pytorch_built-in_torch
        pytorch_built-in_type_promotion
        pytorch_built-in_unary_ufuncs
        pytorch_built-in_view_ops
        pytorch_built-in_vmap
        pytorch_team1
        pytorch_team2
        pytorch_team3
        pytorch_team4
        pytorch_team5
        ras
        runtime_benchmark
        sanity_of_pytopsprof
        sanity_of_topsprof
        simplify_hlo_abs
        simplify_hlo_and
        simplify_hlo_batch-norm-grad
        simplify_hlo_batch-norm-inference
        simplify_hlo_batch-norm-training
        simplify_hlo_bitcast
        simplify_hlo_broadcast
        simplify_hlo_clamp
        simplify_hlo_compare
        simplify_hlo_concatenate
        simplify_hlo_convert
        simplify_hlo_copy
        simplify_hlo_custom-call
        simplify_hlo_divide
        simplify_hlo_exponential
        simplify_hlo_floor
        simplify_hlo_gather
        simplify_hlo_get-tuple-element
        simplify_hlo_iota
        simplify_hlo_log
        simplify_hlo_maximum
        simplify_hlo_minimum
        simplify_hlo_negate
        simplify_hlo_not
        simplify_hlo_pad
        simplify_hlo_power
        simplify_hlo_reduce
        simplify_hlo_reduce-window
        simplify_hlo_reverse
        simplify_hlo_rng
        simplify_hlo_rsqrt
        simplify_hlo_scatter
        simplify_hlo_select
        simplify_hlo_sqrt
        simplify_hlo_subtract
        simplify_hlo_tanh
        ssm
        ssmpt10s
        ssmt10
        system_test
        topsdnn
        topsdnn1
        topsdnn2
        topsdnn3
        topsprof
        topsprof_distribution
        topsprof_double_16card_gpt2
        topsprof_multiple_model
        topsprof_single_8card_gpt2
        trace_of_topsprof
        training_model
        training_model_8card
        training_paddle
        vdtu
        xmodel_leo
        xmodel_leo_1card
        xmodel_leo_8card
        xmodel_x31_leo
        xmodel_x31_leo_1card
        xmodel_x31_leo_8card
        zoo
        )

    if ((NOT PARAM_PROJECT) OR ("${PARAM_PROJECT}" STREQUAL "all"))
        set (PARAM_PROJECT ${EXPECTED_PROJECT_LIST})
    endif ()
    foreach (proj IN LISTS PARAM_PROJECT)
        if (NOT proj IN_LIST EXPECTED_PROJECT_LIST)
            message (FATAL_ERROR "Unsupported project type ${proj}. Please specify PROJECT=[${EXPECTED_PROJECT_LIST}]")
        endif ()
    endforeach ()

    if ((NOT PARAM_PLATFORM) OR ("${PARAM_PLATFORM}" STREQUAL "dtu"))
        set (PARAM_PLATFORM vdk vdk1x edk silicon)
    endif ()
    foreach (plat IN LISTS PARAM_PLATFORM)
        if (NOT plat IN_LIST EXPECTED_PLATFORM_LIST)
            message (FATAL_ERROR "Unsupported platform type ${plat}. Please specify PLATS=[${EXPECTED_PLAT_LIST}]")
        endif ()
    endforeach ()

    if (NOT PARAM_REGRESSION)
        set (PARAM_REGRESSION null)
    endif ()
    foreach (regress IN LISTS PARAM_REGRESSION)
        if (NOT regress IN_LIST EXPECTED_REGRESSION_LIST)
            message (FATAL_ERROR "Unsupported regression type ${regress}. Please specify REGRESSION=[${EXPECTED_REGRESSION_LIST}]")
        endif ()
    endforeach ()

    if (NOT PARAM_CATEGORY)
        set (PARAM_CATEGORY func)
    endif ()
    foreach (category IN LISTS PARAM_CATEGORY)
        if (NOT category IN_LIST EXPECTED_CATEGORY_LIST)
            message (FATAL_ERROR "Unsupported category type ${category}. Please specify CATEGORY=[${EXPECTED_CATEGORY_LIST}]")
        endif ()
    endforeach ()

    if (NOT PARAM_OS)
        set (PARAM_OS ubuntu)
    endif ()
    foreach (os IN LISTS PARAM_OS)
        if (NOT os IN_LIST EXPECTED_OS_LIST)
            message (FATAL_ERROR "Unsupported os type ${os}. Please specify OS=[${EXPECTED_OS_LIST}]")
        endif ()
    endforeach ()

    if (NOT PARAM_ARCH)
        set (PARAM_ARCH x86_64)
    endif ()
    foreach (arch IN LISTS PARAM_ARCH)
        if (NOT arch IN_LIST EXPECTED_ARCH_LIST)
            message (FATAL_ERROR "Unsupported arch type ${arch}. Please specify ARCH=[${EXPECTED_ARCH_LIST}]")
        endif ()
    endforeach ()

    foreach (m IN LISTS PARAM_MODULE)
        if (NOT m IN_LIST EXPECTED_MODULE_LIST)
            #message (FATAL_ERROR "Unsupported module type ${m}. Please specify MODULE=[${EXPECTED_MODULE_LIST}]")
            list (REMOVE_DUPLICATES g_modules)
            set (g_modules "${g_modules}" "${m}" CACHE INTERNAL "module list")
            message (WARNING "Unsupported module type ${m}. Please add this new MODULE: ${m} to EXPECTED_MODULE_LIST in 'add_py_test_n.cmake' file")
        endif ()
    endforeach ()

    if (NOT PARAM_COMMAND)
        message (FATAL_ERROR "there is no COMMAND specified, Please define what COMMAND the test will run!")
    endif ()

    if (ENABLE_SANITIZER_PY_ENV)
        set (PARAM_COMMAND "${SANITIZER_ENVS_PY_EXPORT} ${PARAM_COMMAND}")
    endif ()

    set (PARAM_COMMAND "${SANITIZER_ENVS_EXPORT} ${PARAM_COMMAND}")

    # joint Zebu database upload fixture
    if (PARAM_ZEBU_DATABASE)
        set (PARAM_COMMAND "pmon_start ${PARAM_ZEBU_DATABASE}\
                            ; ${PARAM_COMMAND}\
                            ; ret_value=\$?\
                            ; pmon_stop\
                            ; sync\
                            ; pmon_report ${PARAM_ZEBU_DATABASE}")
    endif ()

    # enable profiler
    if (PARAM_ENABLE_PROFILER)
        set(enable_engine "cdma,cqm,sdma,odma,ts,sip")
        if ("${PARAM_PROJECT}" STREQUAL "pavo")
            set(enable_engine "cqm")
        endif ()
        set (PARAM_COMMAND "efsmt -ssm hbm_swizzle off\
                            ; /bin/bash -c \"for bdf in `lspci -d 1ea0:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"\
                            ; /bin/bash -c \"for bdf in `lspci -d 1e36:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"\
                            ; efsmt -ssm hbm_swizzle\
                            ; efsmt -dpm level=80\
                            ; efsmi --ppo off\
                            ; efsmi --ppo status\
                            ; topsprof --force-overwrite --print-app-log --enable-activities '*/TS,ODMA/*|*/*/CDMA,SDMA,CQM,SIP/*|*/general/operator' --buffer host --export-rawdata ${PARAM_NAME}.data --export-visual-profiler ./${PARAM_NAME} bash -c '${PARAM_COMMAND}'\
                            ; ret_value=\$?\
                            ; efsmt -ssm hbm_swizzle on\
                            ; /bin/bash -c \"for bdf in `lspci -d 1ea0:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"\
                            ; /bin/bash -c \"for bdf in `lspci -d 1e36:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"")
    endif ()

    #return saved return value if profiler or zebu databse enabled
    if (PARAM_ENABLE_PROFILER OR PARAM_ZEBU_DATABASE)
        set (PARAM_COMMAND "${PARAM_COMMAND}\
                            ; exit \$ret_value")
    endif ()


    if (PARAM_WORKING_DIRECTORY)
        set(work_dir ${CMAKE_INSTALL_PREFIX}/${PARAM_WORKING_DIRECTORY})
    else ()
        set(work_dir ${CMAKE_INSTALL_PREFIX})
    endif ()

    # new labels
    unset(labels)
    list (APPEND labels "${PARAM_PROJECT}")
    foreach (plt IN LISTS PARAM_PLATFORM)
        list (APPEND labels "p:${plt}")
    endforeach ()
    foreach (rg IN LISTS PARAM_REGRESSION)
        list (APPEND labels "r:${rg}")
    endforeach ()
    foreach (c IN LISTS PARAM_CATEGORY)
        list (APPEND labels "c:${c}")
    endforeach ()
    foreach (os IN LISTS PARAM_OS)
        list (APPEND labels "os:${os}")
    endforeach ()
    foreach (arch IN LISTS PARAM_ARCH)
        list (APPEND labels "a:${arch}")
    endforeach ()
    #list (APPEND labels "${PARAM_MODULE}")
    #string (TOLOWER "${PARAM_ID}" PARAM_ID)
    #string (TOLOWER "${PARAM_NAME}" PARAM_NAME)
    #string (TOLOWER "${PARAM_PY_VER}" PARAM_PY_VER)

    foreach (m IN LISTS PARAM_MODULE)

        set (test_name ${m}_${PARAM_ID}_${PARAM_NAME})

        # if case not exist, add it
        if (NOT TEST ${test_name})
            add_test (NAME ${test_name} COMMAND /bin/bash -c "${PARAM_COMMAND}" WORKING_DIRECTORY ${work_dir})
            # set labels to it
            set_tests_properties (${test_name} PROPERTIES LABELS "${labels}")
            # set cmd so that we can compare if command is same with any new test
            set_tests_properties (${test_name} PROPERTIES CMD "${PARAM_COMMAND}")
        else ()
            get_test_property (${test_name} CMD old_command)

            #
            # if test exist already, then check if command same, if yes, it means there are duplicate case defined.
            #
            if ("${PARAM_COMMAND}" STREQUAL "${old_command}")
                # check if new labels need to append to an existing test
                get_test_property (${test_name} LABELS old_labels)
                foreach (l IN LISTS labels)
                    if (NOT l IN_LIST old_labels)
                        list (APPEND old_labels ${l})
                    endif ()
                endforeach ()
                set_tests_properties (${test_name} PROPERTIES LABELS "${old_labels}")
            else ()
                #
                # if test exist already and test command are NOT same, which means there are cases need to redefine.
                #
                message (WARNING "there is already a test: ${test_name} with command: ${old_command}")
                #message (FATAL_ERROR "there is already a test: ${test_name} with command: ${PARAM_COMMAND}")
                math(EXPR dup_cases "${dup_cases} + 1")
                set (dup_cases "${dup_cases}" CACHE INTERNAL "duplicate case number")
                message (WARNING "duplicate case number: ${dup_cases}")
            endif ()
        endif ()
    endforeach ()
endfunction ()
