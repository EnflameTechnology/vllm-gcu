#
# Ensure helper scripts are available in the build tree after compilation.
#
# `monitor_gcu.sh` is used for runtime monitoring during test/debug runs; CMake
# does not automatically copy shell scripts into the build directory unless we
# ask it to.
set(_monitor_gcu_sh "${CMAKE_CURRENT_LIST_DIR}/monitor_gcu.sh")
if(EXISTS "${_monitor_gcu_sh}")
  set(_monitor_gcu_sh_out "${CMAKE_CURRENT_BINARY_DIR}/monitor_gcu.sh")

  add_custom_command(
    OUTPUT "${_monitor_gcu_sh_out}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_monitor_gcu_sh}" "${_monitor_gcu_sh_out}"
    DEPENDS "${_monitor_gcu_sh}"
    VERBATIM
  )

  # Make sure the script is present after a normal build.
  add_custom_target(monitor_gcu_script ALL DEPENDS "${_monitor_gcu_sh_out}")

  # Also ship it with installation/package artifacts.
  install(PROGRAMS "${_monitor_gcu_sh}" DESTINATION tests)
endif()

include(ctest_scorpio.cmake)
include(ctest_libra.cmake)
