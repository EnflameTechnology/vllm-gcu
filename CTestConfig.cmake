## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "vllm")
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")


set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "cdash-01.enflame.cn")
set(CTEST_DROP_LOCATION "/submit.php?project=vllm")
set(CTEST_DROP_SITE_CDASH TRUE)

## Set default timeout to 20 minutes
set(DART_TESTING_TIMEOUT 1200)

string(TOLOWER "${CMAKE_C_COMPILER_ID}" COMPILER_STR)
set(BUILDNAME "${COMPILER_STR}-${OUTPUT_DIR}")
#set(MAKECOMMAND "make install -j")

set (BUILDNAME "vllm_checkin")
set (CTEST_LABELS_FOR_SUBPROJECTS "vllm_checkin_label")
#set(CTEST_USE_LAUNCHERS 1)
#set(RULE_LAUNCH_COMPILE )
