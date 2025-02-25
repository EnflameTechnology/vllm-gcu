#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
BUILD_ROOT_DIR=`pwd`

function x86_normal_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  cmake ${project_name} --preset ci_all -B cmake_build
  cd cmake_build
  ninja -j32 install
  ninja -j32 package_all
}

function x86_normal_daily_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  cd ${project_name}
  NEED_DAILY_TEST_CASE=ON VLLM_TARGET_DEVICE=gcu USE_CI=1 $PY_EXECUTOR setup.py bdist_wheel --py-limited-api=${VLLM_ABI_FLAG} -d ../cmake_build/x86_64-linux-rel/python_packages
}

function tar_test_files(){
  cd ${BUILD_ROOT_DIR}/${project_name}
  COMMIT_ID=$(git rev-parse --short=7 HEAD)
  tar zcf vllm_gcu_test_file_${COMMIT_ID}.tgz tests OOT_models
  mv vllm_gcu_test_file_${COMMIT_ID}.tgz ${BUILD_ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/
}

function cape_build() {
  x86_normal_build
}

function main() {
  set -x
  $build_job_name
  cd ${BUILD_ROOT_DIR}/cmake_build/
  epkg get -s buildtree --check-unique --check-missing normal_all_buildtree

}

if [ $# -eq 0 ]; then
  echo "First argument build job name is empty"
  exit 1
elif [ $# -eq 1 ]; then
  build_job_name=$1
fi

export project_name=${project_name:-"vllm"}
export PY_PACKAGE_VERSION=${package_version:-""}

main "$@"
exit $?
