#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
BUILD_ROOT_DIR=`pwd`
export TORCH_VERSION=${torch_gcu_version:-"2.6.0"}

function arm_normal_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  sudo python3.12 -m pip install torch==2.5.1
  cmake ${project_name} --preset ci_all -B cmake_build
  cd cmake_build
  ninja -j${cpu_count} install
  ninja -j${cpu_count} package_all
  epkg get -s buildtree --check-unique --check-missing normal_arm_all_buildtree
}

function x86_normal_ci_build() {
  x86_normal_build
}

function x86_normal_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  sudo python3.12 -m pip install --index-url http://data-oceanus.enflame.cn/artifactory/api/pypi/pypi_virtual/simple --trusted-host data-oceanus.enflame.cn torch==$TORCH_VERSION+cpu patch pyyaml packaging
  cmake ${project_name} --preset ci_all -B cmake_build
  cd cmake_build
  ninja -j${cpu_count} install
  ninja -j${cpu_count} package_all
  epkg get -s buildtree --check-unique --check-missing normal_all_buildtree
}

function x86_normal_daily_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  sudo python3.12 -m pip install --index-url http://data-oceanus.enflame.cn/artifactory/api/pypi/pypi_virtual/simple --trusted-host data-oceanus.enflame.cn torch==$TORCH_VERSION+cpu patch pyyaml packaging
  cmake ${project_name} --preset ci_all -B cmake_build -DNEED_DAILY_TEST_CASE=TRUE
  cd cmake_build
  ninja -j${cpu_count} install
  ninja -j${cpu_count} package_all
  epkg get -s buildtree --check-unique --check-missing normal_all_buildtree
}

function main() {
  set -x
  $build_job_name
}

if [ $# -eq 0 ]; then
  echo "First argument build job name is empty"
  exit 1
elif [ $# -eq 1 ]; then
  build_job_name=$1
fi

export project_name=${project_name:-"vllm"}
export cpu_count=${process_num:-$(nproc)}
export PY_PACKAGE_VERSION=${package_version:-""}
export MAX_JOBS=${cpu_count}

main "$@"
exit $?
