#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
BUILD_ROOT_DIR=`pwd`

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

function x86_normal_build() {
  
  sudo python3.12 -m pip install --index-url http://artifact.enflame.cn/artifactory/api/pypi/pypi-remote/simple --trusted-host artifact.enflame.cn patch pyyaml packaging
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  sudo python3.12 -m pip install torch==2.6.0+cpu --index-url http://data-oceanus.enflame.cn/artifactory/api/pypi/pypi_virtual/simple --trusted-host data-oceanus.enflame.cn
  cmake ${project_name} --preset ci_all -B cmake_build
  cd cmake_build
  ninja -j${cpu_count} install
  ninja -j${cpu_count} package_all
  epkg get -s buildtree --check-unique --check-missing normal_all_buildtree
}

function x86_normal_daily_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  cmake ${project_name} --preset ci_all -B cmake_build
  cd cmake_build
  ninja -j${cpu_count} install
  ninja -j${cpu_count} package_all
  epkg get -s buildtree --check-unique --check-missing normal_all_buildtree

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

function zx_normal_build() {
  echo "Current build job: $FUNCNAME"
  echo `pwd`
  PROJECT_GIT_URL="git@git.tencent.com:sy-zx/enflame_caps"
  sudo python3.10 -m pip install torch==2.6.0 patch pyyaml packaging numpy==1.22.4
  rm -rf cmake_build
  mkdir -p ${BUILD_ROOT_DIR}/cmake_build/ci
  cd ${BUILD_ROOT_DIR}/cmake_build
  touch ${BUILD_ROOT_DIR}/cmake_build/ci/module_packages.json
  if [ -d "tops_extension_binary" ]; then
    echo "tops_extension_binary is already exist"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/tops_extension_binary.git
  fi
  if [ -d "torch_gcu_binary" ]; then
    echo "torch_gcu_binary is already exist"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/torch_gcu_binary.git
  fi
  if [ -d "topsaten_binary" ]; then
    echo "topsaten_binary is already exist"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/topsaten_binary.git
  fi
  if [ -d "caps_binary" ]; then
    echo "caps_binary is already exist"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/caps_binary.git
  fi
  cd -
  cmake vllm --preset ci_all -B cmake_build -DPROJECT_GIT_URL=${PROJECT_GIT_URL}
  cd ${BUILD_ROOT_DIR}/cmake_build
  ninja -j4 install
  ninja -j4 package_all
  cd ${BUILD_ROOT_DIR}
  ./vllm/.pipeline/copy_packages.sh
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
