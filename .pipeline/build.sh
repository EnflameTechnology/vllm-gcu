#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
BUILD_ROOT_DIR=$(pwd)
set -x
ARCH=$(uname -m)

function ci_build() {
  cd ${project_name}
  python3 setup.py bdist_wheel 
}

function main() {
  $build_job_name
}

export project_name=${project_name:-"vllm-gcu"}
export cpu=${process_num:-"10"}

if [ "$#" -eq 1 ]; then
  build_job_name=$1
else
  echo "donot support this build job"
  exit 1
fi

main "$@"
exit $?