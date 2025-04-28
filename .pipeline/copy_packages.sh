#!/bin/bash
# Copyright 2024 Tencent. All Rights Reserved.
#
set -eu -o pipefail
BUILD_ROOT_DIR=`pwd`

#!/bin/bash
ROOT_DIR=`pwd`
echo "Current dir: $ROOT_DIR"
PROJECT_GIT_URL="git@git.tencent.com:sy-zx/enflame_caps"
# Function to pull files using git lfs
pull_from_repo() {
    local repo_name=$1
    local fetch_file_name=$2

    if [ -z "$repo_name" ] || [ -z "$fetch_file_name" ]; then
        echo "Error: Missing arguments. Usage: pull_from_repo <repo_name> <fetch_file_name>"
        return 1
    fi
    cd ${ROOT_DIR}/cmake_build
    if [ -d "${repo_name}" ]; then
        echo "${repo_name} is already exist"
    else
        GIT_LFS_SKIP_SMUDGE=1 git clone ${PROJECT_GIT_URL}/${repo_name}.git
    fi

    echo "Pulling from repo: $repo_name, file: $fetch_file_name"
    cd ${repo_name} && chmod -x ${fetch_file_name} && git lfs pull --include=${fetch_file_name}
    if [ $? -ne 0 ]; then
        echo "Error: Failed to pull file $fetch_file_name from repo $repo_name"
        return 1
    fi
    cd ${ROOT_DIR}/
}

set -x
# find tops_extension
TOPS_EXTENSION_COMMITID=$(grep 'set(TOPS_EXTENSION_COMMIT' ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TOPS_EXTENSION_COMMITID ([a-f0-9]+)\).*/\1/')
echo "Found TOPS_EXTENSION_COMMITID is: $TOPS_EXTENSION_COMMITID"

TOPS_EXTENSION_DAILY_TAG=$(grep "set(TOPS_EXTENSION_DAILY_TAG" ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TOPS_EXTENSION_DAILY_TAG ([0-9.]+)\).*/\1/')
echo "Found TOPS_EXTENSION_DAILY_TAG is: $TOPS_EXTENSION_DAILY_TAG"
TOPS_EXTENSION_310_FILE=${TOPS_EXTENSION_COMMITID}/tops_extension-${TOPS_EXTENSION_DAILY_TAG}+torch.2.5.1-cp310-cp310-linux_x86_64.whl
TOPS_EXTENSION_39_FILE=${TOPS_EXTENSION_COMMITID}/tops_extension_cape-${TOPS_EXTENSION_DAILY_TAG}+torch.2.5.1-cp39-cp39-linux_x86_64.whl

#pull_from_repo "tops_extension_binary" "$TOPS_EXTENSION_310_FILE"
pull_from_repo "tops_extension_binary" "$TOPS_EXTENSION_39_FILE"

cp ${ROOT_DIR}/cmake_build/tops_extension_binary/$TOPS_EXTENSION_310_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/
cp ${ROOT_DIR}/cmake_build/tops_extension_binary/$TOPS_EXTENSION_39_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/

# find xformers
XFORMERS_COMMITID=$(grep 'set(XFORMERS_COMMIT' ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(XFORMERS_COMMITID ([a-f0-9]+)\).*/\1/')
echo "Found XFORMERS_COMMITID is: $XFORMERS_COMMITID"

XFORMERS_DAILY_TAG=$(grep "set(XFORMERS_DAILY_TAG" ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(XFORMERS_DAILY_TAG ([^)]+)\).*/\1/')
echo "Found XFORMERS_DAILY_TAG is: $XFORMERS_DAILY_TAG"

XFORMERS_310_FILE=${XFORMERS_COMMITID}/xformers_cape-${XFORMERS_DAILY_TAG}-cp310-cp310-linux_x86_64.whl
XFORMERS_39_FILE=${XFORMERS_COMMITID}/xformers_cape-${XFORMERS_DAILY_TAG}-cp39-cp39-linux_x86_64.whl

pull_from_repo "xformers_binary" "$XFORMERS_310_FILE"
pull_from_repo "xformers_binary" "$XFORMERS_39_FILE"
cp ${ROOT_DIR}/cmake_build/xformers_binary/$XFORMERS_310_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/
cp ${ROOT_DIR}/cmake_build/xformers_binary/$XFORMERS_39_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/

# find torch_zx_utils
TORCH_ZX_UTILS_COMMITID=$(grep 'set(TORCH_ZX_UTILS_COMMIT' ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TORCH_ZX_UTILS_COMMITID ([a-f0-9]+)\).*/\1/')
echo "Found TORCH_ZX_UTILS_COMMITID is: $TORCH_ZX_UTILS_COMMITID"

TORCH_ZX_UTILS_DAILY_TAG=$(grep "set(TORCH_ZX_UTILS_DAILY_TAG" ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TORCH_ZX_UTILS_DAILY_TAG ([0-9.]+)\).*/\1/')
echo "Found TORCH_ZX_UTILS_DAILY_TAG is: $TORCH_ZX_UTILS_DAILY_TAG"

TORCH_ZX_UTILS_310_FILE=${TORCH_ZX_UTILS_COMMITID}/torch_zx_utils-2.5.1+${TORCH_ZX_UTILS_DAILY_TAG}-cp310-cp310-linux_x86_64.whl
TORCH_ZX_UTILS_39_FILE=${TORCH_ZX_UTILS_COMMITID}/torch_zx_utils-2.5.1+${TORCH_ZX_UTILS_DAILY_TAG}-cp39-cp39-linux_x86_64.whl

pull_from_repo "torch_gcu_binary" "$TORCH_ZX_UTILS_310_FILE"
pull_from_repo "torch_gcu_binary" "$TORCH_ZX_UTILS_39_FILE"
cp ${ROOT_DIR}/cmake_build/torch_gcu_binary/$TORCH_ZX_UTILS_310_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/
cp ${ROOT_DIR}/cmake_build/torch_gcu_binary/$TORCH_ZX_UTILS_39_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/

# find topsaten
TOPSOP_COMMITID=$(grep 'set(TOPSOP_COMMITID' ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TOPSOP_COMMITID "([^"]+)"\).*/\1/')
echo "Found TOPSOP_COMMITID is: $TOPSOP_COMMITID"

TOPSOP_PACKAGE_VERSION=$(grep 'set(TOPSOP_PACKAGE_VERSION' ${ROOT_DIR}/vllm/cmake/fetch_dependences.cmake | sed -E 's/.*set\(TOPSOP_PACKAGE_VERSION "([^"]+)"\).*/\1/')
echo "Found TOPSOP_PACKAGE_VERSION is: $TOPSOP_PACKAGE_VERSION"

TOPSATEN_FILE=${TOPSOP_COMMITID}/topsaten_cape_${TOPSOP_PACKAGE_VERSION}-1_amd64.deb
pull_from_repo "topsaten_binary" "${TOPSATEN_FILE}"

cp ${ROOT_DIR}/cmake_build/topsaten_binary/$TOPSATEN_FILE ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/

cd ${ROOT_DIR}/cmake_build/x86_64-linux-rel/python_packages/
mv tops_extension_cape-${TOPS_EXTENSION_DAILY_TAG}+torch.2.5.1-cp39-cp39-linux_x86_64.whl tops_extension-${TOPS_EXTENSION_DAILY_TAG}+torch.2.5.1-cp39-cp39-linux_x86_64.whl
mv xformers_cape-${XFORMERS_DAILY_TAG}-cp310-cp310-linux_x86_64.whl xformers-${XFORMERS_DAILY_TAG}-cp310-cp310-linux_x86_64.whl
mv xformers_cape-${XFORMERS_DAILY_TAG}-cp39-cp39-linux_x86_64.whl xformers-${XFORMERS_DAILY_TAG}-cp39-cp39-linux_x86_64.whl
mv topsaten_cape_${TOPSOP_PACKAGE_VERSION}-1_amd64.deb topsaten_${TOPSOP_PACKAGE_VERSION}-1_amd64.deb