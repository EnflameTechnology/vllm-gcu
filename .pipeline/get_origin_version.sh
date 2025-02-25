#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#

set -eu -o pipefail
BUILD_ROOT_DIR=`pwd`

cd vllm
version_file="requirements.txt"
if [ ! -f "$version_file" ]; then
  echo "Error: Version file not found at $version_file"
  exit 1
fi

version_line=$(grep -E "vllm==([0-9]+\\.[0-9]+\\.[0-9]+[^ ]*)" "$version_file")

if [[ -n "$version_line" ]]; then
  version=$(echo "$version_line" | sed -E "s/^vllm==([0-9]+\\.[0-9]+\\.[0-9]+[^ ]*)/\1/")
  echo "$version"
else
  echo "Error: VERSION not found in file"
  exit 1
fi