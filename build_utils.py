import os
import subprocess
from pathlib import Path

UNKNOWN = "unknown"

def get_tag(base_dir, tops_version) -> str:
    PACKAGE_VERSION = os.getenv('PACKAGE_VERSION', default='')
    if PACKAGE_VERSION == '' or PACKAGE_VERSION == '123.456':
        PACKAGE_VERSION = tops_version + "." + subprocess.check_output(
            ["git", "show", "-s", "--date=format:'%Y%m%d'", "--format=%cd"],
            cwd=base_dir).decode("ascii").strip().replace("'", "")
    return PACKAGE_VERSION

def get_tops_version(version_file_path):
    tops_version = UNKNOWN
    with open(version_file_path, 'r') as file:
        tops_version = file.read().strip()
    return tops_version
