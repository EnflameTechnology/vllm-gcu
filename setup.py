#!/usr/bin/env python
# coding=utf-8
import glob
import io
import os
import re
import shutil
from distutils.command.clean import clean
from typing import List

import torch

from setuptools import Extension, find_packages, setup
# from tops_extension import (
#     TOPSATEN_HOME as _TOPSATEN_HOME,
#     TopsBuildExtension,
#     TOPSRT_HOME as _TOPSRT_HOME,
# )
from tops_extension import TopsBuildExtension
from tops_extension.torch import TopsTorchExtension
from tops_extension.torch.codegen_utils import gen_custom_ops
from wheel.bdist_wheel import bdist_wheel
from build_utils import get_tag, get_tops_version

ROOT_DIR = os.path.dirname(__file__)

try:
    from tops_extension import TOPSATEN_HOME as _TOPSATEN_HOME
except ImportError:
    _TOPSATEN_HOME = os.getenv("TOPSATEN_HOME", None)

try:
    from tops_extension import TOPSRT_HOME as _TOPSRT_HOME
except ImportError:
    _TOPSRT_HOME = os.getenv("TOPSRT_HOME", None)

if os.getenv("PY_PACKAGE_VERSION"):
    VERSION = os.getenv("PY_PACKAGE_VERSION")
else:
    import vllm

    VLLM_VERSION = vllm.__version__
    tops_version = get_tops_version(f"{BASE_DIR}/.version")
    VERSION = f"{VLLM_VERSION}+{get_tag(BASE_DIR, tops_version)}"

try:
    import tops_extension

    _TOPS_EXTENSION_PATH = tops_extension.__path__[0]
except ImportError:
    _TOPS_EXTENSION_PATH = os.getenv("TOPS_EXTENSION_PATH", None)

try:
    import torch_gcu

    _TORCH_GCU_PATH = torch_gcu.__path__[0]
except ImportError:
    _TORCH_GCU_PATH = os.getenv("TORCH_GCU_PATH", None)

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
# Compiler flags.
CXX_FLAGS = [
    "-g",
    "-O2",
    "-std=c++17",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Wno-write-strings",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
]
TOPSCC_FLAGS = ["-std=c++17", "-Wno-unused-result", f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_library_path(library_name):

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    UN_KNOWN_VERSION = "Unknown"

    command = ["python{}".format(python_version), "-m", "pip", "show", library_name]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"Error occurred when running pip show {library_name}:")
        print(result.stderr)
        return UN_KNOWN_VERSION

    for line in result.stdout.split("\n"):
        if line.startswith("Location:"):
            path = f"{line.split(': ')[1].strip()}/{library_name}"

            return path
    return UN_KNOWN_VERSION


def read_readme() -> str:
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()

    return ""


def read_requirements():
    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


ext_modules = []


class VllmBuildExtension(TopsBuildExtension):
    def build_extension(self, ext: Extension) -> None:
        yaml_files = list(filter(lambda f: f.endswith(".yaml"), ext.sources))
        for yaml_file in yaml_files:
            src_dir, filename = os.path.split(yaml_file)
            gen_custom_ops(
                custom_src_dir=src_dir,
                namespace="vllm_gcu::llm_ops",
                python_output_dir=None,
                include_dir=src_dir,
            )
        ext.sources = list(filter(lambda f: not f.endswith(".yaml"), ext.sources))
        return TopsBuildExtension.build_extension(self, ext)


class VllmBdistWheel(bdist_wheel):
    def initialize_options(self):
        bdist_wheel.initialize_options(self)
        self.py_limited_api = "cp38"


class VllmClean(clean):
    def run(self):
        if os.path.exists(".gitignore"):
            with open(".gitignore", "r") as f_ignore:
                ignores = f_ignore.read()
            pattern = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pattern.match(wildcard)
                if match:
                    if match.group(1):
                        break
                else:
                    for filename in glob.glob(wildcard):
                        shutil.rmtree(filename, ignore_errors=True)

        clean.run(self)

        try:
            import torchgen
            import torchgen.gen

            src_path = os.path.join(ROOT_DIR, "csrc", "src")
            custom_functions = torchgen.gen.parse_native_yaml(
                os.path.join(src_path, "gcu_custom_functions.yaml"),
                os.path.join(
                    torchgen.__path__[0], "packaged", "ATen", "native", "tags.yaml"
                ),
            ).native_functions

            for fn in custom_functions:
                header = os.path.join(src_path, f"{fn.root_name}.h")
                if os.path.exists(header):
                    os.remove(header)
        except ImportError:
            pass


def _get_all_sources(path: str):
    src_path = os.path.join(ROOT_DIR, path)
    assert os.path.exists(src_path), f"{src_path} not exists"

    patterns = ["/**/*.cpp", "/**/*.cc", "/**/*.tops", "/**/gcu_custom_functions.yaml"]
    return [
        f for pattern in patterns for f in glob.glob(path + pattern, recursive=True)
    ]


def _get_include_and_library_dirs():
    assert _TOPS_EXTENSION_PATH, "tops extension path must be set"
    assert _TOPSATEN_HOME, "topsaten is not installed"
    assert _TORCH_GCU_PATH, "torch_gcu is not installed"

    include_dirs = library_dirs = []

    include_dirs.append(os.path.join(_TOPS_EXTENSION_PATH, "include"))
    include_dirs.append(os.path.join(_TOPS_EXTENSION_PATH, "include", "tops_extension"))
    include_dirs.append(os.path.join(_TOPSATEN_HOME, "include", "gcu"))
    include_dirs.append(os.path.join(_TORCH_GCU_PATH, "include"))

    library_dirs.append(os.path.join(_TOPS_EXTENSION_PATH, "lib"))
    library_dirs.append(os.path.join(_TOPSATEN_HOME, "lib"))
    library_dirs.append(os.path.join(_TORCH_GCU_PATH, "lib"))

    return {"include_dirs": include_dirs, "library_dirs": library_dirs}


ext_modules.append(
    TopsTorchExtension(
        name="vllm_gcu._C",
        sources=_get_all_sources("csrc"),
        libraries=["torch_extension", "tops_extension", "topsaten", "torch_gcu"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "topscc": TOPSCC_FLAGS.copy(),
        },
        extra_link_args=[
            "-Wl,--disable-new-dtags",
            "-Wl,-rpath,$ORIGIN/../tops_extension/lib:$ORIGIN/../torch_gcu/lib",
        ],
        py_limited_api=True,
        **_get_include_and_library_dirs(),
    )
)

setup(
    name="vllm_gcu",
    version=VERSION,
    author="Enflame",
    license="Apache 2.0",
    description=("GCU plugin backend for vLLM"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": VllmBuildExtension,
        "clean": VllmClean,
        "bdist_wheel": VllmBdistWheel,
    },
    extras_require={},
    entry_points={
        "vllm.general_plugins": [
            "register_custom_models = vllm_gcu:register_custom_models"
        ],
        "vllm.platform_plugins": [
            "register_platform_plugins = vllm_gcu:register_platform_plugins"
        ],
    },
)
