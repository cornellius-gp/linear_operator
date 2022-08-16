#!/usr/bin/env python3

import io
import os
import re
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 8

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)

readme = open("README.md").read()

torch_min = "1.11"
install_requires = [">=".join(["torch", torch_min])]

# if recent dev version of PyTorch is installed, no need to install stable
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = []
except ImportError:
    pass

# Other requirements
install_requires += ["scipy"]


# Get version
def find_version(*file_paths):
    try:
        with io.open(os.path.join(os.path.dirname(__file__), *file_paths), encoding="utf8") as fp:
            version_file = fp.read()
        version_match = re.search(r"^__version__ = version = ['\"]([^'\"]*)['\"]", version_file, re.M)
        return version_match.group(1)
    except Exception:
        return None


# Run the setup
setup(
    name="linear_operator",
    version=find_version("linear_operator", "version.py"),
    description=(
        "A linear operator implementation, primarily designed for finite-dimensional "
        "positive definite operators (i.e. kernel matrices)."
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Geoff Pleiss",
    author_email="gpleiss@gmail.com",
    project_urls={
        "Documentation": "https://linear_operator.readthedocs.io",
        "Source": "https://github.com/cornellius-gp/linear_operator/",
    },
    license="MIT",
    classifiers=["Development Status :: 2 - Pre-Alpha", "Programming Language :: Python :: 3"],
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "twine", "pre-commit"],
        "test": ["flake8==4.0.1", "flake8-print==4.0.0"],
    },
    test_suite="test",
)
