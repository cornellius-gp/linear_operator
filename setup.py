#!/usr/bin/env python3

import io
import os
import re
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

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

torch_min = "2.0"
install_requires = [">=".join(["torch", torch_min])]

# if recent dev version of PyTorch is installed, no need to install stable
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = []
except ImportError:
    pass

# Other requirements
install_requires += [
    "scipy",
    "jaxtyping",
    "mpmath>=0.19,<=1.3",  # avoid incompatibiltiy with torch+sympy with mpmath 1.4
]


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
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=f">={REQUIRED_MAJOR}.{REQUIRED_MINOR}",
    install_requires=install_requires,
    extras_require={
        "dev": ["pre-commit", "setuptools_scm", "ufmt", "twine"],
        "docs": [
            "myst-parser",
            "setuptools_scm",
            "sphinx",
            "six",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
        ],
        "test": [
            "flake8==5.0.4",
            "flake8-print==5.0.0",
            "pytest",
            "typeguard~=2.13.3"  # jaxtyping seems to only be compatible with older typeguard versions
            # https://github.com/patrick-kidger/jaxtyping/commit/77c263c3def8ea3bcb7d7642c5a8402c16cf76fb
        ],
    },
    test_suite="test",
)
