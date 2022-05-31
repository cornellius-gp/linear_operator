# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import io
import re
import sys
import warnings
import sphinx_rtd_theme  # noqa
from typing import ForwardRef


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), "..", "..", *names), encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# -- Project information -----------------------------------------------------

project = "linear_operator"
copyright = "2020, Cornellius GP"
author = "Cornellius GP"

# The full version, including alpha/beta/rc tags
version = find_version("linear_operator", "__init__.py")
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

# Disable documentation inheritance so as to avoid inheriting
# docstrings in a different format, e.g. when the parent class
# is a PyTorch class.

autodoc_inherit_docstrings = False

# Function to format type hints


def _process(annotation, config):
    """
    A function to convert a type/rtype typehint annotation into a :type:/:rtype: string.
    This function is a bit hacky, and specific to the type annotations we use most frequently.

    This function is recursive.
    """
    # Simple/base case: any string annotation is ready to go
    if type(annotation) == str:
        return annotation

    # Convert Ellipsis into "..."
    elif annotation == Ellipsis:
        return "..."

    # Convert any class (i.e. torch.Tensor, LinearOperator, etc.) into appropriate strings
    # For external classes, the format will be e.g. "torch.Tensor"
    # For any internal class, the format will be e.g. "~linear_operator.operators.TriangularLinearOperator"
    elif hasattr(annotation, "__name__"):
        module = annotation.__module__ + "."
        if module.split(".")[0] == "linear_operator":
            module = "~" + module
        elif module == "builtins.":
            module = ""
        res = f"{module}{annotation.__name__}"

    # Convert any Union[*A*, *B*, *C*] into "*A* or *B* or *C*"
    # Also, convert any Optional[*A*] into "*A*, optional"
    elif "typing.Union" in str(annotation):
        is_optional_str = ""
        args = list(annotation.__args__)
        # Hack: Optional[*A*] are represented internally as Union[*A*, Nonetype]
        # This catches this case
        if args[-1] is type(None):  # noqa E721
            del args[-1]
            is_optional_str = ", optional"
        processed_args = [_process(arg, config) for arg in args]
        res = " or ".join(processed_args) + is_optional_str

    # Convert any Tuple[*A*, *B*] into "(*A*, *B*)"
    elif "typing.Tuple" in str(annotation):
        args = list(annotation.__args__)
        res = "(" + ", ".join(_process(arg, config) for arg in args) + ")"

    # Callable typing annotation
    elif "typing." in str(annotation):
        return str(annotation)

    # Special cases for forward references.
    # This is brittle, as it only contains case for a select few forward refs
    # All others that aren't caught by this are handled by the final case
    elif isinstance(annotation, ForwardRef):
        res = str(annotation.__forward_arg__)
        if res == "LinearOperator":
            res = "~linear_operator.LinearOperator"
        elif "LinearOperator" in res:
            res = f"~linear_operator.operators.{res}"

    # For everything we didn't catch: use the simplist string representation
    else:
        warnings.warn(f"No rule for {annotation}. Using default resolution...", RuntimeWarning)
        res = str(annotation)

    return res


# Options for typehints

always_document_param_types = True
# typehints_use_rtype = False
typehints_defaults = None  # or "comma"
simplify_optional_unions = False
typehints_formatter = _process

# Taken from https://github.com/pyro-ppl/pyro/blob/dev/docs/source/conf.py#L213
# @jpchen's hack to get rtd builder to install latest pytorch
# See similar line in the install section of .travis.yml
if "READTHEDOCS" in os.environ:
    os.system("pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html")
