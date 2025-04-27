# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys

sys.path.insert(0, os.path.abspath(".."))

project = "topolosses"
copyright = "2025, Janek Falkenstein"
author = "Janek Falkenstein"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
modindex_common_prefix = ["topolosses.losses."]

autodoc_mock_imports = [
    "topolosses.losses.topograph.src._topograph",
    "topolosses.losses.betti_matching.src.betti_matching",
    "Topograph",
    "cv2",
    "gudhi",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"
