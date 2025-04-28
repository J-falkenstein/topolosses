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

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
modindex_common_prefix = ["topolosses.losses."]
autoclass_content = "both"


autodoc_mock_imports = [
    "topolosses.losses.topograph.src._topograph",
    "topolosses.losses.betti_matching.src.betti_matching",
    "Topograph",
    "cv2",
    "gudhi",
    "torchvision",
    "scipy",
    "numpy",
    "torch",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"
