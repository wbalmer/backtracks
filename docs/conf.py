# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# import sphinx_readable_theme

# -- Project information -----------------------------------------------------

project = 'backtracks'
copyright = '2024, William Balmer'
author = 'William Balmer'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_automodapi.automodapi',
    'nbsphinx'
]

numpydoc_show_class_members = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ['_build',
                    'Thumbs.db',
                    '.DS_Store',
                    '.ipynb_checkpoints/*']

# -- Options for HTML output -------------------------------------------------

# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme = 'readable'

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    'github_url': 'https://github.com/wbalmer/backtracks',
    'use_edit_page_button': False,
    "logo": {
      "image_light": "_static/backtracks-logo-light.svg",
      "image_dark": "_static/backtracks-logo-dark.svg",
   }
}

html_context = {
    "github_user": "wbalmer",
    "github_repo": "backtracks",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ['_static']

html_search_language = 'en'
