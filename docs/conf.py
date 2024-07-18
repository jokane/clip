# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os


sys.path.insert(0, os.path.join(os.path.split(__file__)[0], '..'))
sys.path.insert(0, os.path.join(os.path.split(__file__)[0], '.'))

import generate

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'clip'
copyright = "2024, Jason O'Kane"
author = "Jason O'Kane"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_generated']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {'collapse_navigation': False,
                       'navigation_depth': 1,
                      'prev_next_buttons_location': None }
html_static_path = ['static']



rst_prolog = '\n'.join(f'.. |{tag}| replace:: ​' for tag in generate.tags)

autodoc_member_order = 'bysource'
