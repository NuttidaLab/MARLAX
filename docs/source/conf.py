# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Marlax'
copyright = '2025, Rudramani Singha'
author = 'Rudramani Singha'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'myst_nb',
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring  = False

source_suffix = {
    '.rst':  'restructuredtext',
    '.md':   'myst-nb',
    '.ipynb':'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = []

html_title = "MARLAX"
html_logo = "../../landing.gif"
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/NuttidaLab/MARLAX',
    "use_repository_button": True,
    "use_download_button": False,
    'repository_branch': 'main',
    "path_to_docs": 'docs/source',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
        'binderhub_url': 'https://mybinder.org'
    },
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ['_static']

import os, sys
sys.path.insert(0, os.path.abspath('../../'))

from marlax import __version__ as version
release = version