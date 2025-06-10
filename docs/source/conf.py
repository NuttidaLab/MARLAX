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

source_suffix = {
    '.rst':  'restructuredtext',
    '.md':   'myst-nb',
    '.ipynb':'myst-nb',
}

autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (for better docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
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