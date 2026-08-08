# Configuration file for the Sphinx documentation builder.
#
# For a full list of options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Path to the SDK source, so autodoc can import it.
sys.path.insert(0, os.path.abspath('../src'))

project = 'Tish Video SDK'
project_copyright = '2026, Cyril PETER'
author = 'Cyril PETER'

version = '0.1'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns: list[str] = []
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_title = f'{project} v{release} Documentation'
html_static_path = ['_static']
