"""
Configuration file for introdl package documentation.
Generated on 2025-09-16T19:41:04.614009
"""

import os
import sys

# Add the package to Python path
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'introdl'
copyright = '2025, DS776 Deep Learning Course'
author = 'DS776 Course Team'
version = '1.4.3'
release = '1.4.3'

# Extensions
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate from docstrings
    'sphinx.ext.napoleon',       # Google/NumPy style docstrings
    'sphinx.ext.viewcode',       # Include source code links
    'sphinx.ext.intersphinx',    # Link to other documentation
    'sphinx.ext.todo',           # Todo notes
    'myst_parser',              # Markdown support (if needed)
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Don't document private functions (starting with _)
autodoc_mock_imports = []

# HTML output configuration
html_theme = 'sphinx_rtd_theme'  # ReadTheDocs theme
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

html_title = 'DS776 introdl Package Documentation'
html_short_title = 'introdl Docs'

# Markdown output configuration
markdown_http_base = 'https://github.com/your-repo'

# General configuration
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Code highlighting
pygments_style = 'sphinx'
highlight_language = 'python'

# Cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# Todo extension
todo_include_todos = True
