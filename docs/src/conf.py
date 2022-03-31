import os
import sys

# enable autodoc to load local modules
sys.path.insert(0, os.path.abspath("../tsts"))

project = "tsts"
copyright = "2021, tsts development team"
author = "tsts development team"
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
]
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_theme = "karma_sphinx_theme"  # "sizzle"
# html_static_path = ["_static"]
# html_theme_options = {}
