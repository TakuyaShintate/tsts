import os
import sys

# enable autodoc to load local modules
sys.path.insert(0, os.path.abspath("../tsts"))

project = "tsts"
copyright = "2021, tsts Development Team"
author = "tsts Development Team"
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
html_theme = "sphinx_material"  # html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "../../img/tsts-logo-small.png"
html_theme_options = {
    "nav_title": "tsts: time set for time series",
    "base_url": "https://takuyashintate.github.io/tsts/",
    "color_primary": "green",
    "color_accent": "light-green",
    "repo_url": "https://github.com/TakuyaShintate/tsts",
    "repo_name": "tsts",
    "globaltoc_depth": 1,
    "globaltoc_collapse": False,
    "globaltoc_includehidden": False,
}
