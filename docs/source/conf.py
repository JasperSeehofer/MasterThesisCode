import os
import sys

# Resolve project root relative to this file so path is correct regardless of CWD.
# (Running `make -C docs html` changes CWD to docs/, making os.path.abspath("../..") wrong.)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


project = "EMRI Bayesian H₀ Inference"
copyright = "2025, Jasper Seehofer"
author = "Jasper Seehofer"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# Autosummary: auto-generate .rst stubs on build (no more manual _generated/ files)
autosummary_generate = True

# Napoleon: standardize on Google style
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

# Autodoc: sensible defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,
}
autodoc_typehints = "description"  # render type hints as prose, not cluttering the signature
autodoc_preserve_defaults = True  # read default values from source AST, not repr()

# Mock heavy/GPU/side-effect modules so Sphinx can import the package without executing them.
# DetectionFraction.__init__ calls matplotlib at import time (side effect).
autodoc_mock_imports = [
    "master_thesis_code.M1_model_extracted_data",
    "cupy",
    "cupyx",
    "few",
    "fastlisaresponse",
    "GPUtil",
]

# Intersphinx: cross-link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.13", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

html_theme = "furo"
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
