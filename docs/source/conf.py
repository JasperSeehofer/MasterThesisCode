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
    "darksiren_emri.M1_model_extracted_data",
    "cupy",
    "cupyx",
    "few",
    "fastlisaresponse",
    "GPUtil",
]

# Intersphinx is intentionally NOT enabled. The CI runner has no reliable
# outbound network to fetch external objects.inv inventories, and the `docs`
# job builds with -W (warnings-as-errors), so a single "failed to reach any of
# the inventories" warning fails the build and blocks the GitHub Pages deploy.
# That warning is untyped and cannot be silenced via suppress_warnings. Without
# intersphinx, external types in the API docs simply render unlinked (no
# warnings, since nitpicky is off), and the build no longer depends on network.
# Re-add `sphinx.ext.intersphinx` + an intersphinx_mapping for a local/networked
# build if external cross-links are wanted.

html_theme = "furo"
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
