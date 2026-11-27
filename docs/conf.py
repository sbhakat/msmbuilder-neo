"""Sphinx configuration for the MSMBuilder Neo documentation.

Build locally with ``pip install ".[docs]"`` and ``make -C docs html``.
Notebooks are executed during the build unless ``MSMB_DOCS_EXECUTE=off``.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import msmbuilder  # noqa: E402
import msmbuilder.version  # noqa: E402

# --- project ----------------------------------------------------------------------
project = 'MSMBuilder Neo'
author = 'The MSMBuilder and MSMBuilder Neo authors'
copyright = '2014-2016 Stanford University and the MSMBuilder authors; ' \
            '2026 the MSMBuilder Neo authors'
version = msmbuilder.version.short_version
release = msmbuilder.version.version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'myst_nb',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx.ext.extlinks',
]

extlinks = {'doi': ('https://doi.org/%s', 'doi:%s')}   # used by scikit-learn docstrings
myst_heading_anchors = 3

source_suffix = {'.rst': 'restructuredtext', '.md': 'myst-nb', '.ipynb': 'myst-nb'}
root_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build', '_templates', 'publications_templ.rst', 'Thumbs.db', '.DS_Store',
                    '**.ipynb_checkpoints', 'jupyter_execute']

# --- API reference ----------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {'members': True, 'inherited-members': True}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks: set = set()
autodoc_mock_imports: list = []
suppress_warnings = ['autosummary.import_cycle', 'myst.header']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'mdtraj': ('https://mdtraj.org/1.9.4/', None),
}

# --- notebooks (MyST-NB) --------------------------------------------------------------
# The example notebooks live in ../examples; they are copied here at build time
# so the repository keeps a single copy.
_gallery = HERE / 'examples' / 'notebooks'
if _gallery.exists():
    shutil.rmtree(_gallery)
_gallery.mkdir(parents=True)
for nb in sorted((HERE.parent / 'examples').rglob('*.ipynb')):
    if '.ipynb_checkpoints' not in nb.parts:
        shutil.copy2(nb, _gallery / nb.name)

nb_execution_mode = os.environ.get('MSMB_DOCS_EXECUTE', 'force')
nb_execution_timeout = int(os.environ.get('MSMB_DOCS_TIMEOUT', '1800'))
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_in_temp = True          # notebooks write their files in a scratch directory
nb_kernel_rgx_aliases = {'.*': 'python3'}
nb_merge_streams = True
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath', 'deflist']

# --- HTML -------------------------------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_title = 'MSMBuilder Neo'
html_logo = '_static/logo-200px.png'
html_favicon = '_static/logo-200px.png'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_sourcelink = False
html_theme_options = {
    'github_url': 'https://github.com/sbhakat/msmbuilder-neo',
    'navbar_align': 'left',
    'header_links_before_dropdown': 6,
    'show_toc_level': 2,
    'navigation_with_keys': False,
    'use_edit_page_button': True,
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version'],
    'icon_links': [{'name': 'Roadmap and issues',
                    'url': 'https://github.com/sbhakat/msmbuilder-neo/issues',
                    'icon': 'fa-solid fa-list-check'}],
}
html_context = {
    'github_user': 'sbhakat',
    'github_repo': 'msmbuilder-neo',
    'github_version': 'main',
    'doc_path': 'docs',
    'default_mode': 'auto',
}
html_sidebars = {'index': [], 'migration': [], 'whats_new': []}


# --- publications page -----------------------------------------------------------------
def render_publications():
    from jinja2 import Environment, FileSystemLoader

    import bibparse
    env = Environment(loader=FileSystemLoader(str(HERE)))
    template = env.get_template('publications_templ.rst')
    pubs = bibparse.entries.parseString((HERE / 'publications.bib').read_text())
    pubs = sorted(pubs, key=lambda x: -int(x.fields['year']))
    (HERE / 'publications.rst').write_text(template.render(publications=pubs))


render_publications()


# --- docstrings ----------------------------------------------------------------------
_TRAILING_UNDERSCORE = None


def _escape_trailing_underscores(app, what, name, obj, options, lines):
    """Fitted attributes such as ``timescales_`` end in an underscore, which
    reStructuredText reads as a link to a target of that name. Escape the
    underscore outside literals (after numpydoc has formatted the docstring)."""
    import re
    global _TRAILING_UNDERSCORE
    if _TRAILING_UNDERSCORE is None:
        _TRAILING_UNDERSCORE = re.compile(r'(?<![\\`\w])([A-Za-z](?:[\w.]*[A-Za-z0-9])?)_(?![\w`])')
    for i, line in enumerate(lines):
        if line.lstrip().startswith(('>>>', '...')) or '::' in line:
            continue
        parts = line.split('``')
        for j in range(0, len(parts), 2):          # outside ``literals``
            parts[j] = _TRAILING_UNDERSCORE.sub(r'\1\\_', parts[j])
        lines[i] = '``'.join(parts)


def setup(app):
    app.connect('autodoc-process-docstring', _escape_trailing_underscores, priority=900)
