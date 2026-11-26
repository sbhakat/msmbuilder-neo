"""Print every file under msmbuilder/ that should be installed as-is.

Meson calls this at configure time so new Python modules and data files are
picked up without editing meson.build. Compiled sources (Cython, C, C++) and
build leftovers are skipped; they are handled by the extension targets.
"""
import os
import sys

SKIP_SUFFIXES = (".pyx", ".pxd", ".pxi", ".c", ".cc", ".cpp", ".h", ".hpp",
                 ".so", ".pyd", ".pyc", ".o")
SKIP_NAMES = {".gitignore", ".DS_Store", "version.py", "version.py.in"}

root = sys.argv[1]
for dirpath, dirnames, filenames in os.walk(os.path.join(root, "msmbuilder")):
    dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
    for name in sorted(filenames):
        if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
            continue
        print(os.path.relpath(os.path.join(dirpath, name), root))
