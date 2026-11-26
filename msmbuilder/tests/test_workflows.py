
import os
import shlex
import shutil
import subprocess
import tempfile

import pytest

from importlib.resources import files


class tempdir:
    def __enter__(self):
        self._curdir = os.path.abspath(os.curdir)
        self._tempdir = tempfile.mkdtemp()
        os.chdir(self._tempdir)

    def __exit__(self, *exc_info):
        os.chdir(self._curdir)
        shutil.rmtree(self._tempdir)


def shell_lines(resource):
    fn = files('msmbuilder').joinpath(resource)
    buf = ''
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith('\\'):
                buf += line.rstrip('\\')
            else:
                yield buf + ' ' + line
                buf = ''


def check_call(tokens):
    try:
        subprocess.check_output(tokens, stderr=subprocess.STDOUT,
                                universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.cmd)
        print(e.output)
        raise


class workflow_tester:
    def __init__(self, fn):
        self.fn = fn
        self.path = f"tests/workflows/{fn}"
        self.description = f"{__name__}.test_{fn}"

    def __call__(self, *args, **kwargs):
        with tempdir():
            for line in shell_lines(self.path):
                check_call(shlex.split(line, posix=False))


@pytest.mark.parametrize('fn', ['basic.sh', 'rmsd.sh', 'ghmm.sh'])
def test_workflows(fn):
    workflow_tester(fn)()
