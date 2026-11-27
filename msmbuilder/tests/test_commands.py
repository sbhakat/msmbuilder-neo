
import glob
import itertools
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import pytest

import hmmlearn.hmm
import mdtraj as md
import numpy as np
from mdtraj.testing import eq

from msmbuilder.dataset import dataset
from msmbuilder.example_datasets import get_data_home, FsPeptide
from msmbuilder.utils import load

# Compatibility issue (caused by FullArgSpec) for python3.11 solved.
# import sys
# if not sys.version_info < (3, 11):
    # pytest.skip("skipping command tests for python>=3.11", allow_module_level=True)

DATADIR = HMM = None

t = FsPeptide().get_cached().trajectories[0][0]
fn = os.path.join("{}","fs_peptide", "fs-peptide.pdb").format(get_data_home())

def setup_module():
    global DATADIR, HMM
    DATADIR = tempfile.mkdtemp()
    # 4 components and 3 features. Each feature is going to be the x, y, z
    # coordinate of 1 atom
    HMM = hmmlearn.hmm.GaussianHMM(n_components=4)
    HMM.transmat_ = np.array([[0.9, 0.1, 0.0, 0.0],
                              [0.1, 0.7, 0.2, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.0, 0.1, 0.1, 0.8]])
    HMM.means_ = np.array([[-10, -10, -10],
                           [-5, -5, -5],
                           [5, 5, 5],
                           [10, 10, 10]])
    HMM.covars_ = np.array([[0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5],
                            [1, 1, 1],
                            [4, 4, 4]])
    HMM.startprob_ = np.array([1, 1, 1, 1]) / 4.0

    # get a 1 atom topology
    topology = t.atom_slice([1]).topology

    # generate the trajectories and save them to disk
    for i in range(10):
        d, s = HMM.sample(100)
        traj = md.Trajectory(xyz=d.reshape(len(d), 1, 3), topology=topology)
        traj.save(os.path.join(DATADIR, 'Trajectory%d.h5' % i))

    assert os.path.exists(os.path.join("{}", "alanine_dipeptide").format(get_data_home()))


def teardown_module():
    shutil.rmtree(DATADIR)


class tempdir:
    def __enter__(self):
        self._curdir = os.path.abspath(os.curdir)
        self._tempdir = tempfile.mkdtemp()
        os.chdir(self._tempdir)

    def __exit__(self, *exc_info):
        os.chdir(self._curdir)
        shutil.rmtree(self._tempdir)


def shell(str):
    split = shlex.split(str, posix=False)
    try:
        subprocess.check_output(split, stderr=subprocess.STDOUT,
                                universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.cmd)
        print(e.output)
        raise

# Potentially may fail ("Fatal Python error: Illegal instruction") in macos: re-run or skip.
def test_atomindices_1():
    # fn = get_mdtraj_fn('2EQQ.pdb')
    # t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -a -p %s' % fn)
        shell('msmb AtomIndices -o all-pairs.txt --all -d -p %s' % fn)
        atoms = np.loadtxt('all.txt', int)
        pairs = np.loadtxt('all-pairs.txt', int)
        eq(t.n_atoms, len(atoms))
        eq(int(t.n_atoms * (t.n_atoms - 1) / 2), len(pairs))


def test_atomindices_2():
    # fn = get_mdtraj_fn('2EQQ.pdb')
    # t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o heavy.txt --heavy -a -p %s' % fn)
        shell('msmb AtomIndices -o heavy-pairs.txt --heavy -d -p %s' % fn)
        atoms = np.loadtxt('heavy.txt', int)
        pairs = np.loadtxt('heavy-pairs.txt', int)
        assert all(t.topology.atom(i).element.symbol != 'H' for i in atoms)
        assert (sum(1 for a in t.topology.atoms if a.element.symbol != 'H') ==
                len(atoms))
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_atomindices_3():
    # fn = get_mdtraj_fn('2EQQ.pdb')
    # t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o alpha.txt --alpha -a -p %s' % fn)
        shell('msmb AtomIndices -o alpha-pairs.txt --alpha -d -p %s' % fn)
        atoms = np.loadtxt('alpha.txt', int)
        pairs = np.loadtxt('alpha-pairs.txt', int)
        assert all(t.topology.atom(i).name == 'CA' for i in atoms)
        assert sum(1 for a in t.topology.atoms if a.name == 'CA') == len(atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_atomindices_4():
    # fn = get_mdtraj_fn('2EQQ.pdb')
    # t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o minimal.txt --minimal -a -p %s' % fn)
        shell('msmb AtomIndices -o minimal-pairs.txt --minimal -d -p %s' % fn)
        atoms = np.loadtxt('minimal.txt', int)
        pairs = np.loadtxt('minimal-pairs.txt', int)
        assert all(t.topology.atom(i).name in ['CA', 'CB', 'C', 'N', 'O']
                   for i in atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_superpose_featurizer():
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -a -p '
              '%s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb SuperposeFeaturizer "
              f"--trjs '{get_data_home()}/alanine_dipeptide/*.dcd'"
              " --transformed distances --atom_indices all.txt"
              f" --reference_traj {get_data_home()}/alanine_dipeptide/ala2.pdb"
              f" --top {get_data_home()}/alanine_dipeptide/ala2.pdb"
              )
        ds = dataset('distances')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.txt'))
        print(ds.provenance)


def test_superpose_featurizer_reftop():
    # see issue #555
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -a -p '
              '%s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb SuperposeFeaturizer "
              f"--trjs '{get_data_home()}/alanine_dipeptide/*.dcd'"
              " --transformed distances --atom_indices all.txt"
              f" --reference_traj {get_data_home()}/alanine_dipeptide/trajectory-0.dcd"
              f" --top {get_data_home()}/alanine_dipeptide/ala2.pdb"
              )
        ds = dataset('distances')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.txt'))
        print(ds.provenance)


def test_atom_pairs_featurizer():
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -d -p '
              '%s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb AtomPairsFeaturizer "
              f"--trjs '{get_data_home()}/alanine_dipeptide/*.dcd'"
              " --transformed pairs --pair_indices all.txt"
              f" --top {get_data_home()}/alanine_dipeptide/ala2.pdb"
              )
        ds = dataset('pairs')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.txt') ** 2)
        print(ds.provenance)


def test_transform_command_1():
    with tempdir():
        shell(f"msmb KCenters -i {get_data_home()}/alanine_dipeptide/*.dcd "
              f"-o model.pkl --top {get_data_home()}/alanine_dipeptide/ala2.pdb "
              "--metric rmsd")
        shell(f"msmb TransformDataset -i {get_data_home()}/alanine_dipeptide/*.dcd "
              "-m model.pkl -t transformed.h5 --top "
              f"{get_data_home()}/alanine_dipeptide/ala2.pdb"
              )
        try:
            eq(dataset('transformed.h5')[0], load('model.pkl').labels_[0])
        except:
            print('Arrays not eq')

    with tempdir():
        shell(f"msmb KCenters -i {get_data_home()}/alanine_dipeptide/trajectory-0.dcd "
              f"-o model.pkl --top {get_data_home()}/alanine_dipeptide/ala2.pdb "
              "--metric rmsd")


def test_transform_command_2():
    with tempdir():
        shell(f"msmb KCenters -i {get_data_home()}/alanine_dipeptide/trajectory-0.dcd "
              f"-o model.pkl --top {get_data_home()}/alanine_dipeptide/ala2.pdb "
              "--metric rmsd "
              "--stride 2")


def test_help():
    shell('msmb -h')


def test_convert_chunked_project_1():
    with tempdir():
        root = os.path.join(get_data_home(), 'alanine_dipeptide')
        assert os.path.exists(root)
        cmd = (f"msmb ConvertChunkedProject out {root} --pattern *.dcd "
               f"-t {root}/ala2.pdb")
        shell(cmd)
        assert set(os.listdir('out')) == {'traj-00000000.dcd',
                                          'trajectories.jsonl'}

        # check that out/traj-00000.dcd really has concatenated all of
        # the input trajs
        length = len(md.open('out/traj-00000000.dcd'))
        assert length == sum(len(md.open(f))
                             for f in glob.glob('%s/*.dcd' % root))

        with open('out/trajectories.jsonl') as f:
            record = json.load(f)
        assert set(record.keys()) == {'filename', 'chunks'}
        assert record['filename'] == 'traj-00000000.dcd'
        assert sorted(glob.glob('%s/*.dcd' % root)) == record['chunks']


def test_vamp_command():
    from msmbuilder.decomposition import VAMP
    rs = np.random.RandomState(0)
    seqs = [np.cumsum(rs.randn(300, 3), axis=0) for _ in range(3)]
    with tempdir():
        with dataset('features', mode='w', fmt='dir-npy') as ds:
            for i, s in enumerate(seqs):
                ds[i] = s
        shell("msmb VAMP -i features -o vamp.pkl -t vamp.h5 "
              "--lag_time 2 --n_components 2")
        model = load('vamp.pkl')
        expected = VAMP(lag_time=2, n_components=2).fit(seqs)
        np.testing.assert_allclose(model.singular_values_, expected.singular_values_)
        np.testing.assert_allclose(dataset('vamp.h5')[0], expected.transform(seqs)[0])


def test_command_defaults_parse():
    # Every documented default must survive the command's own argument types
    # (a parameter documented as int but defaulting to 'auto' used to make
    # e.g. `msmb KMeans` fail before running).
    import argparse

    import msmbuilder.commands  # noqa: F401
    from msmbuilder.cmdline import NumpydocClassCommand

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    checked = 0
    for cmd in set(walk(NumpydocClassCommand)):
        if not getattr(cmd, '_concrete', False) or getattr(cmd, 'klass', None) is None:
            continue
        parser = argparse.ArgumentParser()
        cmd._register_arguments(parser)
        for action in parser._actions:
            if callable(action.type) and isinstance(action.default, str):
                action.type(action.default)     # must not raise
        checked += 1
    assert checked > 50


def test_kmeans_commands_run_with_defaults(tmp_path):
    rs = np.random.RandomState(0)
    path = str(tmp_path / "x.h5")
    with dataset(path, mode="w", fmt="hdf5") as ds:
        for i in range(3):
            ds[i] = rs.randn(100, 2)
    for cmd in ("KMeans", "MiniBatchKMeans"):
        out = str(tmp_path / (cmd + ".h5"))
        assert subprocess.call(["msmb", cmd, "-i", path, "--transformed", out,
                                "--n_clusters", "3", "--random_state", "0"]) == 0
        assert os.path.exists(out)
