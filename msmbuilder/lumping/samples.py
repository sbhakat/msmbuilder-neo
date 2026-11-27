"""Write representative structures of each macrostate to disk."""
import os
import warnings

import mdtraj as md
import numpy as np
import pandas as pd

__all__ = ['save_macrostate_samples']


def _resolve_top(top):
    if top is None or isinstance(top, md.Topology):
        return top
    if isinstance(top, md.Trajectory):
        return top.topology
    return md.load_topology(os.path.expanduser(str(top)))


def save_macrostate_samples(lumper, sequences, trajectories, n_samples=100, top=None,
                            stride=1, out_dir='.', prefix='macrostate', fmt='pdb',
                            superpose=None, weighting='membership', random_state=None):
    """Save sampled structures of each macrostate, one file per macrostate.

    The MSMBuilder equivalent of PyEMMA's ``msm.sample_by_distributions``
    followed by ``coordinates.save_trajs``: frames are drawn for each
    macrostate of a fitted :class:`PCCA` or :class:`PCCAPlus` model (see
    :meth:`PCCA.draw_macrostate_samples`) and written as
    ``<prefix>_<k>.<fmt>`` -- a multi-model PDB by default -- together with
    ``<prefix>_samples.csv``, which lists where every structure came from.

    Parameters
    ----------
    lumper : PCCA or PCCAPlus
        A fitted lumping model.
    sequences : list of array-like
        The microstate label sequences (e.g. cluster labels) of the
        trajectories, in the same order as ``trajectories``.
    trajectories : list of str or md.Trajectory
        The trajectories: file names (loaded frame by frame with MDTraj, so
        they need not fit in memory) or loaded trajectories.
    n_samples : int, default=100
        Structures per macrostate (drawn with replacement).
    top : str, md.Topology or md.Trajectory, optional
        Topology for trajectory files that do not contain one (e.g. .xtc,
        .dcd).
    stride : int, default=1
        Frame ``f`` of a label sequence is frame ``f * stride`` of its
        trajectory (use the stride with which the trajectories were
        featurized).
    out_dir : str, default='.'
        Directory for the output files (created if needed).
    prefix : str, default='macrostate'
    fmt : str, default='pdb'
        Output format (any extension MDTraj can save, e.g. 'pdb', 'xtc',
        'dcd', 'h5').
    superpose : str, optional
        MDTraj atom selection (e.g. 'name CA' or 'backbone'); if given, the
        structures of each macrostate are superposed onto its first one.
    weighting : {'membership', 'assignment'}, default='membership'
        See :meth:`PCCA.draw_macrostate_samples`.
    random_state : int, RandomState or None

    Returns
    -------
    result : dict
        ``files``: the written file of each macrostate (``None`` for a
        macrostate without frames); ``index``: a DataFrame with one row per
        structure (macrostate, sample, trajectory, trajectory_file, frame,
        microstate, membership), also saved as ``<prefix>_samples.csv``.
    """
    if not hasattr(lumper, 'draw_macrostate_samples'):
        raise TypeError('save_macrostate_samples needs a fitted PCCA or PCCAPlus model')
    sequences = list(sequences)
    trajectories = list(trajectories)
    if len(sequences) != len(trajectories):
        raise ValueError('sequences and trajectories must have the same length '
                         '(%d and %d)' % (len(sequences), len(trajectories)))
    stride = int(stride)
    if stride < 1:
        raise ValueError('stride must be at least 1')
    top = _resolve_top(top)
    samples = lumper.draw_macrostate_samples(sequences, n_samples, weighting=weighting,
                                             random_state=random_state)
    internal = lumper._internal_sequences(sequences)
    labels = {i: label for label, i in lumper.mapping_.items()}
    os.makedirs(out_dir, exist_ok=True)

    cache = {}

    def load(t, frame):
        key = (t, frame)
        if key not in cache:
            src = trajectories[t]
            if isinstance(src, md.Trajectory):
                if frame >= src.n_frames:
                    raise IndexError('frame %d is past the end of trajectory %d (%d frames); '
                                     'check stride' % (frame, t, src.n_frames))
                cache[key] = src[frame]
            else:
                path = os.path.expanduser(str(src))
                kw = {'top': top} if top is not None else {}
                try:
                    loaded = md.load_frame(path, frame, **kw)
                except (IndexError, ValueError, OSError) as e:
                    raise IndexError('could not read frame %d of %s (%s); check stride'
                                     % (frame, path, e)) from None
                if loaded.n_frames != 1:
                    raise IndexError('frame %d is past the end of %s; check stride'
                                     % (frame, path))
                cache[key] = loaded
        return cache[key]

    files, rows = [], []
    for k, pairs in enumerate(samples):
        if len(pairs) == 0:
            warnings.warn('macrostate %d has no frames in these sequences; no file written' % k)
            files.append(None)
            continue
        frames = []
        for n, (t, f) in enumerate(pairs):
            t, f = int(t), int(f)
            frames.append(load(t, f * stride))
            micro = int(internal[t][f])
            src = trajectories[t]
            rows.append({'macrostate': k, 'sample': n, 'trajectory': t,
                         'trajectory_file': None if isinstance(src, md.Trajectory) else str(src),
                         'frame': f * stride, 'microstate': labels[micro],
                         'membership': float(lumper.memberships_[micro, k])})
        traj = md.join(frames, check_topology=False)
        if superpose is not None:
            atoms = traj.topology.select(superpose)
            if len(atoms) == 0:
                raise ValueError('superpose selection %r matches no atoms' % superpose)
            traj.superpose(traj[0], 0, atom_indices=atoms)
        path = os.path.join(out_dir, '%s_%d.%s' % (prefix, k, fmt))
        traj.save(path)
        files.append(path)

    index = pd.DataFrame(rows, columns=['macrostate', 'sample', 'trajectory',
                                        'trajectory_file', 'frame', 'microstate',
                                        'membership'])
    index.to_csv(os.path.join(out_dir, '%s_samples.csv' % prefix), index=False)
    return {'files': files, 'index': index}
