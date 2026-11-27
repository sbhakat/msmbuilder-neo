"""Read trajectories with MDAnalysis and featurize them in chunks.

MDAnalysis reads many topology and trajectory formats that MDTraj does not
(for example GROMACS .tpr, AMBER prmtop, CHARMM/NAMD psf with bonds, LAMMPS).
These helpers convert an MDAnalysis ``Universe`` or ``AtomGroup`` into
MDTraj ``Trajectory`` chunks, so every MSMBuilder featurizer works on it
unchanged, while only one chunk of frames is held in memory at a time.

MDAnalysis is an optional dependency: ``pip install msmbuilder-neo[mdanalysis]``.
"""
import mdtraj as md
import numpy as np
from mdtraj.formats.pdb.pdbfile import PDBTrajectoryFile

__all__ = ['mdanalysis_topology', 'iter_mdanalysis', 'load_mdanalysis',
           'featurize_mdanalysis']


def _require_mdanalysis():
    try:
        import MDAnalysis
    except ImportError as e:  # pragma: no cover - exercised only without MDAnalysis
        raise ImportError(
            'This function needs MDAnalysis. Install it with '
            '"pip install MDAnalysis" or "pip install msmbuilder-neo[mdanalysis]".') from e
    return MDAnalysis


def _atoms(obj):
    """AtomGroup for a Universe or AtomGroup."""
    return obj.atoms


def mdanalysis_topology(atomgroup, standard_names=True):
    """Convert the topology of an MDAnalysis Universe or AtomGroup to MDTraj.

    Parameters
    ----------
    atomgroup : MDAnalysis.Universe or MDAnalysis.AtomGroup
        Atoms to convert, in order.
    standard_names : bool, default=True
        Rename atoms and residues to MDTraj's standard names, as
        ``md.load`` does for PDB files (for example ``HN`` -> ``H``).

    Returns
    -------
    topology : mdtraj.Topology

    Notes
    -----
    Elements come from the MDAnalysis topology when it has them, and are
    otherwise guessed from atom and residue names with MDTraj's PDB rules,
    so the result matches loading the same PDB with ``md.load``. A new chain
    starts whenever the segment or chain ID changes. Bonds are copied when
    the MDAnalysis topology has them; otherwise MDTraj's standard residue
    bonds are created, again as ``md.load`` does for PDB files.
    """
    _require_mdanalysis()
    ag = _atoms(atomgroup)
    if standard_names:
        PDBTrajectoryFile._loadNameReplacementTables()

    has_elements = hasattr(ag, 'elements')
    has_chain_ids = hasattr(ag, 'chainIDs')
    has_segids = hasattr(ag, 'segids')
    atoms = list(ag)
    # contiguous runs of atoms from the same residue, in AtomGroup order (the
    # order coordinates are read in)
    runs = []
    for k, atom in enumerate(atoms):
        if k == 0 or atom.resindex != atoms[k - 1].resindex:
            runs.append([atom])
        else:
            runs[-1].append(atom)

    top = md.Topology()
    chain = chain_key = None
    atom_map = {}
    for run in runs:
        first = run[0]
        segid = str(first.segment.segid) if has_segids else ''
        key = (segid, first.chainID if has_chain_ids else None)
        if chain is None or key != chain_key:
            chain = top.add_chain()
            chain_key = key

        res_name = str(first.resname)
        if standard_names and res_name in PDBTrajectoryFile._residueNameReplacements:
            res_name = PDBTrajectoryFile._residueNameReplacements[res_name]
        residue = top.add_residue(res_name, chain, resSeq=int(first.resid),
                                  segment_id=segid)
        replacements = {}
        if standard_names and res_name in PDBTrajectoryFile._atomNameReplacements:
            replacements = PDBTrajectoryFile._atomNameReplacements[res_name]

        for atom in run:
            name = str(atom.name).strip()
            name = replacements.get(name, name)
            element = None
            if has_elements and str(atom.element).strip():
                try:
                    element = md.element.get_by_symbol(str(atom.element).strip().capitalize())
                except KeyError:
                    element = None
            if element is None:
                element = PDBTrajectoryFile._guess_element(name, res_name, len(run))
            atom_map[atom.index] = top.add_atom(name, element, residue,
                                                serial=int(atom.id) if hasattr(atom, 'id') else None)

    try:
        bonds = ag.bonds
    except Exception:  # MDAnalysis raises NoDataError when there are no bonds
        bonds = None
    if bonds is not None and len(bonds) > 0:
        for bond in bonds:
            i, j = bond.atoms[0].index, bond.atoms[1].index
            if i in atom_map and j in atom_map:
                top.add_bond(atom_map[i], atom_map[j])
    else:
        top.create_standard_bonds()
    return top


def _frame_slice(trajectory, start, stop, step):
    n = len(trajectory)
    return range(*slice(start, stop, step).indices(n))


def iter_mdanalysis(atomgroup, chunk=1000, start=None, stop=None, step=None,
                    topology=None, standard_names=True):
    """Iterate over an MDAnalysis trajectory in MDTraj chunks.

    Parameters
    ----------
    atomgroup : MDAnalysis.Universe or MDAnalysis.AtomGroup
        Universe (all atoms) or AtomGroup (a subset, in its order) to read.
    chunk : int, default=1000
        Maximum number of frames per yielded chunk.
    start, stop, step : int, optional
        Which frames to read, as in ``universe.trajectory[start:stop:step]``.
    topology : mdtraj.Topology, optional
        Topology to attach to the chunks. Converting the topology once and
        passing it here avoids repeating the conversion; by default it is
        built with :func:`mdanalysis_topology`.
    standard_names : bool, default=True
        Passed to :func:`mdanalysis_topology` when ``topology`` is None.

    Yields
    ------
    traj : mdtraj.Trajectory
        Up to ``chunk`` frames, with coordinates and box in nm and time in ps.
    """
    _require_mdanalysis()
    ag = _atoms(atomgroup)
    if int(chunk) < 1:
        raise ValueError('chunk must be at least 1, got %r' % chunk)
    if topology is None:
        topology = mdanalysis_topology(ag, standard_names=standard_names)
    if topology.n_atoms != ag.n_atoms:
        raise ValueError('topology has %d atoms but the AtomGroup has %d'
                         % (topology.n_atoms, ag.n_atoms))

    trajectory = ag.universe.trajectory
    frames = _frame_slice(trajectory, start, stop, step)
    for c in range(0, len(frames), int(chunk)):
        block = frames[c:c + int(chunk)]
        xyz = np.empty((len(block), ag.n_atoms, 3), dtype=np.float32)
        time = np.empty(len(block))
        lengths = np.empty((len(block), 3), dtype=np.float32)
        angles = np.empty((len(block), 3), dtype=np.float32)
        has_box = True
        for k, ts in enumerate(trajectory[list(block)]):
            # MDAnalysis works in Angstrom, MDTraj in nm
            xyz[k] = ag.positions / 10.0
            time[k] = ts.time
            dims = ts.dimensions
            if dims is None:
                has_box = False
            else:
                lengths[k] = np.asarray(dims[:3]) / 10.0
                angles[k] = dims[3:]
        yield md.Trajectory(xyz, topology, time=time,
                            unitcell_lengths=lengths if has_box else None,
                            unitcell_angles=angles if has_box else None)


def load_mdanalysis(atomgroup, start=None, stop=None, step=None, topology=None,
                    standard_names=True):
    """Load (a slice of) an MDAnalysis trajectory as one MDTraj Trajectory.

    Parameters are as for :func:`iter_mdanalysis`. All selected frames are
    held in memory; use :func:`iter_mdanalysis` or
    :func:`featurize_mdanalysis` for trajectories that do not fit.

    Returns
    -------
    traj : mdtraj.Trajectory
    """
    chunks = list(iter_mdanalysis(atomgroup, chunk=10**9, start=start, stop=stop,
                                  step=step, topology=topology,
                                  standard_names=standard_names))
    if not chunks:
        if topology is None:
            topology = mdanalysis_topology(atomgroup, standard_names=standard_names)
        return md.Trajectory(np.zeros((0, topology.n_atoms, 3), dtype=np.float32), topology)
    return chunks[0]


def featurize_mdanalysis(featurizer, atomgroups, chunk=1000, start=None, stop=None,
                         step=None, standard_names=True):
    """Featurize MDAnalysis trajectories chunk by chunk.

    Parameters
    ----------
    featurizer : msmbuilder Featurizer
        Any featurizer whose ``partial_transform`` computes each frame's
        features independently of other frames. All MSMBuilder featurizers
        do, so chunking does not change the result, with one small
        exception: MDTraj's SASA (used by ``SASAFeaturizer``) can differ by
        about 0.1% for the first frames of a chunk, depending on how frames
        are batched, within the sampling noise of the Shrake-Rupley method.
    atomgroups : Universe, AtomGroup, or list of them
        One entry per trajectory.
    chunk : int, default=1000
        Frames read and featurized at a time; this bounds memory use.
    start, stop, step : int, optional
        Frames to use from each trajectory.
    standard_names : bool, default=True
        See :func:`mdanalysis_topology`.

    Returns
    -------
    features : list of np.ndarray
        One array of shape (n_frames_i, n_features) per trajectory, as from
        ``featurizer.transform``.
    """
    if not isinstance(atomgroups, (list, tuple)):
        atomgroups = [atomgroups]
    out = []
    for ag in atomgroups:
        top = mdanalysis_topology(ag, standard_names=standard_names)
        parts = [featurizer.partial_transform(t)
                 for t in iter_mdanalysis(ag, chunk=chunk, start=start, stop=stop,
                                          step=step, topology=top)]
        if parts:
            out.append(np.concatenate(parts))
        else:
            out.append(np.zeros((0, 0)))
    return out
