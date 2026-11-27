"""Run a whole MSMBuilder analysis from one YAML file, reproducibly.

A pipeline file lists the input data and a sequence of steps (a featurizer,
then estimators such as a scaler, tICA, a clustering and a Markov state
model) and optional validation (implied timescales, Chapman-Kolmogorov
test). :func:`run_pipeline` (or ``msmb Pipeline pipeline.yaml``) runs it and
writes, into the output directory:

* ``<step>.h5`` -- the output of each step that transforms its data (an
  MSMBuilder HDF5 dataset), and ``<step>.pkl`` -- the fitted step;
* ``implied_timescales.csv`` and ``ck_test.npz`` -- validation results;
* ``provenance.yaml`` -- the configuration, every step's full parameters,
  package versions, SHA-256 checksums of the input files and of every
  step's output data, and timings.

Rerunning the same file reuses steps whose inputs and parameters have not
changed (their outputs are checked against the recorded checksums), and
recomputes the rest. ``verify=True`` (``msmb Pipeline --verify``) instead
reruns everything from scratch in a temporary directory and checks that the
results are identical to the recorded ones.

Steps with a ``random_state`` parameter get the pipeline's ``random_state``
unless they set their own, so a pipeline with a seed is deterministic.
Bitwise reproducibility is expected with the same package versions on the
same platform; across platforms or library versions the numbers can differ
in the last digits, which the recorded checksums will show.
"""
import copy
import datetime
import glob
import hashlib
import importlib
import json
import os
import pickle
import platform
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import yaml

__all__ = ['run_pipeline', 'load_config', 'verify_pipeline', 'PipelineError',
           'PipelineResult', 'EXAMPLE_CONFIG']

SCHEMA_VERSION = 1

# Where step classes named without a module are looked up, in this order
_SEARCH_MODULES = ('msmbuilder.featurizer', 'msmbuilder.preprocessing',
                   'msmbuilder.decomposition', 'msmbuilder.cluster', 'msmbuilder.msm',
                   'msmbuilder.lumping', 'msmbuilder.hmm')

# Packages whose versions are recorded (when imported)
_PACKAGES = ('msmbuilder', 'numpy', 'scipy', 'sklearn', 'mdtraj', 'numba', 'pandas', 'h5py',
             'yaml', 'torch', 'deeptime', 'MDAnalysis', 'dask')

EXAMPLE_CONFIG = """\
# MSMBuilder pipeline: run with  msmb Pipeline pipeline.yaml
version: 1
random_state: 42            # seed for every step with a random_state parameter
output: pipeline_output     # relative paths are relative to this file

data:
  trajectories: "trajectories/*.xtc"   # glob, or a list of globs/files
  topology: topology.pdb
  stride: 1
  # features: features.h5    # or start from an existing dataset (no featurizer step)

steps:
  - name: features
    featurizer: DihedralFeaturizer
    params: {types: [phi, psi]}
  - name: tica
    estimator: tICA
    params: {n_components: 4, lag_time: 10, kinetic_mapping: true}
  - name: clusters
    estimator: MiniBatchKMeans
    params: {n_clusters: 100}
  - name: msm
    estimator: MarkovStateModel
    params: {lag_time: 10, verbose: false}

validation:
  implied_timescales: {lag_times: [1, 2, 5, 10, 20, 50], n_timescales: 5}
  # ck_test: {lag_times: [10, 20, 40, 80], n_metastable_sets: 3}   # needs deeptime
"""


class PipelineError(ValueError):
    """An invalid pipeline file, or a failed verification."""


# --- configuration --------------------------------------------------------------------

_TOP_KEYS = {'version', 'random_state', 'output', 'data', 'steps', 'validation'}
_DATA_KEYS = {'trajectories', 'topology', 'stride', 'chunk', 'features'}
_STEP_KEYS = {'name', 'featurizer', 'estimator', 'params', 'transform'}
_VALIDATION_KEYS = {'implied_timescales', 'ck_test'}
_NAME_RE = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]*$')


def _check_keys(section, given, allowed):
    unknown = set(given) - allowed
    if unknown:
        raise PipelineError('unknown key(s) in %s: %s (allowed: %s)'
                            % (section, ', '.join(sorted(unknown)), ', '.join(sorted(allowed))))


def resolve_class(name):
    """The MSMBuilder class for a step: a class name (searched in the
    featurizer, preprocessing, decomposition, cluster, msm, lumping and hmm
    modules) or a dotted path inside ``msmbuilder``."""
    if not isinstance(name, str) or not name:
        raise PipelineError('a step class must be a non-empty string, got %r' % (name,))
    if '.' in name:
        module, _, attr = name.rpartition('.')
        if module != 'msmbuilder' and not module.startswith('msmbuilder.'):
            raise PipelineError('step classes must come from msmbuilder, got %r' % name)
        try:
            obj = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            raise PipelineError('no class %r' % name) from None
        if not isinstance(obj, type):
            raise PipelineError('%r is not a class' % name)
        return obj
    for module in _SEARCH_MODULES:
        obj = getattr(importlib.import_module(module), name, None)
        if isinstance(obj, type):
            return obj
    raise PipelineError('no MSMBuilder class named %r (looked in %s)'
                        % (name, ', '.join(_SEARCH_MODULES)))


def _lag_list(value, what):
    if isinstance(value, str):      # 'start:stop[:step]', inclusive like msmb
        parts = [int(p) for p in value.split(':')]
        if len(parts) not in (2, 3):
            raise PipelineError('%s: use a list or "start:stop[:step]"' % what)
        step = parts[2] if len(parts) == 3 else 1
        value = list(range(parts[0], parts[1] + 1, step))
    try:
        lags = [int(v) for v in value]
    except (TypeError, ValueError):
        raise PipelineError('%s must be a list of integers' % what) from None
    if not lags or min(lags) < 1:
        raise PipelineError('%s must be positive integers' % what)
    return lags


def load_config(config):
    """Read and check a pipeline configuration.

    Parameters
    ----------
    config : str or dict
        Path to a YAML file, or the configuration itself.

    Returns
    -------
    config : dict
        The checked configuration, with ``base_dir`` (the directory that
        relative paths refer to) added under the key ``_base_dir``.
    """
    if isinstance(config, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(config))
        with open(path) as f:
            cfg = yaml.safe_load(f)
        base_dir = os.path.dirname(path)
    else:
        cfg = copy.deepcopy(config)
        base_dir = os.path.abspath(cfg.pop('_base_dir', os.getcwd())) \
            if isinstance(cfg, dict) else os.getcwd()
    if not isinstance(cfg, dict):
        raise PipelineError('a pipeline file must be a YAML mapping')
    _check_keys('the pipeline', cfg, _TOP_KEYS)
    if cfg.get('version', SCHEMA_VERSION) != SCHEMA_VERSION:
        raise PipelineError('unsupported pipeline version %r (this is version %d)'
                            % (cfg.get('version'), SCHEMA_VERSION))
    cfg['version'] = SCHEMA_VERSION
    seed = cfg.get('random_state')
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise PipelineError('random_state must be an integer')
    cfg.setdefault('output', 'pipeline_output')

    data = cfg.get('data')
    if not isinstance(data, dict):
        raise PipelineError("the pipeline needs a 'data' section")
    _check_keys('data', data, _DATA_KEYS)
    has_trajs, has_features = 'trajectories' in data, 'features' in data
    if has_trajs == has_features:
        raise PipelineError("data needs exactly one of 'trajectories' and 'features'")
    for key in ('stride', 'chunk'):
        if key in data and (not isinstance(data[key], int) or data[key] < 1):
            raise PipelineError('data.%s must be a positive integer' % key)

    steps = cfg.get('steps')
    if not isinstance(steps, list) or not steps:
        raise PipelineError("the pipeline needs a non-empty list of 'steps'")
    names = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise PipelineError('step %d must be a mapping' % (i + 1))
        _check_keys('step %d' % (i + 1), step, _STEP_KEYS)
        name = step.get('name')
        if not isinstance(name, str) or not _NAME_RE.match(name) or name in names \
                or name in ('provenance', 'implied_timescales', 'ck_test'):
            raise PipelineError('step %d needs a unique name of letters, digits, "_", "-" '
                                'or "." (got %r)' % (i + 1, name))
        names.add(name)
        kinds = [k for k in ('featurizer', 'estimator') if k in step]
        if len(kinds) != 1:
            raise PipelineError("step %r needs exactly one of 'featurizer' and 'estimator'"
                                % name)
        if 'featurizer' in step and (i != 0 or not has_trajs):
            raise PipelineError("only the first step can be a featurizer, and only with "
                                "data.trajectories (step %r)" % name)
        params = step.setdefault('params', {})
        if not isinstance(params, dict):
            raise PipelineError('params of step %r must be a mapping' % name)
        if 'transform' in step and not isinstance(step['transform'], bool):
            raise PipelineError('transform of step %r must be true or false' % name)
        resolve_class(step.get('featurizer', step.get('estimator')))
    if has_trajs and 'featurizer' not in steps[0]:
        raise PipelineError('with data.trajectories, the first step must be a featurizer')

    validation = cfg.setdefault('validation', {}) or {}
    cfg['validation'] = validation
    if not isinstance(validation, dict):
        raise PipelineError('validation must be a mapping')
    _check_keys('validation', validation, _VALIDATION_KEYS)
    if 'implied_timescales' in validation:
        its = validation['implied_timescales'] or {}
        _check_keys('validation.implied_timescales', its, {'lag_times', 'n_timescales'})
        its['lag_times'] = _lag_list(its.get('lag_times', '1:10'),
                                     'implied_timescales.lag_times')
        its.setdefault('n_timescales', 5)
        validation['implied_timescales'] = its
    if 'ck_test' in validation:
        ck = validation['ck_test'] or {}
        _check_keys('validation.ck_test', ck, {'lag_times', 'n_metastable_sets'})
        ck['lag_times'] = _lag_list(ck.get('lag_times'), 'ck_test.lag_times')
        if len(ck['lag_times']) < 2 or sorted(ck['lag_times']) != ck['lag_times']:
            raise PipelineError('ck_test.lag_times must be at least two increasing lag times')
        if not isinstance(ck.get('n_metastable_sets'), int) or ck['n_metastable_sets'] < 2:
            raise PipelineError('ck_test.n_metastable_sets must be an integer >= 2')
        validation['ck_test'] = ck
    cfg['_base_dir'] = base_dir
    return cfg


# --- hashing --------------------------------------------------------------------------

def _sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_arrays(arrays):
    """Checksum of the content of a list of arrays (dtype, shape, values)."""
    h = hashlib.sha256()
    for x in arrays:
        x = np.ascontiguousarray(x)
        h.update(('%s|%s;' % (x.dtype.str, x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def _jsonable(value):
    """A JSON/YAML-friendly representation of a parameter value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return {'array_sha256': _sha256_arrays([value]), 'shape': list(value.shape)}
    if hasattr(value, 'get_params'):
        return {'class': '%s.%s' % (type(value).__module__, type(value).__name__),
                'params': _jsonable(value.get_params(deep=False))}
    return repr(value)


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


# --- data -----------------------------------------------------------------------------

def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def _trajectory_files(data, base_dir):
    patterns = data['trajectories']
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pattern in patterns:
        full = os.path.join(base_dir, os.path.expanduser(str(pattern)))
        matches = sorted(glob.glob(full), key=_natural)
        if not matches:
            raise PipelineError('no trajectory files match %r' % pattern)
        files.extend(matches)
    return files


def _save_dataset(path, arrays):
    from .dataset import dataset
    tmp = path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    with dataset(tmp, 'w', 'hdf5') as ds:
        for i, x in enumerate(arrays):
            ds[i] = x
    os.replace(tmp, path)


def _load_dataset(path):
    from .dataset import dataset
    with dataset(path) as ds:
        return [ds[k] for k in ds.keys()]


def _save_model(path, model):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(model, f)
    os.replace(tmp, path)


def _load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# --- running --------------------------------------------------------------------------

class PipelineResult:
    """What :func:`run_pipeline` returns.

    Attributes
    ----------
    output_dir : str
    models : dict
        Step name -> fitted featurizer or estimator.
    outputs : dict
        Step name -> path of the step's output dataset (for steps that
        transform their data).
    provenance : dict
        The content of ``provenance.yaml``.
    """

    def __init__(self, output_dir, models, outputs, provenance):
        self.output_dir = output_dir
        self.models = models
        self.outputs = outputs
        self.provenance = provenance

    @property
    def fingerprint(self):
        """One checksum of all step outputs and validation results."""
        return self.provenance['fingerprint']

    def __repr__(self):
        return 'PipelineResult(output_dir=%r, steps=%r)' % (self.output_dir, list(self.models))


def _versions():
    out = {'python': platform.python_version()}
    for name in _PACKAGES:
        mod = sys.modules.get(name)
        if mod is None and name in ('msmbuilder', 'numpy', 'scipy', 'sklearn', 'mdtraj',
                                    'numba', 'pandas', 'h5py', 'yaml'):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                mod = None
        if mod is not None:
            version = getattr(mod, '__version__', None)
            if name == 'msmbuilder':
                from . import version as v
                version = v.full_version
            out[name] = str(version)
    return out


def _is_msm(model):
    return hasattr(model, 'transmat_') or hasattr(model, 'countsmat_')


def _log(verbose, msg):
    if verbose:
        print(msg, flush=True)


def run_pipeline(config, output=None, force=False, verbose=True):
    """Run a pipeline file.

    Parameters
    ----------
    config : str or dict
        Path to the pipeline YAML file, or its content as a dict.
    output : str, optional
        Output directory; overrides the file's ``output``.
    force : bool, default=False
        Recompute every step, instead of reusing steps whose inputs,
        parameters and outputs are unchanged since the last run.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    result : PipelineResult
    """
    cfg = load_config(config)
    base_dir = cfg['_base_dir']
    out_dir = os.path.abspath(os.path.join(base_dir, output if output is not None
                                           else os.path.expanduser(str(cfg['output']))))
    os.makedirs(out_dir, exist_ok=True)
    prov_path = os.path.join(out_dir, 'provenance.yaml')
    previous = {}
    if os.path.exists(prov_path) and not force:
        with open(prov_path) as f:
            previous = {s['name']: s for s in (yaml.safe_load(f) or {}).get('steps', [])}

    started = time.time()
    seed = cfg.get('random_state')
    if seed is not None:
        np.random.seed(seed)    # for code that draws from NumPy's global generator
    user_cfg = {k: v for k, v in cfg.items() if k != '_base_dir'}
    data = cfg['data']

    # inputs
    if 'trajectories' in data:
        files = _trajectory_files(data, base_dir)
        inputs = {'trajectories': [{'path': os.path.relpath(f, base_dir),
                                    'sha256': _sha256_file(f), 'bytes': os.path.getsize(f)}
                                   for f in files]}
        top = data.get('topology')
        top_path = os.path.join(base_dir, os.path.expanduser(top)) if top else None
        if top_path is not None:
            if not os.path.exists(top_path):
                raise PipelineError('topology file not found: %s' % top)
            inputs['topology'] = {'path': top, 'sha256': _sha256_file(top_path)}
        seqs = None
    else:
        from .dataset import dataset
        feat_path = os.path.join(base_dir, os.path.expanduser(str(data['features'])))
        if not os.path.exists(feat_path):
            raise PipelineError('features dataset not found: %s' % data['features'])
        with dataset(feat_path) as ds:
            seqs = [ds[k] for k in ds.keys()]
        inputs = {'features': {'path': str(data['features']),
                               'content_sha256': _sha256_arrays(seqs),
                               'n_sequences': len(seqs)}}
    upstream = _key('inputs', inputs, {k: data.get(k) for k in ('stride', 'chunk')})

    msmb_version = _versions()['msmbuilder']
    models, outputs, records = {}, {}, []
    labels_for_msm, msm_template = None, None
    n_steps = len(cfg['steps'])
    for i, step in enumerate(cfg['steps']):
        name = step['name']
        step_input = seqs
        cls = resolve_class(step.get('featurizer', step.get('estimator')))
        try:
            model = cls(**step['params'])
        except TypeError as e:
            raise PipelineError('step %r: %s' % (name, e)) from None
        if seed is not None and 'random_state' in model.get_params(deep=False) \
                and 'random_state' not in step['params']:
            model.set_params(random_state=seed)
        params = _jsonable(model.get_params(deep=False))
        class_path = '%s.%s' % (cls.__module__, cls.__name__)
        is_featurizer = 'featurizer' in step
        # Markov state models are fitted, not used to transform, unless asked
        transforms = is_featurizer or (hasattr(model, 'transform')
                                       and step.get('transform', not _is_msm(model)))
        # the key depends on the content of this step's input, so a step is
        # only reused on exactly the data it was fitted to
        key = _key(upstream, class_path, params, bool(transforms), msmb_version)
        data_path = os.path.join(out_dir, name + '.h5') if transforms else None
        model_path = os.path.join(out_dir, name + '.pkl')

        prev = previous.get(name)
        cached = (prev is not None and prev.get('key') == key and os.path.exists(model_path)
                  and (data_path is None or os.path.exists(data_path)))
        t0 = time.time()
        new_seqs = None
        if cached:
            model = _load_model(model_path)
            if data_path is not None:
                new_seqs = _load_dataset(data_path)
                cached = _sha256_arrays(new_seqs) == prev.get('output_sha256')
            cached = cached and _summary(model) == {k: prev.get(k) for k in _summary(model)}
            if not cached:   # the files were changed since: recompute
                model = cls(**step['params'])
                if seed is not None and 'random_state' not in step['params'] \
                        and 'random_state' in model.get_params(deep=False):
                    model.set_params(random_state=seed)
        if cached:
            _log(verbose, '[%d/%d] %s: unchanged, reusing %s'
                 % (i + 1, n_steps, name, os.path.basename(data_path or model_path)))
        else:
            _log(verbose, '[%d/%d] %s: %s' % (i + 1, n_steps, name, cls.__name__))
            if is_featurizer:
                import mdtraj as md
                new_seqs = []
                kw = dict(chunk=int(data.get('chunk', 10000)), stride=int(data.get('stride', 1)))
                if top_path is not None:
                    kw['top'] = top_path
                for f in files:
                    parts = [model.partial_transform(t) for t in md.iterload(f, **kw)]
                    new_seqs.append(np.concatenate(parts))
            else:
                model.fit(step_input)
                if transforms:
                    new_seqs = list(model.transform(step_input))
            _save_model(model_path, model)
            if data_path is not None:
                _save_dataset(data_path, new_seqs)

        if _is_msm(model):
            msm_template, labels_for_msm = model, step_input
        record = {'name': name, 'class': class_path, 'params': params, 'key': key,
                  'model': os.path.basename(model_path), 'cached': bool(cached),
                  'seconds': round(time.time() - t0, 3)}
        if data_path is not None:
            record['output'] = os.path.basename(data_path)
            record['output_sha256'] = _sha256_arrays(new_seqs)
            record['n_sequences'] = len(new_seqs)
            seqs = new_seqs
            outputs[name] = data_path
            upstream = record['output_sha256']
        else:
            upstream = key
        record.update(_summary(model))
        records.append(record)
        models[name] = model

    validation = _validate(cfg['validation'], msm_template, labels_for_msm, seqs, out_dir,
                           verbose)

    fingerprint = _key([r.get('output_sha256') for r in records],
                       [_summary(models[r['name']]) for r in records],
                       {k: v.get('sha256') for k, v in validation.items()})
    provenance = {
        'msmbuilder_pipeline': SCHEMA_VERSION,
        'config': user_cfg,
        'config_sha256': _key(user_cfg),
        'started': datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
        'seconds': round(time.time() - started, 3),
        'environment': {'platform': platform.platform(), 'versions': _versions()},
        'inputs': inputs,
        'steps': records,
        'validation': validation,
        'fingerprint': fingerprint,
    }
    tmp = prov_path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('# Written by msmbuilder.pipeline; do not edit by hand.\n')
        yaml.safe_dump(provenance, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, prov_path)
    _log(verbose, 'Done: %s (fingerprint %s)' % (out_dir, fingerprint[:12]))
    return PipelineResult(out_dir, models, outputs, provenance)


def _summary(model):
    """Key fitted quantities of a model, for the provenance record."""
    out = {}
    if hasattr(model, 'timescales_') and _is_msm(model):
        try:
            ts = np.asarray(model.timescales_, dtype=float)[:5]
            out['timescales'] = [float(t) for t in ts]
            out['n_states'] = int(model.n_states_)
            out['transmat_sha256'] = _sha256_arrays([model.transmat_])
        except Exception:   # noqa: BLE001 - a summary must never fail the run
            pass
    return out


def _validate(spec, msm, labels, last_seqs, out_dir, verbose):
    results = {}
    if not spec:
        return results
    if msm is None:
        from .msm import MarkovStateModel
        msm = MarkovStateModel(verbose=False)
        labels = last_seqs
    if labels is None:
        raise PipelineError('validation needs state labels: end the pipeline with a '
                            'clustering or a Markov state model step')
    if not all(np.issubdtype(np.asarray(x).dtype, np.integer) for x in labels):
        raise PipelineError('validation needs integer state labels; the last step gives '
                            'continuous features (add a clustering step)')
    from sklearn.base import clone
    template = clone(msm)
    if 'verbose' in template.get_params():
        template.set_params(verbose=False)
    if 'implied_timescales' in spec:
        import pandas as pd
        from .msm import implied_timescales
        its = spec['implied_timescales']
        _log(verbose, 'validation: implied timescales at lag times %s' % its['lag_times'])
        ts = implied_timescales(labels, its['lag_times'], n_timescales=its['n_timescales'],
                                msm=template)
        df = pd.DataFrame(np.asarray(ts), columns=['timescale_%d' % (k + 1)
                                                   for k in range(np.shape(ts)[1])])
        df.insert(0, 'lag_time', its['lag_times'])
        path = os.path.join(out_dir, 'implied_timescales.csv')
        df.to_csv(path, index=False)
        results['implied_timescales'] = {'file': 'implied_timescales.csv',
                                         'sha256': _sha256_arrays([df.to_numpy()])}
    if 'ck_test' in spec:
        from .msm import ck_test
        ck = spec['ck_test']
        _log(verbose, 'validation: Chapman-Kolmogorov test at lag times %s' % ck['lag_times'])
        test = ck_test(template, labels, ck['lag_times'], ck['n_metastable_sets'])
        path = os.path.join(out_dir, 'ck_test.npz')
        lagtimes = np.asarray(test.lagtimes)
        np.savez(path, lag_times=lagtimes, predictions=test.predictions,
                 estimates=test.estimates)
        results['ck_test'] = {
            'file': 'ck_test.npz',
            'sha256': _sha256_arrays([lagtimes, test.predictions, test.estimates]),
            'max_deviation': float(np.max(np.abs(test.predictions - test.estimates)))}
    return results


def verify_pipeline(config, provenance=None, verbose=True):
    """Rerun a pipeline from scratch and check it reproduces a recorded run.

    Parameters
    ----------
    config : str or dict
        The pipeline file (or its content).
    provenance : str, optional
        The ``provenance.yaml`` to compare with; by default the one in the
        pipeline's output directory.
    verbose : bool, default=True

    Returns
    -------
    mismatches : list of str
        Empty when every step's output, every model summary and every
        validation result is identical.
    """
    cfg = load_config(config)
    if provenance is None:
        provenance = os.path.join(cfg['_base_dir'], os.path.expanduser(str(cfg['output'])),
                                  'provenance.yaml')
    if not os.path.exists(provenance):
        raise PipelineError('no recorded run to verify against: %s' % provenance)
    with open(provenance) as f:
        recorded = yaml.safe_load(f)
    tmp = tempfile.mkdtemp(prefix='msmb-verify-')
    try:
        new = run_pipeline(cfg, output=tmp, force=True, verbose=verbose).provenance
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    mismatches = []
    old_steps = {s['name']: s for s in recorded.get('steps', [])}
    for step in new['steps']:
        old = old_steps.get(step['name'])
        if old is None:
            mismatches.append('step %s: not in the recorded run' % step['name'])
            continue
        for field in ('class', 'params', 'output_sha256', 'timescales', 'transmat_sha256'):
            if old.get(field) != step.get(field):
                mismatches.append('step %s: %s differs' % (step['name'], field))
    for name, res in new['validation'].items():
        if recorded.get('validation', {}).get(name, {}).get('sha256') != res.get('sha256'):
            mismatches.append('validation %s differs' % name)
    if recorded.get('inputs') != new['inputs']:
        mismatches.append('the input files differ from the recorded ones')
    if verbose:
        if mismatches:
            print('NOT reproduced:\n  ' + '\n  '.join(mismatches))
        else:
            print('Reproduced: every output matches %s' % provenance)
    return mismatches
