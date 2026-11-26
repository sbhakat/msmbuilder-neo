project(
  'msmbuilder-neo',
  'c', 'cpp', 'cython',
  version: '0.1.0.dev0',
  license: 'LGPL-2.1-or-later',
  meson_version: '>= 1.4.0',
  default_options: [
    'buildtype=release',
    'b_ndebug=if-release',
    'c_std=c11',
    'cpp_std=c++14',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
np_dep = dependency('numpy')

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# OpenMP is optional: without it the parallel loops run on one thread.
#
# On macOS with Apple clang, Meson's OpenMP lookup always picks Homebrew's
# libomp. Inside a conda environment NumPy, SciPy and scikit-learn already load
# conda's libomp, and two OpenMP runtimes in one process crash or deadlock.
# So prefer a libomp that ships with the target Python's prefix when there is
# one, and fall back to Meson's normal lookup otherwise.
omp_dep = disabler()
omp_c_dep = disabler()
if not get_option('openmp').disabled() and host_machine.system() == 'darwin' and cc.get_id() == 'clang'
  py_prefix = py.get_variable('prefix', '')
  if py_prefix != ''
    prefix_omp = cc.find_library('omp', dirs: [py_prefix / 'lib'], required: false)
    if prefix_omp.found() and cc.has_header('omp.h', args: ['-I' + py_prefix / 'include'])
      omp_dep = declare_dependency(
        compile_args: ['-Xpreprocessor', '-fopenmp', '-I' + py_prefix / 'include'],
        dependencies: prefix_omp)
      omp_c_dep = omp_dep
      message('Using libomp from the Python prefix: ' + py_prefix / 'lib')
    endif
  endif
endif
if not omp_dep.found()
  omp_dep = dependency('openmp', language: 'cpp', required: get_option('openmp'))
  omp_c_dep = dependency('openmp', language: 'c', required: get_option('openmp'))
endif
if omp_dep.found()
  message('OpenMP enabled')
else
  message('OpenMP not found; building single-threaded kernels')
endif

# The HMM fitters use SSE intrinsics on x86 and sse2neon on ARM.
simd_args = []
if host_machine.cpu_family() in ['x86', 'x86_64'] and cpp.get_argument_syntax() == 'gcc'
  simd_args += ['-mssse3']
endif

# Quiet warnings from generated Cython code and old C sources.
quiet_c = cc.get_supported_arguments('-Wno-unused-function', '-Wno-unreachable-code',
                                     '-Wno-unused-but-set-variable', '-Wno-unused-variable')
quiet_cpp = cpp.get_supported_arguments('-Wno-unused-function', '-Wno-unreachable-code',
                                        '-Wno-unused-but-set-variable', '-Wno-unused-variable',
                                        '-Wno-deprecated-declarations')
add_project_arguments(quiet_c, language: 'c')
add_project_arguments(quiet_cpp, language: 'cpp')

cython_args = ['-3', '--fast-fail']
src_inc = include_directories('msmbuilder/src')

# --- version.py ---------------------------------------------------------------
git_rev = run_command('git', 'rev-parse', 'HEAD', check: false)
version_conf = configuration_data()
version_conf.set('VERSION', meson.project_version())
_v = meson.project_version().split('.')
version_conf.set('SHORT_VERSION', _v[0] + '.' + _v[1])
version_conf.set('GIT_REVISION', git_rev.returncode() == 0 ? git_rev.stdout().strip() : 'Unknown')
version_conf.set('RELEASE', meson.project_version().contains('dev') ? 'False' : 'True')
version_py = configure_file(input: 'msmbuilder/version.py.in', output: 'version.py',
                            configuration: version_conf)
py.install_sources(version_py, subdir: 'msmbuilder')

# --- pure Python modules and package data -------------------------------------
listing = run_command(py, files('tools/list_package_files.py'), meson.project_source_root(),
                      check: true).stdout().strip().split('\n')
foreach f : listing
  parts = f.split('/')
  subdir_parts = []
  foreach p : parts
    if p != parts[-1]
      subdir_parts += p
    endif
  endforeach
  py.install_sources(f, subdir: '/'.join(subdir_parts))
endforeach

# --- compiled extensions --------------------------------------------------------
py.extension_module('_muller',
  'msmbuilder/example_datasets/_muller.pyx',
  cython_args: cython_args,
  dependencies: [np_dep],
  install: true, subdir: 'msmbuilder/example_datasets')

py.extension_module('_markovstatemodel',
  'msmbuilder/msm/_markovstatemodel.pyx',
  'msmbuilder/msm/src/transmat_mle_prinz.c',
  cython_args: cython_args,
  include_directories: include_directories('msmbuilder/msm/src'),
  dependencies: [np_dep],
  install: true, subdir: 'msmbuilder/msm')

py.extension_module('_metzner_mcmc_fast',
  'msmbuilder/msm/_metzner_mcmc_fast.pyx',
  'msmbuilder/msm/src/metzner_mcmc.c',
  cython_args: cython_args,
  include_directories: include_directories('msmbuilder/msm/src'),
  dependencies: [np_dep, omp_c_dep],
  install: true, subdir: 'msmbuilder/msm')

py.extension_module('_ratematrix',
  'msmbuilder/msm/_ratematrix.pyx',
  cython_args: cython_args + ['--cplus', '-I', meson.current_source_dir() / 'msmbuilder/src'],
  override_options: ['cython_language=cpp'],
  include_directories: src_inc,
  dependencies: [np_dep, omp_dep],
  install: true, subdir: 'msmbuilder/msm')

py.extension_module('_speigh',
  'msmbuilder/decomposition/_speigh.pyx',
  cython_args: cython_args + ['--cplus', '-I', meson.current_source_dir() / 'msmbuilder/src'],
  override_options: ['cython_language=cpp'],
  include_directories: src_inc,
  dependencies: [np_dep, omp_dep],
  install: true, subdir: 'msmbuilder/decomposition')

py.extension_module('test_cyblas',
  'msmbuilder/tests/test_cyblas.pyx',
  cython_args: cython_args + ['-I', meson.current_source_dir() / 'msmbuilder/src'],
  include_directories: src_inc,
  dependencies: [np_dep],
  install: true, subdir: 'msmbuilder/tests')

theobald_inc = include_directories('msmbuilder/libdistance/theobald')
py.extension_module('libdistance',
  'msmbuilder/libdistance/libdistance.pyx',
  'msmbuilder/libdistance/theobald/theobald_rmsd.cpp',
  'msmbuilder/libdistance/theobald/center.cpp',
  cython_args: cython_args + ['--cplus'],
  override_options: ['cython_language=cpp'],
  cpp_args: simd_args,
  include_directories: [include_directories('msmbuilder/libdistance/src'), theobald_inc],
  dependencies: [np_dep],
  install: true, subdir: 'msmbuilder')

py.extension_module('_kmedoids',
  'msmbuilder/cluster/_kmedoids.pyx',
  'msmbuilder/cluster/src/kmedoids.cc',
  cython_args: cython_args + ['--cplus'],
  override_options: ['cython_language=cpp'],
  include_directories: include_directories('msmbuilder/cluster', 'msmbuilder/cluster/src'),
  dependencies: [np_dep],
  install: true, subdir: 'msmbuilder/cluster')

hmm_inc = include_directories('msmbuilder/hmm', 'msmbuilder/hmm/src/include',
                              'msmbuilder/hmm/src', 'msmbuilder/hmm/cephes')

# The HMM fitters are C++ classes that call back into Cython through
# `cdef public` functions, so the C++ sources need the header Cython writes
# next to the generated module. Run Cython explicitly so that header is a
# declared output and is built before the C++ sources that include it.
cython = find_program('cython')
foreach hmm : [
    ['gaussian', ['msmbuilder/hmm/src/GaussianHMMFitter.cpp']],
    ['vonmises', ['msmbuilder/hmm/src/VonMisesHMMFitter.cpp',
                  'msmbuilder/hmm/cephes/i0.c', 'msmbuilder/hmm/cephes/chbevl.c']],
  ]
  hmm_name = hmm[0]
  hmm_cy = custom_target(hmm_name + '_cython',
    input: 'msmbuilder/hmm/' + hmm_name + '.pyx',
    output: [hmm_name + '.cpp', hmm_name + '.h'],
    command: [cython, '-3', '--fast-fail', '--cplus', '@INPUT@', '-o', '@OUTPUT0@'],
  )
  py.extension_module(hmm_name,
    hmm_cy, hmm[1],
    cpp_args: simd_args,
    include_directories: hmm_inc,
    dependencies: [np_dep, omp_dep],
    install: true, subdir: 'msmbuilder/hmm')
endforeach
