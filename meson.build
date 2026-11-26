project(
  'msmbuilder-neo',
  version: '0.1.0.dev0',
  license: 'LGPL-2.1-or-later',
  meson_version: '>= 1.4.0',
)

# MSMBuilder Neo is pure Python: the numerical kernels are compiled at run
# time by Numba, and RMSD comes from mdtraj. Meson only installs the package
# files and writes version.py, so wheels are platform independent.
py = import('python').find_installation(pure: true)

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
