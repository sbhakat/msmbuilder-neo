# Vendored Theobald QCP RMSD kernel

These files are copied unchanged from [MDTraj](https://github.com/mdtraj/mdtraj)
release 1.11.2 (`mdtraj/rmsd/src` and `mdtraj/rmsd/include`). They are
licensed under the GNU LGPL v2.1 or later, the same license as MSMBuilder Neo,
and keep their original copyright headers.

`msmbuilder.libdistance` used to link against the `libtheobald.a` static
library that MDTraj installs inside its package. That tied our build to
MDTraj's internal layout and required MDTraj to be importable at build time.
Compiling these sources directly removes both requirements.

Only `msd_atom_major` and `inplace_center_and_trace_atom_major` are used.
The kernel picks NEON on ARM, SSE on x86, or the generic C path when
`__NO_INTRINSICS` is defined.
