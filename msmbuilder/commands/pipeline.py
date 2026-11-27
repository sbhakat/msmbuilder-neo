"""Run a whole analysis (featurization, tICA, clustering, MSM, validation)
from one YAML pipeline file, recording parameters, package versions and
checksums of inputs and outputs in provenance.yaml.

Steps whose inputs and parameters are unchanged since the last run are
reused. Use --verify to rerun everything from scratch and check that it
reproduces the recorded results, and --template to print an example
pipeline file.
"""
import sys

from ..cmdline import Command, argument


class Pipeline(Command):
    _group = '0-Pipeline'
    _concrete = True
    description = __doc__
    config = argument('config', nargs='?', default=None,
                      help='Pipeline YAML file')
    output = argument('-o', '--output', default=None,
                      help="Output directory (overrides the file's 'output')")
    force = argument('--force', action='store_true',
                     help='Recompute every step instead of reusing unchanged ones')
    verify = argument('--verify', action='store_true',
                      help='Rerun from scratch and check the recorded results are reproduced')
    template = argument('--template', action='store_true',
                        help='Print an example pipeline file and exit')
    quiet = argument('-q', '--quiet', action='store_true', help='Print less')

    def __init__(self, args):
        self.args = args

    def start(self):
        from ..pipeline import (EXAMPLE_CONFIG, PipelineError, run_pipeline,
                                verify_pipeline)
        if self.args.template:
            sys.stdout.write(EXAMPLE_CONFIG)
            return
        if self.args.config is None:
            self.error('give a pipeline file (or --template for an example)')
        try:
            if self.args.verify:
                provenance = None
                if self.args.output is not None:
                    import os
                    provenance = os.path.join(self.args.output, 'provenance.yaml')
                mismatches = verify_pipeline(self.args.config, provenance=provenance,
                                             verbose=not self.args.quiet)
                if mismatches:
                    sys.exit(1)
            else:
                run_pipeline(self.args.config, output=self.args.output,
                             force=self.args.force, verbose=not self.args.quiet)
        except PipelineError as e:
            self.error('pipeline error: %s' % e)
