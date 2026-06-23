import logging
import os
import sys

from master_thesis_code import main

if __name__ == "__main__":
    main.main()
    # --- Force a clean exit at the CLI boundary --------------------------------
    # The ``--generate_figures`` command enables matplotlib LaTeX text rendering
    # (``text.usetex=True`` -- the only command that does), whose latex/dvipng
    # helper subprocesses can leave the interpreter blocked during teardown on
    # the cluster.  Observed on combine job 5148384: all 15 figures were written,
    # then the process idled ~43 min until SLURM killed it at walltime (TIMEOUT),
    # wasting the node and poisoning the job's exit state.  By the time main()
    # returns every command has finished and flushed its output, so bypass the
    # stuck teardown with an immediate clean exit.  Scoped to the
    # ``python -m master_thesis_code`` entrypoint so library/test callers of
    # main.main() are unaffected (os._exit would otherwise kill a hosting pytest).
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
