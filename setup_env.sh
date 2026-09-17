#!/usr/bin/env bash
# Finish setting up the active conda environment from environment.yml.
#
# Two things environment.yml cannot do.
#   1. Install flax==0.12.0 with --no-deps. --no-deps is not resolving a
#      conflict: flax 0.12.0 asks for jax>=0.7.1 and the pin satisfies it. It
#      stops pip pulling flax's own jax, optax and orbax and moving the pinned
#      CUDA stack underneath a working environment.
#   2. On Linux, install an activate.d/deactivate.d hook from cluster_setup/ so
#      the env's own cuDNN outranks an older system one on LD_LIBRARY_PATH.
#
# First time:          conda env create -f environment.yml
#                      conda activate jumpy && bash setup_env.sh
# Afterwards:          conda activate jumpy && bash setup_env.sh
#
# By hand, if this will not run:
#   conda env update -n jumpy -f environment.yml --prune
#   pip install --no-deps flax==0.12.0
#   cp cluster_setup/*activate_cudnn_fix.sh into $CONDA_PREFIX/etc/conda/{de,}activate.d/
#
# CAVEAT ON THE PIN. 0.12.0 is pinned because the reported results were
# produced under it. 0.12.7 also runs this codebase and is what the
# figure-drawing machine uses, but it asks for jax>=0.10.0 and so needs
# --no-deps to coexist with the pin. The figure stage reads stored JSON and
# computes nothing, so either redraws a plot identically. Use 0.12.0 to
# reproduce a number.
#
# Verify after. The second command is the test, because jax.devices() succeeds
# even when cuDNN is wrong:
#   python -c "import jax, flax; print(jax.__version__, flax.__version__)"
#   python -c "import jax, jax.numpy as jnp; print(jnp.ones((1,1,4,4)).sum(), jax.devices())"

set -euo pipefail

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "No conda environment is active." >&2
    echo "First time? conda env create -f environment.yml && conda activate jumpy" >&2
    echo "Otherwise:  conda activate jumpy" >&2
    exit 1
fi

echo "Updating conda environment '$CONDA_DEFAULT_ENV' from environment.yml..."
conda env update -n "$CONDA_DEFAULT_ENV" -f environment.yml --prune

echo "Installing flax==0.12.0 (see header comment above)..."
pip install --no-deps flax==0.12.0

if [[ "$(uname -s)" == "Linux" ]]; then
    if [[ ! -d cluster_setup ]]; then
        # set -e would abort the whole script on a missing directory. The hook
        # is a cluster convenience, not a requirement, so skip it and let the
        # rest of the setup and its verification commands stand.
        echo "cluster_setup/ is absent, so the cuDNN hook was not installed." >&2
        echo "It is only needed where an older system cuDNN sits on LD_LIBRARY_PATH." >&2
    else
        echo "Linux detected -- installing the cuDNN LD_LIBRARY_PATH hook..."
        mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
        cp cluster_setup/activate_cudnn_fix.sh "$CONDA_PREFIX/etc/conda/activate.d/"
        cp cluster_setup/deactivate_cudnn_fix.sh "$CONDA_PREFIX/etc/conda/deactivate.d/"
        echo "Installed. Run 'conda deactivate && conda activate $CONDA_DEFAULT_ENV' to"
        echo "apply it now -- every activation after that picks it up automatically."
    fi
fi

echo "Done. Verify with:"
echo "    python -c \"import jax; print(jax.devices())\""
echo "    python -c \"import flax; print(flax.__version__)\""
echo "    python -c \"import jax, jax.numpy as jnp; print(jnp.asarray([1,2,3]).devices())\""
