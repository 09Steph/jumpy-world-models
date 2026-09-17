#!/usr/bin/env bash
# Conda activate.d hook: prioritise this env's own pip-installed cuDNN over
# an older system cuDNN some cluster hosts put on LD_LIBRARY_PATH via
# CUDNN_HOME in ~/.bashrc. Without this, jaxlib silently loads the wrong
# (too-old) cuDNN and any op that touches it fails with
# `FAILED_PRECONDITION: DNN library initialization failed` --
# jax.devices() alone never exercises cuDNN, so it won't catch this.
#
# Installed automatically by setup_env.sh; not meant to be run directly.
# $CONDA_PREFIX is set correctly by conda before this hook runs, so it
# works under whatever name this env is activated as.

export _JUMPY_PRE_CUDNN_FIX_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
