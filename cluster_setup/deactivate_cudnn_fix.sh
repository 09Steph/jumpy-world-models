#!/usr/bin/env bash
# Counterpart to activate_cudnn_fix.sh -- restores LD_LIBRARY_PATH on
# deactivation so switching envs in the same shell doesn't carry this one's
# cuDNN path along. Installed automatically by setup_env.sh.

export LD_LIBRARY_PATH="$_JUMPY_PRE_CUDNN_FIX_LD_LIBRARY_PATH"
unset _JUMPY_PRE_CUDNN_FIX_LD_LIBRARY_PATH
