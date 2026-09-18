# jumpy-world-models

Learning Jumpy World Models. UCL MSc IMLS dissertation, ELEC0054 Project 75.

One transformer, trained offline on trajectories, that takes a state and a
sequence of actions and predicts the state at the **end** of that sequence in a
single forward pass, with no intermediate states generated. The horizon is
sampled per training example, `h ~ U(1, 100)`, so one trained model spans the
whole horizon range. The headline comparison is error against horizon,
measured against matched autoregressive baselines on a frozen test split.

## The three arms

Matched in parameter count and training budget, differing only in how a horizon
is traversed.

| Arm | Label | What it does |
|---|---|---|
| 1 | Jumpy | one forward pass to the endpoint |
| 2 | AR-Endpoint | unrolled, supervised at the endpoint only |
| 3 | AR-Step | unrolled, supervised at every step |

These labels are `ARM_DISPLAY` in `src/eval/figure_style.py`, and the report
uses them throughout.

AR-Endpoint and AR-Step generate intermediate states, so only they can be
compared against the true intermediate trajectory. Every arm reports a stationary-copy floor, and
a climatology floor on continuous representations.

## Layout

```
main.py        argparse, config, logging, hand-off to the runner
config.py      seeds, paths, artefact names, geometries, the Atari game pool
src/data/      the data contract, corpus sources, tokeniser, sampler, split
src/envs/      the NAVIX adapter and the slip wrapper
src/models/    Jumpy, the two autoregressive models, and their shared parts
src/pipeline/  one module per stage, plus sweep, fit, rescore, prune, manifest
src/eval/      metrics, baselines, intermediate states, Atari archive, plots
tests/         pytest suite
cluster_setup/ conda activate.d hooks, installed by setup_env.sh on Linux
results/       Atari ranking and sweeps (git-ignored, regenerated below)
outputs/       run artefacts (git-ignored)
```

Artefacts land in `outputs/<run>/<env>/seed<n>/`, with `arm<k>/checkpoints_<mode>/`,
`arm<k>/eval/`, `data/`, `logs/` and `sentinels/` beneath. `--fast` diverts the
tree to `outputs/fast/`, so a smoke run cannot be mistaken for a reporting run.

**Sentinels decide what is skipped; `--stages` decides what is offered.** A stage
the flag omits is never constructed; a stage it offers still skips on its own
sentinel. Delete `sentinels/<stage>/done.json` to force a rerun.

## Running

```
python main.py --fast
```

**The order is: train and score each cell, then aggregate across seeds, sweep
across arms, fit the exponents, then draw.** Each stage below reads only what
the one before it wrote.

One reporting cell is one run, one environment, one seed, one model, one view.
`--arm` takes a single value, so each model is a separate invocation.

`<run>` below is any name. It is a pure CLI argument and becomes the directory
under `outputs/`, so nothing validates it. The names the report draws from are
declared in `REPORTED_RUNS` in `config.py`, and a run not in that table is not
read by the figure stage.

```
python main.py --run-name <run> --env Navix-FourRooms-v0 --data-seeds 42 \
    --representation symbolic --collection-policy uniform_random --slip 0.0 \
    --obs-mode top_down --arm 2
```

The run axes are `--slip` (0, 0.10, 0.25, refused off the pre-registered grid),
`--collection-policy` (`uniform_random`, `ppo`), `--representation` (`symbolic`,
`rgb`, `greyscale`), `--obs-mode` (`top_down`, `egocentric`), and `--env`.

`--env` accepts only the names declared in `ENVIRONMENT_GEOMETRIES`, and which
axes apply follows the family:

| `--env` | Representation | Collection | Views |
|---|---|---|---|
| `Navix-FourRooms-v0` | `symbolic`, `rgb` | `uniform_random`, `ppo` | both symbolic, `egocentric` on rgb |
| `Navix-DoorKey-Random-5x5-v0` | `symbolic`, `rgb` | `uniform_random`, `ppo` | as above |
| `Navix-Dynamic-Obstacles-16x16-v0` | `symbolic`, `rgb` | `uniform_random`, `ppo` | as above |
| `atari-dqn-replay` | `greyscale` | offline archive | `top_down` |
| `atari-dqn-replay-p24` | `greyscale` | offline archive | `top_down` |
| `atari-dqn-replay-p49` | `greyscale` | offline archive | `top_down` |
| `atari-dqn-replay-long` | `greyscale` | offline archive | `top_down` |
| `nle-katakomba` | `greyscale` | offline archive | `top_down` |

`--slip` applies to NAVIX only. `nle-katakomba` is declared and reachable but
carries no reported result.

**Two card constraints, and the first applies to every environment.**
`main.py` imports `navix` unconditionally, and that import raises on H100
inside `jax.image.resize`, before any `--env` is read. **Nothing in this
repository runs on an H100.** Separately, **PPO collection runs on V100 only**
and fails at run time on A100 inside `jax.lax.scan`. V100 runs everything.

### NAVIX

Both views, both collectors, three slip levels. One invocation per model.

```
python main.py --run-name <run> --env Navix-FourRooms-v0 --data-seeds 42 \
    --representation symbolic --collection-policy uniform_random --slip 0.10 \
    --obs-mode egocentric --arm 3
```

Pixel runs are egocentric and take no slip.

```
python main.py --run-name <run> --env Navix-DoorKey-Random-5x5-v0 \
    --data-seeds 42 --representation rgb --collection-policy ppo --slip 0.0 \
    --obs-mode egocentric --arm 1
```

### Atari

The three positions are a behaviour-quality ladder: a later archive position is
a stronger DQN policy. `atari-dqn-replay-long` is the same corpus converted into
1024-step windows.

**`JWM_ATARI_ROOT` must point at the unpacked archive.** Without it, shard paths
resolve against the repository and raise before any GPU work. There is no slip
and no collector to choose; the corpus is offline.

```
JWM_ATARI_ROOT=/path/to/dqn-replay python main.py --run-name <run> \
    --env atari-dqn-replay-p49 --representation greyscale --obs-mode top_down \
    --data-seeds 42 --arm 1
```

Training to the long horizon takes `--training-horizon-max`, which
`TRAINING_HORIZON_MAX_BY_ENV` sets to 1024 for `atari-dqn-replay-long`.

```
JWM_ATARI_ROOT=/path/to/dqn-replay python main.py --run-name <run> \
    --env atari-dqn-replay-long --representation greyscale --obs-mode top_down \
    --data-seeds 42 --arm 1
```

### Frozen test split

A separate invocation that scores existing checkpoints without reaching
`generate`, `prepare` or `train`.

```
python main.py --source-run <run> --stages evaluate --split test \
    --env Navix-FourRooms-v0 --arm 3 --obs-mode top_down --data-seeds 42 \
    --representation symbolic --collection-policy uniform_random --slip 0.0
```

### Re-scoring against a manifest

Write the manifest from a tree holding the corpus of record, then score. The
pass gates shards against the manifest digests, checks drift, and prunes only
what it regenerated.

```
python main.py --rescore-test-split --write-manifest --runs <run> \
    --manifest-source outputs --manifest outputs/manifests/<run>.json

python main.py --rescore-test-split --runs <run> \
    --manifest outputs/manifests/<run>.json
```

`--data-seeds 42` scores one seed, which is how a corpus too large to hold at
once is cycled through disk. `--regenerate` helps only where regeneration
reproduces: at slip 0.10 and 0.25, never under PPO. `--dry-run` reports the plan
without scoring.

### Cross-seed and cross-arm artefacts

Aggregate, then sweep, then fit. The aggregate refuses to average a series
defined on some seeds and not others, so it runs once the run is whole.

```
python main.py --aggregate --run-name <run>_test --env Navix-FourRooms-v0
python main.py --sweep     --run-name <run>_test --env Navix-FourRooms-v0
python main.py --fit       --run-name <run>_test --env Navix-FourRooms-v0
```

Both read the error metric the aggregates declare and raise rather than assuming
one. `--arm-run 1=<other run>` where a model lives in another run.

### Figures

Last stage, and it reads only the JSON above. It opens no shard and no
checkpoint, so the whole set redraws from a few megabytes of artefacts.

```
python main.py --figures --dry-run
python main.py --figures
python main.py --figures --publish <existing directory>
```

Each figure reports **drawn**, **partial**, **skipped** or **failed**. Partial
drew with a named omission; skipped is a missing artefact; **only failed exits
non-zero**.

`--publish` copies the drawn and partial figures of the reported split only, so
a development twin cannot reach the report. **The destination must already
exist.** `--run-name` filters to one run, `--all` adds the development figures.

**A run draws only if its test cells carry the post-rescore marker and it is
declared in `REPORTED_RUNS`.**

### Intermediate states and displacement

AR-Endpoint and AR-Step only.

```
python main.py --intermediate-states --run-name <run> --env Navix-FourRooms-v0 \
    --truth-source held_out --all-cells
python main.py --displacement-sweep --run-name <run>
```

### Selecting the Atari games

The pool is measured, not chosen: every game is scored on how much of the
screen changes over 100 steps and on episode length, against a displacement bar
and a length gate.

```
python main.py --rank-atari-games --position 0 --out results/atari
python main.py --sweep-atari-checkpoints --out results/atari
```

`ATARI_GAME_POOL` is the position-0 ranking and the five reported games are its
first five. The position 24 and 49 rankings check that pool, they do not set
it.

### Adding an archive position

A position reaches the pipeline **as an environment name**, not a flag, so the
three reported ones need no source change. A fourth is **four declarations in
`config.py` and no logic**, which keeps every position visible in one file.

```python
ATARI_LADDER_F_ENV_NAME: str = "atari-dqn-replay-p37"      # 1. the name
ATARI_LADDER_POSITIONS_F: tuple[int, ...] = (37,)          # 2. which position

ENVIRONMENT_GEOMETRIES = {                                  # 3. screen, grid,
    ATARI_LADDER_F_ENV_NAME: EnvironmentGeometry(...),      #    evaluation grid
}
OFFLINE_SOURCE_BY_ENV = {                                   # 4. which corpus
    ATARI_LADDER_F_ENV_NAME: OFFLINE_SOURCE_ATARI_P37,      #    to convert
}
```

`--env atari-dqn-replay-p37` then works everywhere the other positions do.

## GPU

Pin the card and bound the memory fraction on the command, never exported, so
it cannot leak into a later command on a shared host.

```
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 python main.py ...
```

A card index means nothing without its host, so check occupancy on the host
that will run the command. **Count processes, not memory**: with preallocation
off a running job can report under 1,500 MiB.

## Testing and environment

```
python -m pytest
```

Creating the environment takes two steps, because `environment.yml` cannot
express either of them.

```
conda env create -f environment.yml
conda activate jumpy
bash setup_env.sh
```

`setup_env.sh` installs `flax==0.12.0` with `--no-deps`, the version the
reported results were produced under; `--no-deps` stops pip moving the pinned
CUDA stack. On Linux it also installs the `cluster_setup/` hooks so the
environment's own cuDNN outranks an older system one on `LD_LIBRARY_PATH`.
**Without that hook `jax.devices()` still succeeds and the first cuDNN op
fails.** Re-run it after any `environment.yml` change.

One environment name everywhere, local and cluster, since both share this file.
