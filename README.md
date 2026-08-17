# jumpy-world-models

Learning Jumpy World Models. UCL MSc IMLS dissertation, ELEC0054 Project 75.

One transformer, trained offline on trajectories, that takes a state and a
sequence of actions and predicts the state at the **end** of that sequence in a
single forward pass, with no intermediate states generated. The horizon is
sampled per training example, `h ~ U(1, 100)`, so one trained model spans the
whole horizon range. The headline comparison is error against horizon versus a
matched single-step autoregressive baseline, whose error is expected to compound
where the direct model's is not.

## Experiments

- **E1 -- does it learn?** And if not, is the limit capacity, dataset size or
  data quality?
- **E2 -- error against horizon.** The headline: direct prediction against the
  autoregressive comparator, five seeds.
- **E3 -- stochastic dynamics.** Conditional. First on the cut list.
- **E4 -- observability.** Full top-down grid against an egocentric partial
  view. Conditional.

Every arm reports a stationary-copy baseline. Most of a grid is static between
two timesteps, so a model that simply copies its input scores highly while
having learned nothing about dynamics.

## Layout

```
main.py                     thin orchestrator (argparse, config, logging, stages)
config.py                   seeds, paths, artefact filenames, hyperparameters
src/
  data/                     (empty)
  models/    actions.py     action-space handling
  envs/      navix_env.py   the NAVIX adapter
  pipeline/  base.py        Stage base class, sentinel skipping
             aggregate.py, aggregate_stats.py
                            cross-seed mean/std and rliable IQM + bootstrap CI
  utils/                    paths, logging, determinism, sentinels,
                            platform guard
tests/                      pytest suite
outputs/                    run artefacts (git-ignored; empty skeleton)
```

Run artefacts are written to `outputs/<run_name>/seed<n>/`, with
`checkpoints/`, `eval/`, `logs/` and `sentinels/` beneath. `--fast` diverts the
whole tree to `outputs/fast/`, so a smoke run can never be mistaken for a
reporting run.

Arriving with the build: the trajectory source, store, window sampler and
tokeniser under `src/data/`; the direct transformer and its matched baseline
under `src/models/`; and a new `src/eval/` holding the metrics, the
stationary-copy baseline and the error-against-horizon sweep.

## Running

```
python main.py --seed 42 --run-name h1_baseline
python main.py --fast --seed 42 --run-name smoke
```

`--fast` is the smoke path: reduced sizes, results explicitly not for reporting.

## Testing

```
python -m pytest
```
