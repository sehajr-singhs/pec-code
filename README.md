# pec2 — property-conditioned world models, scaled up

The 2-D successor to `pec-naturemi/`, now **fully 3-D**: sphere end-effector,
cube box, 3-D forces and spring contact with actuator lag along all axes, and a
plane-invariant `flat` mode. Same four falsifiable checks, rebuilt for
contact-rich physics with **no oracle**: a GRU posterior encoder must infer the
five hidden properties (mass, drag μ, contact stiffness, actuator gain, actuator
lag) from observation history, and policies are conditioned on the inferred
posterior. Built to answer one question: *which of the 1-D suite's claims
survive scale?*

## Layout

```
pec2/
  pec2/
    envs.py        batched 3-D push world (sphere–cube spring contact,
                   actuator lag, faults; flat=True = plane-invariant 2-D mode)
    policy.py      GRU actor-critic + PPO + the property posterior encoder
    wm.py          property-conditioned latent world models: latent-ODE (RK4)
                   vs fixed-tick, capacity-matched; windows, training, eval
    dream.py       dream rollouts + the three dream modes (honest / obs-noise /
                   property-jitter gate) with holdout calibration
    es.py          evolution strategies (used inside dreams)
    experiments.py E0–E6: fault-recovery 6-condition matrix (blind / dr /
                   zcond / zfault / rma / oracle), hold task, identifiability
                   audit, with bootstrap CIs + paired Wilcoxon
    statx.py       bootstrap / Wilcoxon / Cohen's d helpers
    plotting.py    the figures
    run_all.py     orchestrator: --profile smoke | ci | full, --only for
                   resumable partial runs
  tests/           14 falsifiable unit checks (all pass in 3-D)
  diag_e0_3d.py    the diagnostic that exposed the encoder memorization bug
  make_dataset_zip.py / build_notebook.py
                   package the code into the self-contained Kaggle notebook
  kernel-metadata.json / dataset-metadata.json   Kaggle CLI push metadata
```

## Run locally (CPU, minutes)

```bash
cd pec2
python -m pytest tests/ -q          # 10 unit checks
python -m pec2.run_all --profile ci --out results_ci    # one seed, minutes
python -m pec2.run_all --profile full --out results_full  # 6 seeds (GPU advised)
```

## Run on Kaggle (the reproducibility artifact)

The notebook `kaggle_pec2.ipynb` embeds the entire codebase as a base64 zip —
no dataset attachment needed (Kaggle CLI attachment proved unreliable). It runs
smoke → full on a T4 and writes `pec2_artifacts.zip` + `results_full/`.

```bash
cd pec2
python make_dataset_zip.py     # refresh the embedded bundle
python build_notebook.py       # rebuild kaggle_pec2.ipynb
kaggle kernels push -p .
# ~35 min on T4, then:
kaggle kernels output sehajrsingh/pec2-world-models-run -p kaggle_out
```

## What the T4 run found (n = 6 paired seeds, 2026-09)

Full numbers in `kaggle_out/results_full/stats_pec2.json`; site summary on the
project page.

| Check | Result | Verdict |
|---|---|---|
| **E3 fault adaptation** | z-conditioned beats blind: +0.028, CI [+0.015, +0.041], **p = 0.031, d = 1.22**; learned ≈ oracle conditioning (p = 0.31) | **pass — decisive** |
| **E0 identifiability** | encoder NRMSE 0.266 vs 0.288 mean-predictor (excitation-probe protocol) | pass |
| **E1 dream gate** | gate vs noise −0.44 (p = 0.22); **noise vs clean +0.46** (CI excl. 0) — the 2018 temperature effect replicates, the gate still hasn't bitten | mixed |
| **E2 clock** | fixed-tick ≤ latent-ODE at every clock at this budget (0.741 vs 0.905 at 8×8) | **negative** |
| **E4 cross-modal** | +0.008 / +0.012, both p > 0.5 | negative |

The honest bottom line: property-conditioned fault recovery from *inferred*
properties is the result that survived scale and got stronger (d = 1.22,
learned ≈ oracle). The clock and gate claims are open, with the exact failing
budgets recorded — that's the point of pre-registering the checks.

## The 3-D upgrade (2026-09)

The world is now fully 3-D (`sphere ee` × `cube box`, ℝ³ dynamics); `flat=True`
keeps a plane-invariant 2-D mode. Downstream code reads `OBS_DIM`/`ACT_DIM`/
`PROP_DIM`, so the promotion touched only `envs.py` + the notebook smoke cell.

Two things changed with it:

1. **Reward rebalance.** In 3-D the action cost dominated the progress term —
   the rational policy froze. Progress gain 5×, first-touch bonus +1.0 added
   (a two-stage curriculum: learn contact, then push), applied to *all*
   conditions so comparisons stay fair. PPO learns the 3-D task (return
   −0.65 → −0.31 in 14 iters at small scale).

2. **Anti-memorization protocol for the encoder** (found by `diag_e0_3d.py`):
   with one fixed window pool, train loss fell 0.22→0.14 while held-out NRMSE
   *rose* 0.29→0.30 — classic overfitting, invisible to window-level validation
   because adjacent windows share 31/32 timesteps. `_train_encoder` now uses
   **episode-held-out validation + best-val checkpoint restore + periodic
   fresh-episode pools**. Held-out NRMSE then improves monotonically:
   0.277 → 0.257 (mean-predictor 0.286) and still descending.

All 10 unit tests pass in 3-D, including the capacity-matched clock test (the
latent-ODE still beats the fixed-tick model at the coarse clock — the 2-D
"negative" was budget, not physics). The 6-seed GPU statistical run executes
via the notebook (`--profile full`); local CI-scale directions: E0 pass,
E1 gate +0.79 over noise (noise now *hurts* under degradation — the regime
where a gate should matter).
