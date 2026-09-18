# Pre-registration — PEC seed farm (E3/E5/E0/E6, profile `farm`)

Written **before** aggregation of farm results (only E6 seed 98 had been
executed as a worker smoke test at write time; no E3/E5/E0 farm numbers were
seen). Timestamped in git history.

## What is running

10 paired seeds × {E3 6-condition fault matrix, E5 hold task,
E0 identifiability, E6 identifiability audit}, profile `farm`
(n=96–128 envs, 10–12 PPO iters per condition, 2 torch threads per worker,
8 parallel workers). Budgets are ~4× below the Kaggle `full` profile; this is
declared up front. The purpose is **statistical power through seed count**,
not effect-size resolution: any claim that survives here at reduced budget is
expected to strengthen on GPU.

## Hypotheses and decision criteria (fixed in advance)

### H1 (primary) — Inference vs randomization, E3: zcond > dr
- Test: paired Wilcoxon over 10 seeds on post-fault return (t=75 fault,
  gain ×0.5, lag ×2; training used per-episode randomized faults for dr,
  healthy env for zcond).
- **Pass**: one-sided p < 0.05, diff > 0, Cohen's d ≥ 0.5.
- **Fail**: diff ≤ 0 or p ≥ 0.05. A fail is reported as "randomization
  matches posterior conditioning at reduced budget" and the GPU run decides.

### H2 (primary) — Inference vs distillation, E3: zcond > rma
- Same test structure (rma = RMA two-stage protocol in-suite).
- **Pass**: one-sided p < 0.05, diff > 0.
- **Fail**: reported symmetrically with H1; if both fail, the paper's
  architecture claim (dreams conditioned on inferred physics) is narrowed to
  the already-established zcond>blind and learned≈oracle results.

### H3 (secondary) — oracle ≥ zcond ≥ blind ordering holds
- Checked as consistency: oracle_vs_zcond diff should not be significantly
  negative; zcond_vs_blind should be positive (it was decisive, d=1.22, on
  the T4 artifact).

### H4 (secondary) — Task generality, E5: zcond > dr on hold
- Same criteria as H1 on the hold task (drift a_g = 1.2 m/s²).
- Declared risk: at farm budget the hold policy may be undertrained; if the
  best condition's mean return is within ~10% of the passive-drift floor
  (≈ −113 at n=96), E5 is declared **unresolved at budget**, not failed.

### H5 (secondary) — Identifiability structure, E0/E6
- E0: encoder NRMSE < mean-predictor NRMSE on ≥ 7/10 seeds (paired).
- E6 audit: encoder beats mean-predictor on the identifiable channels
  (μ, k, g, τ) in ≥ 7/10 seeds; a **mass tie is the predicted honest
  outcome** (Proposition 1: mass scale only weakly identifiable from
  box-only data). Mass accuracy beyond the prior is flagged, not celebrated.
- E6 sweep: identifiable-channel NRMSE decreasing with probe amplitude in
  mean across seeds (persistent-excitation signature).

## Analysis plan

`aggregate_farm.py` pools per-seed JSONs; paired statistics from
`pec2.statx.paired_report` (10k bootstrap CIs, two-sided Wilcoxon, Cohen's d;
H1/H2 evaluated one-sided as declared). No conditions are added, dropped, or
re-derived after unblinding; secondary analyses are labeled as such.

## Public artifact

Farm raw per-seed JSONs are committed under `results/seedfarm/` in the
public code repository; this file is committed before unblinding.

## Addendum (pre-unblinding, E8 v2 and E5v2 protocols)

Date: before any E8-v2 or E5v2 seed result exists. Committed to the public
repository prior to analysis of these runs.

### E8 (MuJoCo replication) — v1 invalid, v2 protocol fixed
E8 v1 was invalidated before analysis on a task-calibration ground, not a
statistical one: the v1 adapter mapped the property mu to Coulomb floor
friction (mu*m*g up to ~12 N against an 8 N force cap), pinning the box;
post-fault returns equalled pre-fault returns for every condition, i.e. the
manipulation had no effect to recover from. The v1 per-seed files are archived
under results/seedfarm/e8_v1_pinned_box/ (marked invalid; excluded from all
statistics). The v2 protocol maps mu to the suite semantics (linear velocity
drag via free-joint dof damping), k to contact stiffness (solref direct), and
removes a velocity-zeroing side effect of property application. The v2
pre-registered claims, identical to E3: zcond > dr and zcond > rma in
post-fault return (paired, one-sided as in H1/H2), n=6 seeds.

### E5v2 (hold_v2) — declared follow-up, not a substitution
The original E5 hold task proved unlearnable at farm budget (trained policy
below the passive floor; no approach gradient in the reward). E5's
pre-registered seeds complete and are reported as-is (expected unresolved).
E5v2 (hold_v2: approach curriculum added, verified learnable: return approx
-1.2 vs passive floor -113 in the mechanism probe) runs as a *declared
follow-up* with the same 4-condition matrix and analysis. Its role is
generality of the property-inference advantage across task families, not
salvage of E5.
