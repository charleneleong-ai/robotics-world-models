# Value-aware model learning — fixing the coverage↔accuracy wall

**Status:** ✅ EXECUTED (2026-07-19) — **PARTIAL**: the lever works but doesn't fully close the wall. **Date:** 2026-07-19.

> **Result:** Up-weighting the reward loss (10× + per-frame `×(1+10·reward+30·|Δreward|)`) plus 50% demo-success oversampling in the WM batches moved the reward head's prediction at the demo success state from **0.09 → 0.47 (~5×)** and flipped the gradient from steering-away to holding through the success region — confirming the wall is loss-addressable. But it under-predicts the full 1.0 cliff (a ~0.47 plateau; success frames are still ~0.7% of batches), and task success stays 0 because the actor never learned to *place* (0.035). So value-aware learning is **necessary but not sufficient** — the residual needs a sharper reward-cliff model *and* a placing policy. Full result folded into [`plugchargerdense.md`](../experiments/plugchargerdense.md#follow-up-2--value-aware-model-learning-partially-closes-the-wall).

## Why

[PR #10](../experiments/plugchargerdense.md) closed with a diagnosed, named negative: DreamerV3 cannot solve even PickCube because its **world model is inaccurate in the terminal region the stalling actor never visits**. The decisive measurement — teacher-forcing a solver *success* demo through the trained world model — showed the reward head predicts **~0.09 at the placed/success states where the true reward is 0.59 → 1.0** (a ~10× under-prediction; the value head never spikes). Because CEM planning maximizes the model's *predicted* reward, it cannot target — and actively steers away from — a payoff the model does not represent. Swapping the reactive actor for search did not help.

This is **objective mismatch** ([Lambert et al.](https://hf.co/papers/2002.04523)): DreamerV3 fits its world model by (largely observation-reconstruction-dominated) maximum likelihood over replay, weighting all states roughly equally — so rarely-visited-but-decision-critical success states are under-fit. The literature's fix is **value-aware / task-aware model learning**: weight the model's learning by what matters for control, not by raw likelihood. This spec applies that fix to our own measured problem — the tightest possible follow-up, and it closes the loop on PR #10.

## Hypothesis

If the world model is made to **accurately model the success payoff** — via reward-loss up-weighting ([HarmonyDream](https://hf.co/papers/2310.00344)) plus up-weighting the demo/success transitions in the *world-model* training loss (not just the actor's replay) — then the reward head will predict ~0.59 → 1.0 at the demo's terminal states (vs the measured 0.09), and the SAME planner (or even the reactive actor) will reach success, because it can finally "see" the payoff. If it still fails, the wall is deeper than model fitting (e.g. the RSSM cannot represent the sharp reward cliff, or the coverage↔accuracy coupling is fundamental) — itself a stronger claim.

## Method

Two concrete levers on the existing JAX DreamerV3 setup, ablated independently then combined:

1. **HarmonyDream-style loss harmonization** — DreamerV3's world-model loss is dominated by observation reconstruction; up-weight the **reward-prediction** term (learned harmonization of the observation / reward / dynamics loss scales) so the reward head is fit accurately, especially at reward-transition steps. This is a loss-reweighting change to the WM training, not an architecture change.
2. **Demo/success up-weighting in the *world-model* loss** — get success experience into the reward/value heads directly: sample demo transitions (which contain the full place→static→success sequence) at a fixed elevated ratio in the batches the *world model* trains on (distinct from PR #10's actor-replay prefill, which fed the actor but left the WM's reward head under-fit). Optionally add a value-aware weighting ([VaGraM](https://hf.co/papers/2204.01464)) that fits the model where value is sensitive.

Retrain the PickCube world model with each lever (and combined), then re-run the PR #10 diagnostic and the planner/actor eval.

## Gate — PickCube (same discipline)

The whole finding is on trivial PickCube, so that's the gate. Two measurements, in order:

1. **Model-accuracy (the direct test):** re-run [`wm_diag.py`](../../) — teacher-force the success demo through the retrained WM. **Does the reward head now predict the 0.59 → 1.0 jump at placed/success** (vs the measured 0.09)? This alone confirms whether value-aware learning fixed the *model*, independent of the policy.
2. **Task success:** evaluate the actor (and the PR #10 CEM planner) on the retrained WM. Does `success/avg` climb past 0.5 with grasp holding?

- **GO:** reward head predicts success **and** success climbs → the coverage↔accuracy wall is fixable by value-aware model learning → apply to the PegInsertion / StackCube / PlugCharger ladder (the generative-WM column at last).
- **Partial:** reward head predicts success but the policy still doesn't reach it → the cap moves to the *policy* (planner search / actor), a cleaner separation than PR #10 achieved.
- **NO-GO:** even a correctly-weighted WM under-predicts success → the RSSM cannot represent the reward cliff, or the coupling is fundamental → a deeper negative.

## Honest framing

This applies a *named, recent* method (HarmonyDream / VaGraM / ViVa) directly to the failure mode measured in PR #10, rather than adding an unrelated lever — the most targeted available follow-up. Effort is moderate (loss-reweighting + batch-sampling changes to the WM training, both on the existing setup — no new algorithm). The main risk is that DreamerV3's harmonization / loss-scale internals fight a clean reward-up-weighting; if so, the demo-up-weighting-in-WM-loss lever is the simpler fallback and independently informative.

## Deliverable

- WM-loss reweighting + demo-in-WM-loss changes in the `jax_dreamer` DreamerV3 (isolated env on `pi-a100-80gb`).
- The re-run diagnostic: reward-head prediction at the demo terminal, retrained-WM vs the PR #10 baseline (0.09 vs true 1.0).
- PickCube gate result (model-accuracy + task success), folded into [`plugchargerdense.md`](../experiments/plugchargerdense.md) as the resolution of the generative-WM thread.

## Reusable assets (already on `pi-a100-80gb`)

`~/dreamerv3_jax/` (JAX DreamerV3), the CEM planner in `dreamerv3/agent.py plan()`, the WM diagnostic `~/dreamerv3_jax/wm_diag.py`, the `embodied.envs.maniskill` adapter (force-demo + grasp/placed/static logging), PickCube demos at `~/demos/pickcube_mp/`. Env `jax_dreamer` (py3.11, jax[cuda12] 0.4.33, gymnasium 0.29.1). W&B project `chaleong/wm-manip`.
