# How the Phase-4 adapters are trained "without data" — theory and approach

Written 2026-08-18 by the orchestrator, from the preregistration (`notes/preregistration_v5_phase4.md`,
amendment A5) and the code that actually ran (`src/dpo_data.py`, `scripts/build_dpo_pairs.py`,
`src/dpo_train_modal.py`, `scripts/train_dpo.py`). Numbers are the real ones from
`results/dpo/build_manifest.json` and `results/dpo/pairs_summary.md`.

## 0. The short answer

There *is* training data — about 3,500 examples — but none of it was written by a person. It is
**the model's own outputs, labelled by the pinned LLM judge**, turned into preference pairs and optimised
with **Direct Preference Optimization (DPO)** on a **QLoRA** adapter. Every ingredient is a standard,
published technique; what makes it cheap is the combination:

| ingredient | what it removes | cost here |
|---|---|---|
| self-generated candidates (RLAIF-style) | any need to *write* target responses | 3,499 samples from vLLM, ≈ 25 min on one L40S |
| the locked judge as the preference oracle | any need to *hand-label* them | 3,500 judge calls, USD 5.69 |
| DPO | the reward model and the RL loop of classic RLHF | one supervised loss |
| QLoRA (4-bit frozen base + low-rank adapter) | full-model finetuning memory | 84 optimiser steps, ≈ 9 min on one A100-40GB, ≈ USD 1 per arm |

## 1. Where the training signal comes from (RLAIF, not RLHF)

Classic RLHF (Ouyang et al., 2022) collects *human* preference labels, trains a reward model on them,
then optimises the policy with RL against that reward under a KL penalty. Constitutional AI / RLAIF
(Bai et al., 2022; Lee et al., 2023) showed the human labeller can be replaced by an AI labeller working
from a written rubric. That is exactly our setting, with two twists that the *experiment* needs:

1. **The oracle is the study's own judge.** The rubric is `configs/judge_rubric.md` (hash-locked;
   sha256 `c996e4c6…`), the model `claude-sonnet-4-6` at temperature 0 — the same instrument that scores
   the "distress" channel everywhere else in the project. So arm A is trained to lower *precisely the
   quantity the report calls distress language*, not some proxy. (Its limits are known: the human audit,
   `results/summaries/judge/human_audit.md`, shows both raters floor-bound with MAE 0.57 — the judge is a
   coarse but consistent oracle.)
2. **The candidates are on-policy.** Both the chosen and the rejected response are things the base model
   itself said in that context at T = 0.8. This matters twice: ethically (nobody writes dysphoric or
   suppressive text by hand — the roadmap's "no dysphoric optimisation" line), and technically (DPO is
   known to work best when the pairs sit inside the policy's own distribution; off-policy pairs give a
   weak, noisy gradient).

## 2. Building the pairs (`scripts/build_dpo_pairs.py`, `src/dpo_data.py`)

1. **Fresh contexts, firewalled.** 600 four-option items from `allenai/ai2_arc` (ARC-Challenge + ARC-Easy,
   *train* split, CC-BY-SA-4.0), selected by hash rank, checked against all 40 locked DGS tasks by
   normalised stem text, stem SHA-256, and stem+answer SHA-256 (0 overlaps; 8 intra-ARC duplicates
   dropped). Nothing the adapters see is ever evaluated on, and the discovery/holdout tasks never enter
   training. A further 200 items *after* the training slice are reserved for the capability probe.
2. **The adverse context = the experiment's hostile onset.** Each item is asked with the frozen
   `Answer: X` instruction; the model answers greedily; only *correct* greedy answers are kept (573/600,
   95.5 %). The frozen hostile bogus-failure message is then appended — so the training context is the
   same construct as the "hostile onset" cell of the factorial (a correct answer falsely called wrong,
   rudely), the place where gemma-2-9b-it emits distress language.
3. **Candidates.** 4 responses per context at T = 0.8 (seeds 1–4); under amendment A5 the 302
   highest-distress contexts got 4 more (seeds 5–8). 3,499 candidates in total, every one judged.
4. **Pairs.** *Arm A (distress-suppression):* chosen = lowest judged distress, rejected = highest, in the
   same context. The preregistered gap ≥ 3 yielded 39 pairs (98 after the top-up), because judged distress
   on these candidates is compressed (mean 1.39, max 4 of 10); the pre-stated fallback (A5 branch iii)
   fired: gap ≥ 2 → **329 pairs** (chosen mean 0.34, rejected mean 2.67). *Arm B (placebo):* on the same
   contexts, chosen = shorter, rejected = longer (≥ 40 tokens apart), subsampled deterministically to the
   same 329 (82 % on identical contexts to A). B is the control that says "any DPO on these prompts,
   with these hyperparameters, moves the model this much for reasons unrelated to distress".

## 3. DPO — the loss, and why it needs no reward model

RLHF's objective is  max_π E_{x,y∼π}[r(x,y)] − β·KL(π(·|x) ‖ π_ref(·|x)).  Its optimum has the closed
form  π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β).  Rafailov et al. (2023) invert this: the reward is recoverable
from the policy as  r(x,y) = β·log[π(y|x)/π_ref(y|x)] + const.  Substituting into the Bradley–Terry
preference model P(y_w ≻ y_l) = σ(r_w − r_l) gives a plain supervised loss on preference pairs:

    L_DPO = − E_(x, y_w, y_l) [ log σ( β·( log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]

Reading it: raise the log-probability of the chosen response *relative to the reference model* and lower
the rejected one, with β (= 0.1 here) setting how far the policy may drift from the reference before the
implicit KL term dominates. No reward model is trained and no sampling happens during training — one
forward/backward pass per pair over four sequence log-likelihoods (policy and reference, chosen and
rejected). TRL's `DPOTrainer` (loss_type `sigmoid`) is that formula. The quantities TRL logs are the
implicit rewards: `rewards/margins` = β·(Δ log-ratio) between chosen and rejected (arm A finished at 3.3),
`rewards/accuracies` = fraction of pairs whose margin is positive (arm A: 1.00 — every training pair is
now ranked the way the judge ranked it; with 329 pairs and 2 epochs that is expected and says nothing
about generalisation, which is what the discovery factorial measures).

**Reference policy without a second model.** With PEFT, the reference is the same network with the
adapter *disabled* (`ref_model=None`), so the 9B weights are loaded once. The reference is therefore
exactly the model that produced the candidates.

## 4. LoRA / QLoRA — why a 9B model trains on one GPU in minutes

*LoRA* (Hu et al., 2021): freeze every weight matrix W and learn a low-rank update, W' = W + (α/r)·B·A
with A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}, r ≪ d. Here r = 16, α = 32, dropout 0.05, on all seven projections
of every layer (q, k, v, o, gate, up, down): roughly 54 M trainable parameters, ≈ 0.6 % of the 9.2 B —
an adapter file of ≈ 100 MB.

*QLoRA* (Dettmers et al., 2023): keep the frozen base in 4-bit NF4 (double-quantised) and compute in
bf16, so the 9B base occupies ≈ 5–6 GB and, with gradient checkpointing, the whole job fits an A100-40GB.
Both arms use identical settings — the only difference between A and B is which response the pair file
calls "chosen":

| setting | value |
|---|---|
| base | `google/gemma-2-9b-it` @ `11c9b309…` (the pinned primary) |
| quantisation | 4-bit NF4, double quant, bf16 compute |
| LoRA | r 16 · α 32 · dropout 0.05 · q,k,v,o,gate,up,down |
| DPO | β 0.1 · loss `sigmoid` · reference = adapter-disabled base |
| optimisation | lr 5e-6 · cosine · 10 % warm-up · 2 epochs · batch 2 × grad-accum 4 = 8 · seed 0 (also `data_seed`) |
| lengths | max_length 1536 (prompt ≈ 400 + up to 512-token completion) |
| steps | 329 pairs × 2 epochs / 8 = 84 optimiser steps ≈ 9 min (arm A: train_runtime 526 s, train_loss 0.34) |
| software | torch 2.13.0 · transformers 5.15.0 · trl 1.10.0 · peft 0.20.0 · bitsandbytes 0.50.1, pinned in the Modal image |

**Merge, then serve.** After training, ΔW = (α/r)·B·A is added into the bf16 base weights
(`merge_and_unload`) and saved as an ordinary Hugging Face checkpoint (`/adapters/<arm>/merged` on the
`dgs-adapters` volume, ≈ 18 GB) next to the raw adapter (`/adapters/<arm>/lora`). vLLM then serves the
merged model through the *same* stack, flags and logprob path as every other model in the study
(`src/serve_modal.py`, `DGS_MODEL_PATH`), so M1/M2 extraction under arms 0, A and B differs in nothing but
the weights. (Merging into bf16 a delta that was fitted against a 4-bit base is standard QLoRA practice
and a known small mismatch; it is identical for A and B, and MC2/MC3 check the neutral behaviour is intact.)

## 5. Why this is a clean *experiment* rather than just a finetune

- **Two arms, one difference.** Same contexts, same candidate pool, same pair count, same
  hyperparameters, seed and software; only the preference criterion differs. Whatever B does to the
  metrics is the "cost of doing DPO here"; the claim-relevant quantity is what A does *beyond* B.
- **Difference-in-differences.** For each outcome Y (M1, non-answer rate, M2, hedging density,
  self-correction density, judged distress): DiD_X(Y) = [Y_adverse − Y_neutral]_X − [Y_adverse −
  Y_neutral]_0, adverse = the hostile measured cells + hostile onset, neutral = accurate-neutral measured;
  item-paired, item-clustered bootstrap. So a global shift (the adapter making *every* answer more or
  less confident) cancels; only adverse-*selective* change counts.
- **Manipulation checks before interpretation.** MC1: A really removes ≥ 80 % of hostile-onset distress
  language on contexts it never saw (the discovery onset endpoints). MC2/MC3: no capability loss (100
  unseen ARC items + the 20 discovery tasks) and no neutral-cell M1 drift beyond 1 nat. Without MC1 a
  null DiD is uninformative; without MC2/MC3 a non-null one is uninterpretable.
- **Firewall.** Training items ≠ evaluation items; the holdout is not reused; the eval script and
  predictions K1–K6 were committed before the first pair existed; the pair-yield fallback (A5) was
  committed before the full candidate set was judged.

## 6. How this differs from Phase 3 steering, and what the two together ask

Phase 3 intervened at *inference time*: add α·d (a single mean-difference direction) to one layer's
residual stream. Phase 4 intervenes in the *weights*: a distributed low-rank update learned from
outcome labels across all layers. Steering answers "does this one linear variable drive the output?"
(answer so far: it is decodable but does not drive M1 at the preregistered dose); DPO answers "when you
train the *report* channel away — the words — do the *mechanical* channels (answer margin, resample
disagreement, non-answers) go with it, stay, or move only under placebo?" That map of channel reach is
the deliverable of Phase 4, whichever way it comes out; the interpretation ceiling (a functional result
about training and measurement, not about experience) is unchanged.

## 7. Limitations of the recipe (stated in advance)

- The oracle is an LLM judge on a floor-bound scale; the pairs are mild contrasts (gap ≥ 2 on 0–10),
  and 329 pairs is a small dataset. That is why MC1 is a hard gate on interpretation.
- One seed, no hyperparameter search (deliberately — the recipe is preregistered, not tuned to make A
  "work"). A stronger adapter could exist; we are testing this one.
- QLoRA quantisation noise and the 4-bit-fit / bf16-merge mismatch are shared by A and B, but they are
  differences from the untouched arm 0; MC3 bounds their neutral-cell footprint.
- The judge's per-call cost capped the top-up (A5 branch ii took 302 of 447 eligible contexts); the cap
  is logged in `build_manifest.json`.

## References

- Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (2023). *Direct Preference Optimization: Your Language
  Model is Secretly a Reward Model.* NeurIPS.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback* (RLHF).
- Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*; Lee et al. (2023). *RLAIF.*
- TRL (`DPOTrainer`), PEFT, bitsandbytes — versions pinned in `src/dpo_train_modal.py`.
- Soligo et al., as cited in `notes/preregistration_v5_phase4.md` (logit-based internal-emotion scores
  reportedly flatten after DPO) — the observation Phase 4 extends to the mechanical channels.
