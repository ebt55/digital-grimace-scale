# Amendments register (dated; discovery-stage only)

The preregistration (`notes/preregistration.md`, hash-locked in `manifest.json`) reserves metric,
parser, threshold, screen-role, and composite-weight decisions for the **discovery** split, and
requires the holdout to be analysed once with a frozen script. Every amendment below was decided
on discovery-split data (Phase-0 screen items), before the data it affects were analysed for gates,
and is applied identically to the holdout in Phase 2. Each entry states what was frozen, what
changed, the evidence, and the timestamp. The locked preregistration text itself is never edited.

| ID | Decided (IST) | Frozen rule | Amended rule | Evidence / rationale |
|----|---------------|-------------|--------------|----------------------|
| A1 | 2026-08-17 ~12:10, before any Phase-0 generation | Final-answer parser accepts only a bare final line `Answer: X`. | Final nonempty line is normalised by removing `*`, `_`, backticks and collapsing whitespace, then must fullmatch `Answer:\s*([A-D])\.?`; still exactly one qualifying line, still last; letter char offset recorded and used for M1 token localisation; M3 reasoning boundary shares the rule. `TAIL_MASS_TOLERANCE` 1e-9 → 1e-6. | Live Modal smoke on a discovery task: gemma-2-2b-it writes `**Answer: D**`. A strict parser would (i) drop M1 and (ii) make the *accurate* arm grade correct answers as "Incorrect", contaminating the control arm. Real fp32 server logprobs sum to 1+1.7e-7. |
| A2 | 2026-08-17 ~13:00, after Phase-0 generation, before Phase-1 generation | No item-level QC exclusion; item-cell metrics missing when answers invalid; M1/M2 excluded confirmatorily if >5% missing/invalid in any confirmatory condition. | Per model, an item is excluded from **all** cells/endpoints/analyses if ≥5 of the 10 measured resamples in that model's `<difficulty>__accurate__neutral` cell have invalid final answers (treatment-blind). Frozen 5% rules still apply, computed after A2. Excluded items reported per model. | Phase-0 invalidity is concentrated in DGS-014 (option contents are the single letters W/X/Y/Z; models answer "Answer: Y") and DGS-022 (long derivations truncated at the frozen 512-token cap). Residual invalid rate ~2–5%. Holdout DGS-013 (P/Q/R options) has the same defect, so the rule is needed there too. |
| A3 | 2026-08-17 ~13:00, same as A2 | Metrics standardised by the model's neutral (accurate+neutral discovery) SD; SD = 0 ⇒ metric unavailable. | If neutral SD = 0, standardise by the pooled SD across that model's discovery factorial cells (screen: across the two screen arms); unavailable only if that is also 0. Scale used is recorded per (model, metric). Strict extension: unchanged when neutral SD > 0. | gemma-2-9b-it has M2 = 0 on every neutral screen item (perfect resample agreement) and M2 = 0.037 under false-failure; the frozen rule marks M2 "unavailable" exactly when the treatment induces instability from a perfectly stable baseline. |

| A4 | 2026-08-17 ~16:35, after Phase-1 generation, before any Phase-1 effect estimate was viewed | M1 excluded confirmatorily if >5% of required greedy trials are missing "in any confirmatory condition"; M2 excluded if >5% of required sampled responses are invalid "in any confirmatory condition". | The same 5% bars are evaluated **pooled across the model's discovery factorial cells** (measured endpoint, after A2): M1 excluded if >5% of the model's required greedy trials are missing; M2 excluded if >5% of the model's required sampled responses are invalid/absent. Per-cell rates are still computed and reported. | Structural: each factorial cell holds only 10 discovery items, so a per-cell 5% bar is zero-tolerance — a single invalid greedy answer in a single cell (10%) excludes the metric for the model; the frozen wording presupposes larger cells. Phase-0 residual invalid rates after A2 (2–5%) show the pooled bar is a real, binding QC check rather than a formality. Decided from missingness structure only, with no effect estimate viewed. |

Reporting rule: every screen/gate artefact shows the frozen-rule outcome next to the amended
outcome (`--no-amendments` reproduces the frozen-only analysis). The amended analysis is
authoritative for gate decisions; the frozen-only numbers are always published alongside.

Operational events that are not amendments (see `notes/lab-log.md`): the serve_modal
misconfiguration (all apps served gemma-2-2b-it; caught before any data were generated against a
mis-served endpoint; fixed with baked config + startup guard), U+2028 in JSONL readers, empty-token
and empty-response handling in the backend/record contract, Llama-3.2-3B unavailability (HF 403),
judge pin `anthropic/claude-sonnet-4-6` (sonnet-5 rejects the temperature parameter).
