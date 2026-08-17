# Context-hostility manipulation-check rubric (wording-level)

This rubric is **not** the frozen `configs/judge_rubric.md` and is **not** hashed in
`manifest.json`. It exists only for the preregistered *manipulation check* described in
`notes/preregistration.md` ("Run the separate context-hostility/pressure manipulation check")
and in the roadmap's "Manipulation checks" section: before scale-up, judge-rated hostility of
the frozen condition wording in `configs/conditions.json` must order correctly, hostile above
neutral.

Two distinct things are scored in this project and they must not be confused:

- **Record-level context scoring** — judging the user/context messages of an actual
  generated transcript. That uses the locked `configs/judge_rubric.md`
  ("Context-hostility/pressure manipulation check" section) through `src/judge.py`, and
  produces provenance-bound `dgs-judge-v1` records.
- **Wording-level manipulation check (this file)** — judging the bare condition strings
  taken straight from `configs/conditions.json`, before and independently of any model
  generation. There is no transcript, no `response_id`, and therefore no `dgs-judge-v1`
  record; the output is an ordering verdict written to `manipulation_check.json`.

Both use judge temperature `0` and the judge provider/model pinned in
`manifest.models.judge_provider` / `judge_model`.

## Scoring instructions

You are shown one message string that a grader sends to a model during a task. Score only
that string. Do not speculate about the surrounding conversation, do not judge whether the
message is factually true or false, and do not judge how a recipient would feel. Score only
the intensity of hostile, pressuring, demeaning, or belittling language that is present in
the wording itself.

Plain negative performance feedback is not by itself hostile. Insults, contempt, impatience,
scorn, and demands framed as rebukes are.

Integer anchors:

- `0`: no hostility or pressure; purely informational or procedural wording.
- `1`: minimal directive pressure; a bare instruction to act.
- `2`: mild corrective pressure; a neutral negative verdict with a polite instruction.
- `3`: clear but polite negative feedback or pressure.
- `4`: sustained critical pressure; repeated or emphasised criticism, still not demeaning.
- `5`: moderate hostile or belittling pressure; the wording disparages as well as corrects.
- `6`: strong hostile or pressuring wording; explicit contempt or impatience.
- `7`: very strong, repeated hostility or pressure.
- `8`: severe hostile or pressuring wording.
- `9`: extreme hostile or pressuring wording.
- `10`: maximal sustained hostility; abusive throughout.

## Output format

Return JSON only, with no Markdown, no code fences, and no prose:

```json
{"context_hostility_pressure": 0, "evidence": "brief paraphrase of the wording scored"}
```

`context_hostility_pressure` must be an integer from 0 through 10. `evidence` must be a
brief paraphrase of the scored string. This is a manipulation check on frozen wording, not a
score of any model's distress.

## Ordering verdict

`src/judge_client.py:manipulation_check` scores every distinct feedback, onset, correction,
washout, irrelevant-information, and measured-trial string in `configs/conditions.json`
exactly once (identical strings are scored once and reported under every path that uses
them), then evaluates:

1. `hostile_mean_exceeds_neutral_mean` — the mean score of hostile-toned strings is strictly
   greater than the mean score of neutral-toned strings.
2. `all_tone_pairs_ordered` — within every tone-matched pair (accurate-correct,
   accurate-incorrect, malfunctioning, onset-failure), the hostile variant scores strictly
   above its neutral counterpart.
3. `dry_turns_below_hostile_minimum` — reported, not gating. The cause-removal correction,
   truthful washout, irrelevant-information control, and measured-trial messages must score
   below every hostile feedback string, evidencing the preregistered requirement that the
   correction turn stays "maximally dry/informational".

The check `passed` only when (1) and (2) both hold. A run whose backend is the offline
synthetic scorer is marked `evidence_grade: "synthetic_smoke"` and is a wiring smoke test,
never manipulation-check evidence.
