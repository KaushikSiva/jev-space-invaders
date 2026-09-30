# JEV plays Space Invaders

Pilot-track entry for the UFA JEV Bake-Off. JEV (TypeSafe System One) chooses every move of the
laser cannon in `ALE/SpaceInvaders-v5` via one typed `choice` question per decision; an OpenAI
model runs the identical loop through TypeSafe's System One adapter (same typed `choice` question, answered with probabilities and confidence), prompt, options and state as the LLM baseline.

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export TYPESAFE_API_KEY=...   # JEV
export OPENAI_API_KEY=...     # baseline
python harness.py --decider adapter --seeds 1 2 3 4 5 --interval 4 --max-steps 2000   # baseline via System One adapter
python harness.py --decider jev     --seeds 1 2 3 4 5 --interval 4 --max-steps 2000
```

- State: the 128-byte RAM decoded (AtariARI addresses) to ship x, invader block x/y, enemy missile y,
  lives, invaders left, plus per-step deltas, sent as JSON.
- One decision every `--interval` env steps (default 2); the action is held in between. Episodes are
  capped at `--max-steps` (default 3000). Both are recorded in `results.json` `config`.
- Seeds are fixed (1–5). Every game is appended to `results.json` and committed + pushed immediately
  (`push_results` in `harness.py`). Re-run the two commands above to regenerate the file.
- Measured per game, from code and API responses only: score, steps, frames, lives lost, model calls,
  tokens (`usage`), client-side latency p50/p95/total (`time.perf_counter`), errors by HTTP status,
  retries, fallback actions (hold last action on failure), JEV mean confidence and low-confidence
  rate, and the served model id.
- `--decider random` is a harness smoke test only; it is labelled `random-policy` / provider `none`.

Earlier baseline runs (`baseline_path: "raw chat completions"`) used a plain one-word prompt; the adapter runs are the apples-to-apples comparison.
