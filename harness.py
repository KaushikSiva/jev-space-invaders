"""Space Invaders harness: JEV (TypeSafe System One) picks every move; an OpenAI LLM
runs the identical loop as the baseline. Every game is appended to results.json and
committed + pushed immediately (see push_results).

Usage:
  python harness.py --decider jev    --seeds 1 2 3 4 5
  python harness.py --decider openai --seeds 1 2 3 4 5
  python harness.py --decider random --seeds 1        # harness smoke test, non-JEV
"""
import argparse, json, os, random, statistics, subprocess, time, urllib.error, urllib.request
from importlib.metadata import version

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
ACTIONS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# RAM addresses from the AtariARI annotations for SpaceInvaders.
RAM = {"player_x": 28, "enemies_x": 26, "enemies_y": 24, "missile_y": 9,
       "invaders_left": 17, "lives": 73, "score_hi": 104, "score_lo": 102}


def encode_state(ram, prev_ram):
    """Compact JSON state from the 128-byte RAM (text only, as JEV requires)."""
    s = {k: int(ram[v]) for k, v in RAM.items() if k not in ("score_hi", "score_lo")}
    s["enemies_dx"] = int(ram[RAM["enemies_x"]]) - int(prev_ram[RAM["enemies_x"]]) if prev_ram is not None else 0
    s["missile_dy"] = int(ram[RAM["missile_y"]]) - int(prev_ram[RAM["missile_y"]]) if prev_ram is not None else 0
    s["ship_minus_enemies_x"] = s["player_x"] - s["enemies_x"]
    return s


INSTRUCTIONS = ("You control the laser cannon in Atari Space Invaders. Pick the next action. "
                "Stay under the invader formation to hit it, keep firing, and sidestep when an "
                "enemy missile (missile_y rising toward the ship) is close. Fields are raw RAM values.")
CRITERIA = {"NOOP": "Hold position without firing", "FIRE": "Shoot straight up",
            "RIGHT": "Move right", "LEFT": "Move left",
            "RIGHTFIRE": "Move right while shooting", "LEFTFIRE": "Move left while shooting"}


def post(url, body, headers, timeout=20):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Decider:
    def __init__(self, kind, model):
        self.kind, self.model = kind, model
        self.served = None
        self.reset()

    def reset(self):
        self.lat, self.conf, self.errors, self.calls = [], [], {}, 0
        self.retries = self.fallbacks = self.low_conf = self.in_tok = self.out_tok = 0

    def decide(self, state, last):
        if self.kind == "random":
            t = time.perf_counter(); a = random.randrange(6); self.lat.append((time.perf_counter() - t) * 1000); return a
        for attempt in range(3):
            t = time.perf_counter()
            try:
                self.calls += 1
                a = self._jev(state) if self.kind == "jev" else self._adapter(state) if self.kind == "adapter" else self._openai(state)
                self.lat.append((time.perf_counter() - t) * 1000)
                return a
            except Exception as e:
                if not isinstance(e, urllib.error.HTTPError):
                    self.lat.append((time.perf_counter() - t) * 1000)
                    code = str(getattr(e, "status_code", None) or getattr(e, "status", None) or type(e).__name__)
                    self.errors[code] = self.errors.get(code, 0) + 1
                    break

                self.lat.append((time.perf_counter() - t) * 1000)
                self.errors[str(e.code)] = self.errors.get(str(e.code), 0) + 1
                if e.code in (429, 529, 500, 502, 503) and attempt < 2:
                    self.retries += 1; time.sleep(0.3 * (attempt + 1)); continue
                break
        self.fallbacks += 1
        return last

    def _jev(self, state):
        r = post(JEV_URL, {"state": state, "model": self.model, "questions": {
            "action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}},
            {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})
        self.served = r.get("model", self.served)
        ans = r["answers"]["action"]
        u = r.get("usage", {}); self.in_tok += u.get("input_tokens", 0); self.out_tok += u.get("output_tokens", 0)
        self.conf.append(ans["confidence"])
        if ans["confidence"] < 0.6: self.low_conf += 1
        return ACTIONS.index(ans["choice"])

    def _adapter(self, state):
        """Baseline through TypeSafe's System One adapter: identical typed choice question, answered by OpenAI."""
        if not hasattr(self, "_client"):
            from system_one_adapter import SystemOneAdapterClient, Choice
            self._client = SystemOneAdapterClient(structured_outputs=True, llm_answer_mode="probabilities", normalize_probabilities=True)
            self._q = {"action": Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)}
        r = self._client.system_one(state=state, questions=self._q, provider="openai", model=self.model)
        self.served = getattr(r, "model", None) or self.served
        u = r.usage; self.in_tok += u.input_tokens_total or 0; self.out_tok += u.output_tokens_total or 0
        self.retries += u.n_retries or 0
        ans = r.answers["action"]
        self.conf.append(ans.confidence)
        if ans.confidence < 0.6: self.low_conf += 1
        return ACTIONS.index(ans.choice)

    def _openai(self, state):
        prompt = (f"{INSTRUCTIONS}\nOptions: " + json.dumps(CRITERIA) + f"\nState: {json.dumps(state)}\n"
                  "Answer with exactly one option name and nothing else.")
        r = post(OPENAI_URL, {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 8},
                 {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        self.served = r.get("model", self.served)
        u = r.get("usage", {}); self.in_tok += u.get("prompt_tokens", 0); self.out_tok += u.get("completion_tokens", 0)
        text = r["choices"][0]["message"]["content"].strip().upper().replace(" ", "")
        for name in sorted(ACTIONS, key=len, reverse=True):
            if name in text: return ACTIONS.index(name)
        raise ValueError(f"unparseable reply {text!r}")


def pct(xs, p):
    if not xs: return 0
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 1)


def play(decider, seed, interval, max_steps):
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="ram")
    obs, info = env.reset(seed=seed)
    decider.reset(); start = time.time()
    score, steps, prev, action, lives0 = 0.0, 0, None, 0, info.get("lives", 3)
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        if steps % interval == 0:
            action = decider.decide(encode_state(obs, prev), action)
        prev = obs
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward; steps += 1
    env.close()
    run = {"seed": seed, "score": score, "steps": steps, "frames": info.get("episode_frame_number"),
           "lives_lost": lives0 - info.get("lives", 0), "terminated": bool(terminated), "truncated": bool(truncated or steps >= max_steps),
           "model_calls": decider.calls, "input_tokens": decider.in_tok, "output_tokens": decider.out_tok,
           "latency_ms_p50": pct(decider.lat, 50), "latency_ms_p95": pct(decider.lat, 95), "latency_ms_total": round(sum(decider.lat), 1),
           "errors_by_status": decider.errors, "retries": decider.retries, "fallback_actions": decider.fallbacks,
           "wall_clock_s": round(time.time() - start, 1), "served_model": decider.served or ("random-policy" if decider.kind == "random" else None)}
    if decider.conf:
        run["mean_confidence"] = round(statistics.mean(decider.conf), 3)
        run["low_conf_rate"] = round(decider.low_conf / len(decider.conf), 3)
    return run


def load():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f: return json.load(f)
    return {"schema_version": 2, "models": [], "config": {}, "runs": [], "baseline": {"model": None, "runs": []}}


def push_results(msg):
    """Commit and push results.json after every game, as the arena rules require."""
    root = os.path.dirname(RESULTS)
    subprocess.run(["git", "add", "results.json"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=root, check=True)
    subprocess.run(["git", "push"], cwd=root, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", choices=["jev", "adapter", "openai", "random"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--interval", type=int, default=2, help="env steps per decision (action held between)")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    model = a.model or {"jev": "jev-latest", "adapter": "gpt-4.1-mini", "openai": "gpt-4.1-mini", "random": "random-policy"}[a.decider]
    d = Decider(a.decider, model)
    for seed in a.seeds:
        run = play(d, seed, a.interval, a.max_steps)
        run["notes"] = a.note
        res = load()
        res["config"] = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "repeat_action_probability": 0.25, "full_action_space": False,
                         "max_num_frames_per_episode": 108000, "obs_type": "ram", "wrappers": [], "decision_interval_steps": a.interval,
                         "max_steps": a.max_steps, "state_encoding": "RAM bytes (AtariARI addresses) decoded to ship x, invader block x/y, missile y, lives, invaders left, deltas; JSON",
                         "ale_py_version": version("ale-py"), "gymnasium_version": version("gymnasium")}
        role = "baseline" if a.decider in ("openai", "adapter") else "decider"
        entry = {"role": role, "provider": {"jev": "typesafe", "adapter": "openai", "openai": "openai", "random": "none"}[a.decider], "requested_model": model,
                 "served_model": run["served_model"],
                 "sdk_package": "system-one-adapter" if a.decider == "adapter" else "http (urllib)",
                 "sdk_version": version("system-one-adapter") if a.decider == "adapter" else None}
        res["models"] = [m for m in res["models"] if m["role"] != role or m["provider"] != entry["provider"]] + [entry]
        if a.decider in ("openai", "adapter"):
            run["baseline_path"] = "system-one-adapter" if a.decider == "adapter" else "raw chat completions"
            res["baseline"]["model"] = model; res["baseline"]["runs"].append(run)
        else:
            res["runs"].append(run)
        with open(RESULTS, "w") as f: json.dump(res, f, indent=2)
        n = len(res["runs"]) + len(res["baseline"]["runs"])
        print(f"[{a.decider}] seed {seed}: score {run['score']} steps {run['steps']} p50 {run['latency_ms_p50']}ms")
        if not a.no_push: push_results(f"results: run {n}, {a.decider} seed {seed}, score {run['score']:g}")


if __name__ == "__main__":
    main()
