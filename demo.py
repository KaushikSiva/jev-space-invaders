"""Render demo footage: one seeded game per decider with a live overlay (move, decision
latency, confidence, score), then optionally stack two clips side by side.

  python demo.py --decider adapter --seed 4 --out clips/baseline.mp4
  python demo.py --decider jev     --seed 4 --out clips/jev.mp4
  python demo.py --stack clips/jev.mp4 clips/baseline.mp4 --out clips/side_by_side.mp4

Real-time mode (--realtime, default on): the game advances at Atari speed (60 fps / frameskip 4
= 15 steps per second) and a decider that is still thinking keeps the previous action, so slow
decisions visibly cost control, exactly as they would in a live game.
"""
import argparse, os, subprocess, tempfile, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym, ale_py
from harness import ACTIONS, Decider, encode_state

gym.register_envs(ale_py)
LABEL = {"jev": "JEV (TypeSafe)", "adapter": "GPT-4.1-mini (System One adapter)", "openai": "GPT-4.1-mini", "random": "random policy"}


def font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(f): return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render(decider_kind, seed, out, max_steps, realtime):
    d = Decider(decider_kind, {"jev": "jev-latest", "adapter": "gpt-4.1-mini", "openai": "gpt-4.1-mini", "random": "random-policy"}[decider_kind])
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="ram", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    tmp = tempfile.mkdtemp(); f_big, f_small = font(22), font(17)
    score, steps, prev, action, pending_until, last_ms, last_conf = 0, 0, None, 0, 0.0, 0.0, None
    frame_i, done = 0, False
    step_s = 4 / 60  # one env step of game time
    while not done and steps < max_steps:
        game_t = steps * step_s
        if game_t >= pending_until:
            n_before = len(d.conf)
            action = d.decide(encode_state(obs, prev), action)
            last_ms = d.lat[-1] if d.lat else 0.0
            last_conf = d.conf[-1] if len(d.conf) > n_before else None
            # In real time the chosen move only lands after the decision latency has elapsed.
            pending_until = game_t + (last_ms / 1000 if realtime else 0)
        prev = obs
        obs, r, term, trunc, info = env.step(action)
        score += r; steps += 1; done = term or trunc
        img = Image.fromarray(env.render()).resize((480, 630), Image.NEAREST)
        canvas = Image.new("RGB", (480, 760), (12, 12, 20)); canvas.paste(img, (0, 130))
        g = ImageDraw.Draw(canvas)
        g.text((14, 10), LABEL[decider_kind], fill=(255, 220, 90), font=f_big)
        g.text((14, 44), f"move: {ACTIONS[action]}", fill=(230, 230, 230), font=f_small)
        g.text((14, 68), f"decision: {last_ms:,.0f} ms" + (f"   confidence: {last_conf:.2f}" if last_conf is not None else ""), fill=(120, 220, 255), font=f_small)
        g.text((14, 92), f"score: {score:g}   lives: {info.get('lives')}   seed {seed}", fill=(230, 230, 230), font=f_small)
        canvas.save(f"{tmp}/{frame_i:06d}.png"); frame_i += 1
    env.close()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", f"{tmp}/%06d.png", "-pix_fmt", "yuv420p", "-c:v", "libx264", out], check=True)
    med = sorted(d.lat)[len(d.lat) // 2] if d.lat else 0
    print(f"{out}: {LABEL[decider_kind]} seed {seed} score {score:g} steps {steps} median decision {med:.0f} ms")


def stack(a, b, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-i", b, "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2:shortest=0[v]", "-map", "[v]", "-pix_fmt", "yuv420p", "-c:v", "libx264", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", choices=list(LABEL))
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--stack", nargs=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stack: stack(*a.stack, a.out)
    else: render(a.decider, a.seed, a.out, a.max_steps, not a.no_realtime)
