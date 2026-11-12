#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun program/v1).

Model: time-averaged rank kernel with momentum.
The chart is a noisy rank process: playing-count growth has ~zero
correlation with rank moves, but rank momentum is real (corr ~0.24) and
rank diffusion grows with the square root of the horizon. For each hour
of the scored window we place a Gaussian kernel over the last chart's
ranks, centred on each game's momentum-adjusted effective rank and
widened by diffusion (sigma * sqrt(hour)); the forecast is the average
of the hourly kernels, which is the optimal single forecast for a mean
log score. 30% of each place stays unnamed, covering new entrants and
the long tail the kernel cannot name.

Band sigmas (rank-change std per hour) are estimated from the input
history at run time, with fallbacks from the practice archive.
Backtested on the practice archive: log score about -40.8 on two
72-hour-equivalent windows, beating the persistence baseline by +11 to
+16 points. Deterministic: no randomness, no wall clock, no network,
stdlib only.
"""
import json
import math
import os
from datetime import datetime

W0 = 1.5        # kernel width at hour 0 (initial rank uncertainty)
SIG_SCALE = 1.0  # multiplier on estimated diffusion sigma
SIGMA_FLOOR = 0.3  # minimum sigma: never trust an over-narrow kernel
MASS = 0.7       # named share per place; 1-MASS stays unnamed
MOM = 0.5        # momentum strength: effective rank shift per hour
VEL_CAP = 1.5    # max |rank| change per hour from momentum
MIN_P = 0.002    # drop named rows below this probability
MAX_ROWS_PER_PLACE = 100
FALLBACK_SIGMA = {(1, 10): 0.88, (11, 30): 2.08, (31, 10 ** 9): 3.48}


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: parse_ts(s["at"]))
    return task, snaps


def estimate_sigmas(snaps):
    """Rank-change std per hour, by rank band, from consecutive reads."""
    band_var = {(1, 10): [], (11, 30): [], (31, 10 ** 9): []}
    for i in range(1, len(snaps)):
        dt = (parse_ts(snaps[i]["at"]) - parse_ts(snaps[i - 1]["at"])) / 3600.0
        if dt <= 0:
            continue
        prev = {r["universe_id"]: r["rank"]
                for r in (snaps[i - 1].get("up_and_coming") or [])}
        for r in (snaps[i].get("up_and_coming") or []):
            u = r["universe_id"]
            if u in prev:
                d = r["rank"] - prev[u]
                for (lo, hi) in band_var:
                    if lo <= prev[u] <= hi:
                        band_var[(lo, hi)].append((d * d) / dt)
                        break
    sigmas = {}
    for band, vals in band_var.items():
        est = math.sqrt(sum(vals) / len(vals)) if len(vals) >= 20 else FALLBACK_SIGMA[band]
        sigmas[band] = max(SIGMA_FLOOR, est)
    return sigmas


def estimate_velocities(snaps, last_t):
    """Per-game rank-improvement velocity (ranks/hour), EWMA over the
    last few intervals with a 12h half-life weight, capped."""
    hist = {}
    for s in snaps:
        t = parse_ts(s["at"])
        for r in (s.get("up_and_coming") or []):
            hist.setdefault(r["universe_id"], []).append((t, r["rank"]))
    vel = {}
    for uid, series in hist.items():
        series.sort()
        v, wsum = 0.0, 0.0
        for j in range(len(series) - 1, max(0, len(series) - 4), -1):
            (t1, r1), (t0, r0) = series[j], series[j - 1]
            dt = (t1 - t0) / 3600.0
            if dt <= 0:
                continue
            w = 0.5 ** ((last_t - t1) / 3600.0 / 12.0)
            v += w * (r0 - r1) / dt
            wsum += w
        vel[uid] = max(-VEL_CAP, min(VEL_CAP, (v / wsum if wsum else 0.0)))
    return vel


def sigma_for(rank, sigmas):
    if rank <= 10:
        return sigmas[(1, 10)] * SIG_SCALE
    if rank <= 30:
        return sigmas[(11, 30)] * SIG_SCALE
    return sigmas[(31, 10 ** 9)] * SIG_SCALE


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, snaps = load_input(input_dir)
    places = int(task.get("places", 10))
    hours = int(task.get("window", {}).get("slots", 72))
    last_t = parse_ts(snaps[-1]["at"])
    sigmas = estimate_sigmas(snaps)
    vel = estimate_velocities(snaps, last_t)
    chart = sorted(snaps[-1].get("up_and_coming") or [], key=lambda r: r["rank"])

    # Average the hourly kernels over the window.
    acc = {}
    for h in range(1, hours + 1):
        for k in range(1, places + 1):
            w = W0 + sigma_for(k, sigmas) * math.sqrt(h)
            tot = 0.0
            ws = []
            for r in chart:
                reff = r["rank"] - MOM * vel.get(r["universe_id"], 0.0) * h
                reff = max(1.0, reff)
                ww = math.exp(-((reff - k) ** 2) / (2.0 * w * w))
                ws.append((r["universe_id"], ww))
                tot += ww
            if tot <= 0:
                continue
            for uid, ww in ws:
                acc[(uid, k)] = acc.get((uid, k), 0.0) + ww / tot

    probabilities = []
    rows_by_place = {}
    for (uid, k), s in acc.items():
        p = MASS * s / hours
        if p >= MIN_P:
            rows_by_place.setdefault(k, []).append((uid, p))
    for k in range(1, places + 1):
        rows = sorted(rows_by_place.get(k, []), key=lambda x: (-x[1], x[0]))
        rows = rows[:MAX_ROWS_PER_PLACE]
        for uid, p in rows:
            probabilities.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: order by momentum-adjusted mid-window rank (tie-break only).
    mid = hours / 2.0
    ranked = sorted(chart, key=lambda r: (
        max(1.0, r["rank"] - MOM * vel.get(r["universe_id"], 0.0) * mid),
        r["rank"]))
    top10 = [r["universe_id"] for r in ranked[:places]]

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
