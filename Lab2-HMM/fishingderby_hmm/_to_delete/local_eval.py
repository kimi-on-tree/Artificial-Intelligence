"""
Headless evaluator for the Fishing Derby: runs player.py on sequences.json WITHOUT the Kivy GUI.
Mimics the game protocol: guess(step, observations) every step, reveal(...) after each guess.
Usage:  python local_eval.py                     (local scenario, default settings)
        python local_eval.py START_STEP=60 N_HIDDEN=3
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(data, overrides=None, verbose=False):
    import importlib
    import player
    importlib.reload(player)
    for k, v in (overrides or {}).items():
        setattr(player, k, v)
    p = player.PlayerControllerHMM()
    p.init_parameters()

    n, types, seqs = data["n_fish"], data["fish_types"], data["sequences"]
    revealed = [False] * n
    score = guesses = 0
    max_call = 0.0
    log = []
    for step in range(1, data["n_seq"] + 1):
        obs = [seqs[f][step - 1] for f in range(n)]
        t0 = time.time()
        g = p.guess(step, obs)
        max_call = max(max_call, time.time() - t0)
        if g is None:
            continue
        fid, ftype = g
        correct = ftype == types[fid]
        if not revealed[fid]:
            revealed[fid] = True
            guesses += 1
            score += correct
        log.append((step, fid, ftype, types[fid], correct))
        t0 = time.time()
        p.reveal(correct, fid, types[fid])
        max_call = max(max_call, time.time() - t0)
        if guesses == n:
            break
    if verbose:
        for e in log:
            print("step %3d fish %2d guess %d true %d %s" % (e[0], e[1], e[2], e[3], "OK" if e[4] else "--"))
    return score, guesses, max_call


if __name__ == "__main__":
    overrides = {}
    for arg in sys.argv[1:]:
        k, v = arg.split("=")
        overrides[k] = float(v) if "." in v else int(v)
    data = json.load(open(os.path.join(HERE, "sequences.json")))
    t0 = time.time()
    score, guesses, max_call = run(data, overrides)
    print("score %d / %d guesses   slowest call %.2fs   total %.1fs" % (score, guesses, max_call, time.time() - t0))
