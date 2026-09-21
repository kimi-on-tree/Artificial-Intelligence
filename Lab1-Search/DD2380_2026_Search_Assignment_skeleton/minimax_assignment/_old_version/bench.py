#!/usr/bin/env python3
"""
Headless test harness (NOT part of the Kattis submission).

Simulates the game the same way the real controller does: a FRESH root Node is
built from a message on every move, exactly like player_loop() does.
Green = our agent, Red = random opponent.

Usage:  python bench.py [rounds]
"""
import gc
import json
import os
import random
import sys
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.shared import ACTION_TO_STR
from player import PlayerControllerMinimax

STR_TO_ACTION = {v: k for k, v in ACTION_TO_STR.items()}


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    fishes = data["init_fishes"]
    return (
        {int(k): tuple(v["init_pos"]) for k, v in fishes.items()},   # positions
        {int(k): v["score"] for k, v in fishes.items()},             # scores
        {int(k): tuple(v) for k, v in data["init_players"].items()}, # hooks
        {int(k): v for k, v in data["sequence"].items()},            # sequences
    )


def make_message(fish_pos, fish_scores, hooks, sequences, t, caught, scores):
    return {
        "game_over": False,
        "observations": {f: sequences[f][t:] for f in fish_pos},
        "fishes_positions": dict(fish_pos),
        "hooks_positions": hooks,
        "caught_fish": caught,
        "player_scores": scores,
        "fish_scores": fish_scores,
    }


def pick_child(node, move):
    for child in node.compute_and_get_children():
        if child.move == move:
            return child
    return node.compute_and_get_children()[0]


def snapshot(state):
    hooks = state.get_hook_positions()
    caught = state.get_caught()
    s0, s1 = state.get_player_scores()
    return dict(state.get_fish_positions()), {0: hooks[0], 1: hooks[1]}, \
           {0: caught[0], 1: caught[1]}, {0: s0, 1: s1}


def run(path, rounds, seed=0):
    random.seed(seed)
    controller = PlayerControllerMinimax()
    fish_pos, fish_scores, hooks, sequences = load(path)
    caught = {0: None, 1: None}
    scores = {0: 0, 1: 0}
    t = 0
    times, depths = [], []

    for _ in range(rounds):
        if not fish_pos or t + 2 >= len(next(iter(sequences.values()))):
            break
        msg = make_message(fish_pos, fish_scores, hooks, sequences, t, caught, scores)
        node = Node(message=msg, player=0)
        gc.collect()  # mimic player_loop(): collect outside the timed window

        t0 = time.time()
        move = controller.search_best_next_move(node)
        times.append(time.time() - t0)
        depths.append(controller.last_depth)

        after_green = pick_child(node, STR_TO_ACTION[move])
        red_options = after_green.compute_and_get_children()
        if not red_options:
            break
        after_red = random.choice(red_options)
        fish_pos, hooks, caught, scores = snapshot(after_red.state)
        t += 2

    print("%-14s moves=%3d  green=%3d red=%3d diff=%+4d | depth min/avg/max=%2d/%4.1f/%2d"
          " | time avg=%5.1fms max=%6.1fms %s"
          % (os.path.basename(path), len(times), scores[0], scores[1],
             scores[0] - scores[1],
             min(depths), sum(depths) / len(depths), max(depths),
             1000 * sum(times) / len(times), 1000 * max(times),
             "OK" if max(times) < 0.075 else "*** OVER 75ms ***"))


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    for name in sorted(os.listdir("observations")):
        if name.endswith(".json"):
            run(os.path.join("observations", name), rounds)
