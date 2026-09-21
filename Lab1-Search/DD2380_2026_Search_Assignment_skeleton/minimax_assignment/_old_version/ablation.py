#!/usr/bin/env python3
"""
Ablation study (NOT part of the Kattis submission) -> evidence for the journal.

Every variant gets the same time budget on the same set of real game positions.
We report the average depth reached by iterative deepening and the average
number of nodes expanded per move.

Usage:  python ablation.py
"""
import gc
import math
import random
import time

import bench
from fishing_game_core.game_tree import Node
from fishing_game_core.shared import ACTION_TO_STR
from player import PlayerControllerMinimax, SearchTimeout, EXACT, LOWER, UPPER

STR_TO_ACTION = {v: k for k, v in ACTION_TO_STR.items()}


class Ablated(PlayerControllerMinimax):
    """The submitted search, with individual extensions switchable off.

    ordering: "none" | "static" (sort children by evaluation)
              | "tt" (try the best move of the previous iteration first)
    """

    def __init__(self, pruning=True, ordering="tt", table=True):
        super(Ablated, self).__init__()
        self.pruning = pruning
        self.ordering = ordering
        self.table = table
        self.nodes = 0

    def _alphabeta(self, node, depth, alpha, beta):
        if time.time() - self.start_time > self.TIME_LIMIT:
            raise SearchTimeout()
        self.nodes += 1

        if not self.pruning:
            alpha, beta = -math.inf, math.inf
        alpha_orig, beta_orig = alpha, beta

        state = node.state
        key = self._state_key(node)
        best_move = None
        if self.table:
            entry = self.tt.get(key)
            if entry is not None:
                stored_depth, value, flag, best_move = entry
                if stored_depth >= depth:
                    if flag == EXACT:
                        return value
                    elif flag == LOWER:
                        alpha = max(alpha, value)
                    else:
                        beta = min(beta, value)
                    if alpha >= beta:
                        return value

        if not state.get_fish_positions():
            return self._utility(state)
        if depth <= 0:
            return self._evaluate(state)
        children = node.compute_and_get_children()
        if not children:
            return self._utility(state)

        maximizing = (state.get_player() == 0)
        if len(children) > 1:
            if self.ordering == "tt" and best_move is not None:
                for i, child in enumerate(children):
                    if child.move == best_move:
                        if i:
                            children = ([children[i]] + children[:i]
                                        + children[i + 1:])
                        break
            elif self.ordering == "static" and depth > 1:
                children = sorted(children, key=lambda c: self._evaluate(c.state),
                                  reverse=maximizing)

        best_move = children[0].move
        if maximizing:
            value = -math.inf
            for child in children:
                cv = self._alphabeta(child, depth - 1, alpha, beta)
                if cv > value:
                    value, best_move = cv, child.move
                alpha = max(alpha, value)
                if self.pruning and alpha >= beta:
                    break
        else:
            value = math.inf
            for child in children:
                cv = self._alphabeta(child, depth - 1, alpha, beta)
                if cv < value:
                    value, best_move = cv, child.move
                beta = min(beta, value)
                if self.pruning and beta <= alpha:
                    break

        if self.table:
            if value <= alpha_orig:
                flag = UPPER
            elif value >= beta_orig:
                flag = LOWER
            else:
                flag = EXACT
            self.tt[key] = (depth, value, flag, best_move)
        return value


def collect_positions(path, n, seed=0):
    """Replay a game with the full agent to obtain n realistic root messages."""
    random.seed(seed)
    driver = PlayerControllerMinimax()
    fp, fs, hk, seq = bench.load(path)
    caught, sc, t = {0: None, 1: None}, {0: 0, 1: 0}, 0
    out = []
    while len(out) < n:
        if not fp or t + 2 >= len(next(iter(seq.values()))):
            break
        msg = bench.make_message(fp, fs, hk, seq, t, caught, sc)
        out.append(msg)
        gc.collect()
        node = Node(message=msg, player=0)
        mv = driver.search_best_next_move(node)
        ag = bench.pick_child(node, STR_TO_ACTION[mv])
        ro = ag.compute_and_get_children()
        if not ro:
            break
        fp, hk, caught, sc = bench.snapshot(random.choice(ro).state)
        t += 2
    return out


VARIANTS = [
    ("minimax only (no extensions)", dict(pruning=False, ordering="none",   table=False)),
    ("+ alpha-beta",                 dict(pruning=True,  ordering="none",   table=False)),
    ("+ alpha-beta + static order",  dict(pruning=True,  ordering="static", table=False)),
    ("+ alpha-beta + TT",            dict(pruning=True,  ordering="none",   table=True)),
    ("SUBMITTED: ab + TT + TT-order", dict(pruning=True, ordering="tt",     table=True)),
]

if __name__ == "__main__":
    positions = []
    for f in ("observations/test_0.json", "observations/test_1.json",
              "observations/test_2.json", "observations/test_3.json"):
        positions += collect_positions(f, 15)
    print("%d test positions, %.0f ms budget each\n"
          % (len(positions), 1000 * Ablated.TIME_LIMIT))
    print("%-32s %-12s %-14s" % ("variant", "avg depth", "avg nodes/move"))
    print("-" * 60)
    for name, cfg in VARIANTS:
        depths, nodes = [], []
        for msg in positions:
            agent = Ablated(**cfg)
            gc.collect()
            agent.search_best_next_move(Node(message=msg, player=0))
            depths.append(agent.last_depth)
            nodes.append(agent.nodes)
        print("%-32s %-12.2f %-14.0f" % (name, sum(depths) / len(depths),
                                         sum(nodes) / len(nodes)))
