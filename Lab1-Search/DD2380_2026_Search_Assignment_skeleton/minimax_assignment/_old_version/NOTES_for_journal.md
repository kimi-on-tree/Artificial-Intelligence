# Lab1 Search — measured evidence for the Assignment Learning Journal

All numbers produced on this machine with `minimax_assignment/bench.py` and
`minimax_assignment/ablation.py`. Re-run them before quoting in the journal.

## Files

| file | role |
|---|---|
| `player.py` | **submitted to Kattis.** Skeleton provided `PlayerControllerHuman`, `player_loop`, and the `search_best_next_move` signature; everything else is my implementation. |
| `player_skeleton_backup.py` | untouched original skeleton (for the "provided vs implemented" statement in J1) |
| `player_v1_static_ordering.py` | earlier version, ordering by static evaluation — kept as the *alternative* compared in J4 |
| `bench.py` | headless self-play harness vs a random opponent; reports depth + time per move |
| `ablation.py` | switches each extension off, same 55 ms budget, same 49 positions |

`bench.py` / `ablation.py` are test tooling and are **not** part of the Kattis submission.

## Ablation — 49 real game positions, 55 ms budget each

| variant | avg depth reached | avg nodes expanded / move |
|---|---|---|
| minimax only (no extensions) | 5.18 | 3817 |
| + alpha-beta | 7.20 | 2939 |
| + alpha-beta + static-evaluation move ordering | 7.35 | 2695 |
| + alpha-beta + transposition table | 12.18 | 5041 |
| **submitted: alpha-beta + TT + TT-best-move ordering** | **12.55** | 5089 |

Readings:
* alpha-beta alone buys ~2 ply.
* Sorting children by their static evaluation cuts nodes by ~8 % but the
  evaluation calls cost about as much as the cut-offs save: depth is unchanged.
  → rejected as the ordering scheme.
* The transposition table is the single largest win (+5 ply). In this game the
  fish trajectories are fixed by the observation sequence, so a given ply is
  reached by many different move orders — transpositions are abundant.
* Reusing the best move stored in the TT is free ordering and adds ~0.4 ply
  on top. Direct head-to-head over the same 49 positions:
  static ordering 12.08 avg depth (min 8) vs TT-move ordering 12.88 (min 9).

## Timing — self-play vs random opponent (`bench.py 200`)

| scenario | moves | green | red | depth min/avg/max | time avg | time max |
|---|---|---|---|---|---|---|
| test_0 | 46 | 34 | 2 | 8 / 12.7 / 16 | 47.7 ms | 55.2 ms |
| test_1 | 158 | 3 | 0 | 8 / 12.3 / 15 | 55.0 ms | 55.5 ms |
| test_2 | 200 | 9 | 0 | 9 / 11.4 / 14 | 55.1 ms | 55.5 ms |
| test_3 | 4 | 13 | 10 | 14 / 15.5 / 16 | 33.4 ms | 55.0 ms |

Limit is 75 ms; worst observed 55.5 ms, i.e. 0.5 ms of overshoot past the
55 ms internal budget.

## The garbage-collection finding (good J3 material)

First working version exceeded the 75 ms limit (worst move 758 ms, then 88 ms
after the harness was fixed to rebuild the root every move like the real
controller does). Cause: the search allocates tens of thousands of `State`
objects per move and CPython's *cyclic* collector fired inside the timed
window (`Node.parent` / `Node.children` form reference cycles, so refcounting
alone cannot free them).

Controlled measurement, 60 moves, identical positions:

| | avg | p90 | max |
|---|---|---|---|
| GC on during search | 56.2 ms | 58.3 ms | **74.7 ms** |
| GC paused during search | 54.9 ms | 55.1 ms | **55.6 ms** |

Fix: `gc.disable()` around the search in `search_best_next_move`, `gc.enable()`
in the `finally`, and `gc.collect()` in `player_loop` *after* `self.sender(...)`,
i.e. outside the timed window.
Trade-off / limitation: peak memory per move is higher, and the collection cost
is only moved, not removed — it is paid while blocked on `self.receiver()`.

## Design decisions worth writing up

1. **Repeated-state key** (`_state_key`). Two nodes share a TT entry only if
   `(depth, player to move, both hook positions, both hooked fish, score
   difference, the exact set and positions of the remaining fish)` match.
   `depth` is in the key because it is the index into the observation sequence
   — the same board at a different ply has completely different fish motion
   ahead of it. The score difference is in the key because the evaluation is
   relative to it. Dropping either would make the table unsound.
2. **Cut-off vs terminal.** `_utility` returns the exact score difference and is
   used only for real terminal states (no fish left, or observation sequence
   exhausted). `_evaluate` is used at the depth cut-off.
3. **Evaluation function.** `score difference` + `0.9 × value of each hooked
   fish` + `0.05 × proximity advantage`, where proximity is
   `max over positive fish of value × exp(-0.25 × d)` and `d` is Manhattan
   distance with a wrap-around x axis. The weights are deliberately ordered so
   that a real point always dominates a hooked fish, which always dominates
   proximity — proximity only breaks ties and gives the search a gradient to
   follow when no capture is inside the horizon. Negative-value fish are
   excluded from the proximity term so the agent is never attracted to them.
4. **Iterative deepening + anytime behaviour.** A depth is only committed once
   its whole root loop finished; a `SearchTimeout` exception unwinds the
   recursion and the move from the last complete iteration is played. This is
   what makes a hard time limit safe.
