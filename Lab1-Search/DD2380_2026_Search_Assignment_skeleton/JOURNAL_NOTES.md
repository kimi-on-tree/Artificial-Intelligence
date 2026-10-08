# Journal notes (points + evidence, NOT the journal text)

Write the journal yourself, in your own words, in LAB1.md. These are bullet points to draw from.
Rules: 700-1000 words (max 1200, excluding question prompts and J6). No source code, no screenshots.
Refer to files / classes / functions by name.

Suggested word budget: J1 ~130 | J2 ~6 x 60 = 360 | J3 ~2 x 120 = 240 | J4 ~220 | J5 ~130  -> ~1080

Structure used below (changed from the first version of these notes):
- J3 decision 1 = evaluation function, J3 decision 2 = transposition table
- J4 = transposition table (strongest evidence: depth 6.25 -> 12.13, Kattis 19 -> 22)
- the timing / garbage-collection story = J5 "hardest issue resolved"

Where the numbers come from: local headless runs on my own machine (Python 3.10), each move built from a
fresh root Node as in player_loop, red = random mover (the real opponent.py only runs on Python 3.6/3.7).
"93 positions" = positions sampled from the four observation files. Say this once; it is also a limitation.

Kattis history (keep these, they are evidence):
| version                                              | Kattis        |
| iterative deepening, gc.collect() after sending      | 24/25 cases, 1 run-time error |
| + garbage collection counted inside the budget       | 25/25 cases   |
| + new evaluation function (proximity weight 0.05)    | 25/25, score 19 |
| proximity weight 0.05 -> 0.3                         | 25/25, score 19 (no change) |
| + transposition table  (FINAL)                       | 25/25, score 22 |

---------------------------------------------------------------------------------------------------------
## J0 - Submission record

- Name, KTH email (already in LAB1.md). Fix the typo "namme2", write partner: none.
- Kattis submission ID: <FILL IN: the ID of the 22-point submission>
- Kattis score: 22 / 25
- Aiming for: A

---------------------------------------------------------------------------------------------------------
## J1 - Solution overview (~100-150 words)

Flow from input to output:
- player_loop (skeleton) receives the game message -> Node(message) builds the root (game_tree.py, provided)
- search_best_next_move: iterative deepening, depth 1, 2, 3 ... until TIME_BUDGET (55 ms) is spent;
  returns the best move of the deepest iteration that FINISHED; a new transposition table per move
- _search_root: one alpha-beta iteration at the root; previous iteration's best move first
- _alphabeta: deadline check -> _is_terminal -> _utility (terminal) / _evaluate (cut-off) ->
  transposition-table lookup (_state_key) -> children ordered by the stored best move (_order_children)
  -> MAX/MIN loops with alpha-beta cut-offs -> store (value, depth, EXACT/LOWER/UPPER, best move)
- _evaluate: score margin + hooked fish + proximity to good fish (_proximity); _early_bonus rewards
  scoring earlier
- SearchTimeout aborts an unfinished iteration; gc.disable() in player_loop, gc.collect() inside the budget

Provided vs mine:
- Provided: game_tree.py (Node, State, compute_and_get_children, compute_next_state), the
  receive/send loop in player_loop, the signature of search_best_next_move, PlayerControllerHuman
- Implemented: everything else in PlayerControllerMinimax, SearchTimeout, the gc lines in player_loop

---------------------------------------------------------------------------------------------------------
## J2 - Assignment questions (~40-80 words each)

Q1 (your current answer is fine). Possible additions:
- successor = compute_and_get_children -> compute_next_state: apply the mover's action, move every fish
  by observation[depth], update catches/scores, switch player
- a player with a fish on the line can only move "up" (branching factor 1); x wraps; hooks cannot pass
- fish movement does not depend on the actions -> deterministic, no chance nodes

Q2 (your current answer is fine). Possible additions:
- in code: _is_terminal (no fish left OR node.depth >= len(observations))
- boundary case: the skeleton's compute_and_get_children only detects the second condition; with no fish
  left it still returns 5 children -> "no children" cannot be the terminal test

Q3 - why score difference is a reasonable cut-off evaluation
- same unit as the terminal utility; at a terminal state it EQUALS gamma(A,s), so cut-off and terminal
  values are comparable in one tree
- O(1) from the state, and it is computed at every leaf (thousands of times per move)

Q4 - when is v most reliable
- near the end: few fish / few observation steps left -> little can still change
- early on it can be uninformative: at the start of test_0 all 5 moves evaluated to 0

Q5 - misleading example (use your own numbers), e.g.:
- green leads 2:0 (v = +2) but a 10-point fish is on red's line one step below the surface; red's next
  "up" lands it -> 2:10
- v only counts landed fish; it ignores fish on a line and fish about to be caught
- (link to your code: this is why _evaluate adds the hooked-fish term)

Q6 - why eta > 0 does not mean A can force a win
- eta counts reachable terminal states; it ignores WHO controls each choice
- tree: A has one move to s; at s B chooses among 4 terminals: 3 A-wins, 1 A-loss.
  eta(A,s) = 3 - 1 = +2 > 0, but a rational B picks the loss -> minimax value = loss

---------------------------------------------------------------------------------------------------------
## J3 - Decisions and evidence (two decisions; J4 builds on decision 2)

### Decision 1: evaluation function  [unexpected behaviour + revised approach]
Goal: make the agent act when no point can be scored within the search horizon.
Problem found in the GUI: with the plain margin the green boat never moved in test_0, 2, 3. Two causes:
- horizon effect (test_0): nearest good fish ~3 moves to hook + 5 to pull = ~15 plies before any
  point; search reached ~7 -> all 5 moves = 0 -> first child "stay", every move
- procrastination (test_2): 9-point fish next to the hook, out of red's reach. Depth-5 root values:
  without bonus all five moves = -1.0 (tie -> "stay"); with _early_bonus only "left" is best (-1.005)
Choice: _evaluate = margin + 0.9 x hooked fish + w x proximity; _early_bonus = 0.001 x points x depth_left
(applied as a shift of the alpha-beta window; 93 positions x depth 3,4: identical values to plain minimax)
Evidence - same games (3 seeds, <= 60 moves, random red), test_0 green:red / number of "stay" moves:
  margin only 24:40 (160/180 stay) | + hooked 71:4 (130/180) | + proximity 93:4 (3/152) | + early bonus 106:2
Revised: proximity weight 0.05 -> 0.3 (one step towards an 11-point fish was worth only ~0.005, the same
size as the other small terms). The real constraint is only w < 0.9 (hooking must beat hovering).
Kattis did NOT change (19 -> 19) -> suggested the bottleneck was search depth, not the evaluation.
Trade-off / limitation: weights set by reasoning about their order, not tuned; evaluation now costs
O(number of fish) per leaf; random opponent is much weaker than the Kattis one.
Code: _evaluate, _proximity, _early_bonus, HOOKED_WEIGHT, PROXIMITY_WEIGHT, EARLY_BONUS

### Decision 2: transposition table  (short here, details in J4)
- goal: stop re-searching repeated states; measured 93% of nodes at depth 5 are repeats
- alternatives: no table; table used only for move ordering; table used only for value reuse; both
- evidence: avg depth 6.25 -> 12.13, Kattis 19 -> 22
Code: _state_key, _alphabeta steps (c) lookup, (d) ordering, (f) store, _order_children, self._table

---------------------------------------------------------------------------------------------------------
## J4 - Advanced analysis: transposition table
(A needs: compare alternatives with concrete evidence, justify the choice under the 75 ms limit)

Why it should help
- moves commute (up-then-left = left-then-up; blocked moves = stay) and the fish follow the observation
  sequence whatever the hooks do, so different move orders reach the same state
- measured (40 positions, full expansion): repeats per level
  depth 3: 66% | depth 4: 83% | depth 5: 93% (99,408 nodes, only 6,659 distinct states)

How it works
- key (_state_key): node.depth (= index into the observation sequence, i.e. how the fish move next),
  player to move, both hooks, fish on each line, score margin, remaining fish + positions (frozenset)
- why these: two nodes may share an entry only if everything that affects the search below them is
  equal (the assignment's own requirement). Same board at another depth = different fish future.
  Margin is needed because evaluation and utility are relative to it
- left out: fish values (constant), the path (the future does not depend on it)
- consequence of node.depth in the key: the same state can never reappear below itself -> no cycles,
  unlike chess (no repetition problem)
- entry = (depth searched, value, EXACT / LOWER / UPPER, best move)
- bounds: a pruned node only proves value >= beta (LOWER) or <= alpha (UPPER). Reused only if it decides
  the node for the current window (EXACT; LOWER >= beta; UPPER <= alpha) and was searched deep enough
- table cleared every move (depth is relative to the move's root) but shared across the iterations of
  iterative deepening -> the previous iteration's best move orders EVERY node, not only the root
- SearchTimeout skips the store -> no half-finished entries

Evidence - alternatives compared (93 positions, 55 ms budget, completed depth)
| variant                           | avg depth | median | min |
| no table                          | 6.25      | 6      | 5   |
| table for move ordering only      | 7.23      | 7      | 6   |
| table for value reuse only        | 9.62      | 9      | 6   |
| both (submitted)                  | 12.13     | 12     | 7   |
- the two uses reinforce each other (+1.0 and +3.4 alone, +5.9 together): better ordering -> earlier
  cut-offs -> smaller subtrees -> more of them finish and become reusable
- fixed depth 5: states generated per position 1351 without table -> 233 with a warmed table
- correctness: depth 3, 4, 5 on 93 positions, root values with and without the table: 0 mismatches
- timing: 350 moves in full games, worst 55.8 ms (limit 75)
- Kattis 19 -> 22, while the evaluation change before it gave 19 -> 19
- against the random opponent the results did not change (test_0 34:2 both) -> a weak opponent does
  not reveal the value of deeper search; only Kattis did

Choice under the 75 ms constraint
- key building costs O(number of fish) per node, but doubling the depth shows it is repaid many times
- leaves (depth_left = 0) are not stored: evaluating is about as cheap as building a key (not measured -
  say so, it is an untested design choice)

Alternatives / limitations to mention
- store "value minus current margin" and drop the margin from the key -> more hits (states that differ
  only in score would share entries); not done, more complex, depends on the evaluation being additive
- keep the table between moves by keying on absolute time instead of node.depth; not done
- bound handling is the simple version: bounds are used only for cut-offs, not to narrow the window
- a wrong key silently gives wrong values -> that is why the equality test above matters

---------------------------------------------------------------------------------------------------------
## J5 - Individual learning and contribution (~100-150 words)  - write honestly

- Contribution (be accurate): worked alone with an AI assistant. What you did: chose the direction at
  each step (iterative deepening; rejected the prediction-based timer; asked to simplify the GC handling;
  chose the order of improvements), found in the GUI that the boat never moved, noticed Kattis showed
  19 despite 25/25 test cases, tried the proximity-weight change, reviewed and questioned the code
- Most important thing understood (pick ONE you can explain in the oral exam), e.g.:
  alpha-beta returns a BOUND for pruned nodes -> strict ">" at the root, and EXACT/LOWER/UPPER in the table
- Hardest issue resolved - the run-time error (24/25):
  first fix gc.collect() after sending; the next message arrives while it still runs (collection 13-17 ms
  avg, up to ~30 ms) -> search + collection up to 83 ms. Fix: gc.disable() once, gc.collect() at the start
  of search_best_next_move after the deadline is set -> worst 56 ms, 25/25. Cost: depth 9.07 -> 8.83.
  (Kattis never says which case failed, so this is a hypothesis supported by local measurements)
- Open limitation: 3 Kattis points still lost; evaluation weights untuned; tested only against a random
  opponent locally

---------------------------------------------------------------------------------------------------------
## J6 - Use of generative AI  (not counted in words; must be truthful)

- Tool: Claude (Anthropic), used during development and for preparing these notes
- Used for: explaining the algorithms; writing the implementation in player.py; designing and running the
  local experiments; diagnosing the Kattis run-time error; organising evidence for the journal
- Parts influenced: the PlayerControllerMinimax implementation; all experiment numbers
- How you checked / corrected / rejected (examples):
  - Kattis results at every step (24/25 RTE -> 25/25 -> 19 -> 19 -> 22) and GUI runs
  - equality tests: alpha-beta vs plain minimax, table vs no table - identical root values
  - rejected the prediction-based timer; asked to simplify the GC handling; changed the proximity weight
  - assistant claims that turned out wrong and were corrected by evidence: that compute_and_get_children
    detects "no fish left"; that gc.collect() after sending is outside the timed window; that "25/25"
    on Kattis meant 25 points
  - you found the "boat never moves" problem yourself in the GUI
- The journal text: written by you (only these notes were prepared with AI) - state this if true
