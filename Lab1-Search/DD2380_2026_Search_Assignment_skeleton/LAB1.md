# LAB1

## J0

Name1: Mo Kong

Email1: mkong@kth.se

Name2: None

Submission ID: 20485310

Score: 23

## J1

Central idea: My agent chooses the green boat's move with minimax and alpha-beta pruning, and also applied iterative deepening, move ordering and a transposition table.

Flow from input to output:

- Input: player_loop receive the game message -> Node(message) holding the hooks, fish, scores and the remaining observation sequence.
- Search: search_best_next_move runs iterative deepening on current root: depth 1, 2, 3,...until the time budget ends. Each iteration calls _search_root, and it calls the recursive _alphabeta on every root child. _alphabeta scores scores terminal states with _utility and depth-limit states with _evaluate. Otherwise it checks the transposition table( _state_key ), expands children with compute_and_get_children, and searches the stored best move first( _order_children ). The table is cleared every move but shared by its iterations.
- Output: The best move of the deepest finished iteration is returned

Everything mentioned above is implemented in PlayerControllerMinimax.

## J2

**Q1: The state space:** The current player's index; The player scores; The positions of the two hooks; The positions of the uncaught fishes; The score values associated with each fish index. **Initial state** of a scenario is decided by the observation files. The file offers the initial position of each player's hook, the initial position of fish and value, and each fish movement sequence. **The successor function** applies one of five actions, move the fish by the next observation, update catches and scores, and swich player.

**Q2:** The terminal states: 1. There's no fish left 2. The observation sequence is exhausted. For the green player, the utility of a terminal state is the final score difference(positive=win; negative=loss; zero=tie)

**Q3:** When at a terminal state it's the same with terminal utility, so estimates at the cut-off values can indicate deeper exact values; and a lead is a real progress since landed points cannot be lost; also it's O(1), so it saves time because it is computed at every leaf.

**Q4:** Near the end of the game when there's few fish or few observation steps left, so little can be changed; At a terminal state v is exact. Early on it can say nothing: at the start of test_0 all five root moves evaluated to 0, although an 11-point fish was reachable, because landing it takes about 15 plies while the search reached about 7.

**Q5:** Green leads 2:0, so v = +2. It is red's turn and the only fish left, worth 10, is on red's line at y = 18. Red moves up, lands it and ends the game at 2:10. v counts only landed points and ignores the hooked fish and red's next move.

**Q6:** η counts reachable winning and losing terminals but ignores who makes each choice. Example: A has one move, to s, where B chooses among four terminal states: three A-wins and one A-loss. η(A,s) = 3 − 1 = 2 > 0, but a rational B picks the loss, so A cannot force a win. 

## J3

### Decision 1: Searching deeper

**Goal:** Search deep enough to see catches, without exceeding 75 ms per move.

**Alternatives and Evidence:** My first version was plain minimax with a cut-off at depth 3. It passed all test cases but scored only 12. Adding alpha-beta pruning still gave 12, because pruning only skips branches that cannot change the result, and a fixed depth must be safe for the slowest position, so the search stayed far shorter than the roughly 15 plies needed to land a fish.

**Choice and reason:** From Section 2.5 I chose iterative deepening together with move ordering. Iterative deepening searches depth until time budget ends, so easy positions are searched deeper and there is always a finished answer. Each finished iteration also tells the next one which move was best, and searching that move first lets alpha-beta prune earlier. The score rose to 18. I then added a transposition table, cleared at every move. Each entry stores the depth searched, the value, whether it is exact or only a bound, and the best move. A repeated state is answered from the table when its entry was searched deep enough; otherwise its stored best move is searched first. The score rose to 22 (see J4).

**Code location**: iterative deepening in `search_best_next_move`, `_search_root` and `_order_root_children`, stopped by `SearchTimeout` at `TIME_BUDGET`; the transposition table in `_alphabeta`, `_state_key`, `_order_children`, `self._table` and the entry types `EXACT`, `LOWER`, `UPPER`.

**Limitation.** I did not try more complex techniques such as quiescence search or keeping the table between moves. The 55 ms budget was chosen as a safety margin, not tuned: a larger budget could search deeper but risks exceeding 75 ms on a slower judge machine, while a smaller one wastes time.

### Decision2: Evaluation function

**Goal:** make the agent act even when no point can be scored within the search horizon.

**Failure case:** After I got score 18 by applying iterative deepening, I found the green boat stayed still in the GUI in test_0. In test_0 the nearest valuable fish needs about 15 plies to be landed, deeper than the search, so the score difference was 0 for every move and ties went to "stay". And when a fish could be landed either now or a few moves later with the same final score, the search saw no difference, so it kept delaying. 

**Choice and reason:** `_evaluate` now adds two terms to the score difference: fish already hooked, at 0.9 of their value because they are almost certain to be landed, and proximity to the best positive fish, which gives the hook a direction. A small bonus for scoring earlier breaks the ties behind procrastination. I rejected "never stay on a tie", because staying is correct when only negative fish remain. 

**Evidence:** In the GUI, the boat started moving actively towards the fish, lowering its hook even when a fish was deep; Kattis rose from 18 to 19; I also tunned the params to improve performance. With the transposition table in place, I raised `PROXIMITY_WEIGHT` from 0.03 to 0.3 and `EARLY_BONUS` from 0.003 to 0.03: the boat followed fish deeper more actively in the GUI and Kattis rose from 22 to 23. Raising `EARLY_BONUS` further to 0.05 dropped the score back to 22, maybe because it then rushes for small nearby fish instead of larger ones further away. The result improved from 24:12 to 26:10(green wins) after I tunned the params.

**Code location**: `_evaluate`, `_proximity`, `_early_bonus` (applied in `_search_root` and `_alphabeta`) and the constants `HOOKED_WEIGHT`, `PROXIMITY_WEIGHT`, `EARLY_BONUS`.

**Limitation.** A bonus that is too small only breaks exact ties; one that is too large prefers small early catches over larger later ones. The weights were chosen by reasoning and a few Kattis submissions, not systematic tuning, and a time-limited search is not perfectly rep eatable, so a one-point difference may partly be noise. Also, both weights were changed together, so the gain cannot be attributed to either one alone.

