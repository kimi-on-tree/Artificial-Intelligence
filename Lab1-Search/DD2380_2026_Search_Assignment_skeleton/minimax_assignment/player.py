#!/usr/bin/env python3
import gc
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


# ==========================================================================
#  iterative deepening alpha-beta under a time budget, with an
#  evaluation function that sees the potential of a position and a small
#  preference for scoring early.
# ==========================================================================

class SearchTimeout(Exception):
    pass


class PlayerControllerMinimax(PlayerController):

    # Time budget for one move, in seconds. 
    TIME_BUDGET = 55e-3

    # ---- Evaluation weights -------------------------------------------------
    # The weights are ordered so that each term can only break ties left by
    # the terms above it:
    #     a point already scored            1.0 per point
    #   > a fish hooked but not yet landed  HOOKED_WEIGHT per point
    #   > being close to a good fish        at most PROXIMITY_WEIGHT * 11
    #   > scoring one ply earlier           EARLY_BONUS per point per ply
    #
    HOOKED_WEIGHT = 0.9
    PROXIMITY_WEIGHT = 0.3
    EARLY_BONUS = 1e-3
    BOARD_SIZE = 20
    SURFACE_Y = 19

    # ---- Transposition table entry types ------------------------------------
    # With alpha-beta, a stored value is not always exact: a node whose search
    # was cut off only proves a bound.
    EXACT = 0   # all children searched inside the window: the true value
    LOWER = 1   # search failed high (value >= beta): true value >= stored
    UPPER = 2   # search failed low (value <= alpha): true value <= stored

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        # Absolute time (perf_counter) at which the current search must stop.
        self._deadline = 0.0
        # Transposition table for the current move: state key ->
        # (depth_left, value, entry type, best move).
        self._table = {}

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        """
        gc.disable()

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})


    # ----------------------------------------------------------------------
    #  Entry point: iterative deepening
    # ----------------------------------------------------------------------
    def search_best_next_move(self, initial_tree_node):
        """
        Run a complete alpha-beta search to depth 1, then depth 2, then 3, ...
        until the time budget runs out, and play the best move of the DEEPEST
        ITERATION THAT FINISHED.
        """
        self._deadline = time.perf_counter() + self.TIME_BUDGET

        gc.collect()

        # A new table for every move. The keys use node.depth, which is
        # counted from THIS move's root, so an entry from the previous move
        # would describe a different moment in the observation sequence.
        # Within one move the table is shared by all iterations of the
        # iterative deepening, which is what makes its best moves useful for
        # ordering the next, deeper iteration.
        self._table = {}

        # Edge case: the root is already terminal, no move matters.
        if self._is_terminal(initial_tree_node):
            return ACTION_TO_STR[0]  # "stay"

        children = initial_tree_node.compute_and_get_children()

        # Fallback in case not even depth 1 completes.
        best_move = children[0].move

        # Searching deeper than the end of the observation sequence is
        # pointless: every branch is terminal by then.
        max_useful_depth = (len(initial_tree_node.observations)
                            - initial_tree_node.depth)

        depth = 1
        while depth <= max_useful_depth:
            try:
                best_move = self._search_root(children, depth, best_move)
            except SearchTimeout:

                break
            depth += 1

        return ACTION_TO_STR[best_move]

    def _search_root(self, children, depth, previous_best):
        """
        One complete alpha-beta iteration to the given depth.
        """
        alpha = float("-inf")
        beta = float("inf")
        ordered = self._order_root_children(children, previous_best)
        best_move = ordered[0].move

        root_margin = self._margin(children[0].parent.state)
        for child in ordered:
            bonus = self._early_bonus(root_margin, child, depth)
            value = bonus + self._alphabeta(child, depth - 1,
                                            alpha - bonus, beta - bonus)
            # Strict '>': a later child that was cut off returns a BOUND
            # (<= alpha), not its exact value, so it must never replace the
            # current best on a tie.
            if value > alpha:
                alpha = value
                best_move = child.move

        return best_move

    def _order_root_children(self, children, previous_best):
        """
        Put the previous iteration's best move first; keep the rest in order.
        """
        first = [c for c in children if c.move == previous_best]
        rest = [c for c in children if c.move != previous_best]
        return first + rest

    # ----------------------------------------------------------------------
    #  Recursion: minimax with alpha-beta pruning
    # ----------------------------------------------------------------------
    def _alphabeta(self, node, depth_left, alpha, beta):
        """
        Return the minimax value of this node from MAX's (green's) point of
        view, skipping branches that provably cannot change the result.

        """

        if time.perf_counter() > self._deadline:
            raise SearchTimeout()

        # (a) Terminal state -> exact utility. Checked before the depth test:
        #     a game that ends exactly at the cut-off has a known result and
        #     must not be scored as if play continued.
        if self._is_terminal(node):
            return self._utility(node.state)

        # (b) Cut-off -> heuristic evaluation.
        if depth_left == 0:
            return self._evaluate(node.state)

        # (c) Transposition table lookup. Many move orders reach the same
        #     state (e.g. up-then-left and left-then-up; fish move the same
        #     way whatever the hooks do): measured, 93% of the nodes at
        #     depth 5 are repeats. A stored result is reused only if it was
        #     searched at least as deep as needed now, and a bound only if it
        #     already decides this node for the current window.
        key = self._state_key(node)
        entry = self._table.get(key)
        tt_move = None
        if entry is not None:
            entry_depth, entry_value, entry_type, tt_move = entry
            if entry_depth >= depth_left:
                if entry_type == self.EXACT:
                    return entry_value
                if entry_type == self.LOWER and entry_value >= beta:
                    return entry_value
                if entry_type == self.UPPER and entry_value <= alpha:
                    return entry_value

        # (d) Expand. The best move stored for this state (usually by the
        #     previous, shallower iteration) is searched first: alpha-beta
        #     prunes most when the best move comes first. This extends the
        #     root-only ordering to every node.
        children = self._order_children(
            node.compute_and_get_children(), tt_move)
        margin = self._margin(node.state)
        alpha_orig, beta_orig = alpha, beta
        best_move = children[0].move

        # (e) Recurse, narrowing the window as better values are found.
        if node.state.get_player() == 0:
            # MAX node: raise alpha.
            value = float("-inf")
            for child in children:
                bonus = self._early_bonus(margin, child, depth_left)
                child_value = bonus + self._alphabeta(
                    child, depth_left - 1, alpha - bonus, beta - bonus)
                if child_value > value:
                    value, best_move = child_value, child.move
                alpha = max(alpha, value)
                if alpha >= beta:
                    # Beta cut-off
                    break
        else:
            # MIN node: lower beta.
            value = float("inf")
            for child in children:
                bonus = self._early_bonus(margin, child, depth_left)
                child_value = bonus + self._alphabeta(
                    child, depth_left - 1, alpha - bonus, beta - bonus)
                if child_value < value:
                    value, best_move = child_value, child.move
                beta = min(beta, value)
                if alpha >= beta:
                    # Alpha cut-off
                    break

        # (f) Store, recording what kind of value this is relative to the
        #     window the node was searched with. Only completed nodes reach
        #     this line: a SearchTimeout unwinds past it, so an interrupted
        #     search never leaves a wrong entry behind.
        if value <= alpha_orig:
            entry_type = self.UPPER
        elif value >= beta_orig:
            entry_type = self.LOWER
        else:
            entry_type = self.EXACT
        self._table[key] = (depth_left, value, entry_type, best_move)
        return value

    def _order_children(self, children, first_move):
        """Search first_move (if any) first, keep the rest in order."""
        if first_move is None:
            return children
        first = [c for c in children if c.move == first_move]
        rest = [c for c in children if c.move != first_move]
        return first + rest

    def _state_key(self, node):
        """
        Two nodes may share a table entry only if everything that affects the
        search below them is equal:
          * node.depth - the position in the observation sequence, i.e. how
            the fish will move from here; the same board at another depth
            has a different future
          * the player to move (implied by the depth, kept for safety)
          * both hook positions and the fish on each line
          * the score margin - evaluation and utility are relative to it
          * the remaining fish and their positions (a frozenset, so the
            order of the dictionary does not matter)
        Not included: the fish values (constant during the game) and the
        path to the node (the future does not depend on it).
        """
        state = node.state
        hooks = state.get_hook_positions()
        return (node.depth,
                state.get_player(),
                hooks[0], hooks[1],
                state.get_caught(),
                self._margin(state),
                frozenset(state.get_fish_positions().items()))

    # ----------------------------------------------------------------------
    #  Terminal test
    # ----------------------------------------------------------------------
    def _is_terminal(self, node):
        """
        The game is over when either
          * every fish has been landed (no fish positions left), or
          * the observation sequence is exhausted (the fishing day is over).
          
        """
        return (not node.state.get_fish_positions()
                or node.depth >= len(node.observations))

    # ----------------------------------------------------------------------
    #  Exact utility of a terminal state, gamma(A, s)
    # ----------------------------------------------------------------------
    def _utility(self, state):
        """
        The game is over, so the scores on the board are the final scores and
        there is no future catch to account for. From MAX's point of view a
        positive margin is a win, a negative one a loss, zero a draw.
        """
        score_p0, score_p1 = state.get_player_scores()
        return score_p0 - score_p1

    # ----------------------------------------------------------------------
    #  Preference for scoring early
    # ----------------------------------------------------------------------
    def _margin(self, state):
        """Score difference (green - red) of a state."""
        score_p0, score_p1 = state.get_player_scores()
        return score_p0 - score_p1

    def _early_bonus(self, parent_margin, child, depth_left):
        """
        Small extra value for points scored on the move from the parent into
        this child, larger the earlier in the search it happens.
        """
        gain = self._margin(child.state) - parent_margin
        return self.EARLY_BONUS * gain * depth_left

    # ----------------------------------------------------------------------
    #  Evaluation function used at the cut-off, v(A, s)
    # ----------------------------------------------------------------------
    def _evaluate(self, state):
        """
        Estimate the final score difference of a non-terminal state.

            v = (green score - red score)
              + HOOKED_WEIGHT    * (fish on green's line - fish on red's line)
              + PROXIMITY_WEIGHT * (green proximity      - red proximity)
        """
        value = self._margin(state)

        fish_scores = state.get_fish_scores()
        caught = state.get_caught()
        if caught[0] is not None:
            value += self.HOOKED_WEIGHT * fish_scores[caught[0]]
        if caught[1] is not None:
            value -= self.HOOKED_WEIGHT * fish_scores[caught[1]]

        value += self.PROXIMITY_WEIGHT * (self._proximity(state, 0)
                                          - self._proximity(state, 1))
        return value

    def _proximity(self, state, player):
        """
        How promising the best reachable fish is for one player:
        
            max over free fish with positive value of  value / (1 + moves)
            
        """
        caught = state.get_caught()
        if caught[player] is not None:
            return 0.0

        hook_x, hook_y = state.get_hook_positions()[player]
        fish_scores = state.get_fish_scores()
        best = 0.0
        for fish, (fish_x, fish_y) in state.get_fish_positions().items():
            if fish == caught[0] or fish == caught[1]:
                continue
            fish_value = fish_scores[fish]
            if fish_value <= 0:
                continue
            dx = abs(hook_x - fish_x)
            dx = min(dx, self.BOARD_SIZE - dx)          # wrap-around x axis
            moves = dx + abs(hook_y - fish_y) + (self.SURFACE_Y - fish_y)
            best = max(best, fish_value / (1.0 + moves))
        return best
