#!/usr/bin/env python3
"""
DD2380 Artificial Intelligence - A1 Minimax (KTH Fishing Derby)

Implemented by the student (everything below PlayerControllerHuman):
  * search_best_next_move : iterative deepening driver + time budget
  * _alphabeta            : minimax with alpha-beta pruning + transposition table
  * _evaluate             : evaluation function for cut-off (non-terminal) states
  * _utility              : exact utility for terminal states
  * _state_key            : repeated-state key (see report J3/J4)
"""
import gc
import math
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


# --------------------------------------------------------------------------- #
#  Student implementation                                                      #
# --------------------------------------------------------------------------- #

class SearchTimeout(Exception):
    """Raised to unwind the recursion when the per-move time budget is spent."""
    pass


# Transposition-table entry flags
EXACT = 0
LOWER = 1   # value is a lower bound (fail-high / beta cut-off)
UPPER = 2   # value is an upper bound (fail-low)

SPACE_SUBDIVISIONS = 20


class PlayerControllerMinimax(PlayerController):

    # The game controller allows 75 ms per move. We stop well before that so
    # that unwinding the recursion and sending the message still fit.
    TIME_LIMIT = 0.050
    # Upper bound on iterative deepening; never reached in practice.
    MAX_DEPTH = 16

    # Evaluation weights
    W_CAUGHT = 0.9      # a fish on our rod is almost, but not quite, a point
    W_PROXIMITY = 0.05  # tie-breaker only: must never outweigh a real point
    DECAY = 0.25        # how fast the attraction to a fish decays with distance

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.start_time = 0.0
        self.tt = {}
        # Diagnostics (used by bench.py, not by the game itself)
        self.last_depth = 0

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

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

            # Reclaim the game tree *outside* the timed window. The search
            # allocates tens of thousands of State objects per move; letting
            # the cyclic collector run inside the 75 ms window caused pauses
            # of up to ~30 ms (see report, J3).
            node = None
            gc.collect()

    # ----------------------------------------------------------------- #
    #  1. Iterative deepening driver                                     #
    # ----------------------------------------------------------------- #
    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for
        player 0 (green boat).
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        self.start_time = time.time()
        self.tt = {}
        self.last_depth = 0

        # The cyclic garbage collector is paused for the whole search: it is
        # triggered by the allocation of the search tree itself and its pauses
        # are unpredictable (measured up to ~30 ms inside the 75 ms window).
        # It is re-enabled here and the tree is collected in player_loop(),
        # i.e. after the move has already been sent.
        gc.disable()
        try:
            children = initial_tree_node.compute_and_get_children()
            if not children:
                return ACTION_TO_STR[0]  # "stay"

            # Root move ordering starts as-is; after every completed iteration
            # the children are re-sorted by their backed-up value, so the next
            # (deeper) iteration examines the most promising move first.
            ordered = children
            best_move = ordered[0].move

            try:
                for depth in range(1, self.MAX_DEPTH + 1):
                    alpha = -math.inf
                    beta = math.inf
                    scored = []
                    for child in ordered:
                        value = self._alphabeta(child, depth - 1, alpha, beta)
                        scored.append((value, child))
                        if value > alpha:
                            alpha = value
                    # Iteration completed within budget -> commit its result.
                    scored.sort(key=lambda pair: pair[0], reverse=True)
                    best_move = scored[0][1].move
                    ordered = [child for _, child in scored]
                    self.last_depth = depth
            except SearchTimeout:
                # Keep the best move of the last fully completed iteration.
                pass
        finally:
            gc.enable()

        return ACTION_TO_STR[best_move]

    # ----------------------------------------------------------------- #
    #  2. Minimax with alpha-beta pruning + transposition table          #
    # ----------------------------------------------------------------- #
    def _alphabeta(self, node, depth, alpha, beta):
        """
        :param node:  game tree node to evaluate
        :param depth: remaining search depth (plies)
        :param alpha: best value MAX can already guarantee
        :param beta:  best value MIN can already guarantee
        :return: minimax value from the perspective of player 0 (MAX)
        """
        if time.time() - self.start_time > self.TIME_LIMIT:
            raise SearchTimeout()

        alpha_orig = alpha
        beta_orig = beta

        state = node.state
        key = self._state_key(node)

        entry = self.tt.get(key)
        if entry is not None and entry[0] >= depth:
            _, value, flag = entry
            if flag == EXACT:
                return value
            elif flag == LOWER:
                if value > alpha:
                    alpha = value
            else:  # UPPER
                if value < beta:
                    beta = value
            if alpha >= beta:
                return value

        # Terminal test 1: no fish left -> the game is decided.
        if not state.get_fish_positions():
            return self._utility(state)

        # Cut-off test: depth exhausted -> use the evaluation function.
        if depth <= 0:
            return self._evaluate(state)

        children = node.compute_and_get_children()
        # Terminal test 2: the observation sequence is exhausted.
        if not children:
            return self._utility(state)

        maximizing = (state.get_player() == 0)

        # Move ordering: sort the successors by their static evaluation so that
        # the most promising branch is searched first and produces cut-offs.
        # Only worth its own cost when there is still a subtree below.
        if depth > 1 and len(children) > 1:
            children = sorted(children,
                              key=lambda c: self._evaluate(c.state),
                              reverse=maximizing)

        if maximizing:
            value = -math.inf
            for child in children:
                child_value = self._alphabeta(child, depth - 1, alpha, beta)
                if child_value > value:
                    value = child_value
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for child in children:
                child_value = self._alphabeta(child, depth - 1, alpha, beta)
                if child_value < value:
                    value = child_value
                if value < beta:
                    beta = value
                if beta <= alpha:
                    break

        if value <= alpha_orig:
            flag = UPPER
        elif value >= beta_orig:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[key] = (depth, value, flag)

        return value

    # ----------------------------------------------------------------- #
    #  3. Repeated-state detection                                       #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _state_key(node):
        """
        Build a key that identifies a node for the transposition table.

        Two nodes may only share an entry when *everything* that influences the
        remaining search is identical:
          - node.depth   : position in the observation sequence (the fish moves
                           below this node are fully determined by it)
          - player       : whose turn it is
          - hook positions, which fish are still in the water and where,
            which fish are hooked
          - the score difference (the evaluation is relative to it)
        """
        state = node.state
        hooks = state.get_hook_positions()
        caught = state.get_caught()
        score_0, score_1 = state.get_player_scores()
        return (node.depth,
                state.get_player(),
                hooks[0], hooks[1],
                caught[0], caught[1],
                score_0 - score_1,
                tuple(sorted(state.get_fish_positions().items())))

    # ----------------------------------------------------------------- #
    #  4. Utility / evaluation                                           #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _utility(state):
        """Exact utility of a terminal state from MAX's (green) perspective."""
        score_0, score_1 = state.get_player_scores()
        return float(score_0 - score_1)

    def _evaluate(self, state):
        """
        Evaluation function for a cut-off (non-terminal) state.

        value = (score difference)                      <- dominant term
              + 0.9 * (value of the fish on each rod)   <- nearly-secured points
              + 0.05 * (proximity advantage)            <- tie-breaker / gradient
        """
        score_0, score_1 = state.get_player_scores()
        fish_scores = state.get_fish_scores()
        fish_positions = state.get_fish_positions()
        hooks = state.get_hook_positions()
        caught_0, caught_1 = state.get_caught()

        value = float(score_0 - score_1)

        # A hooked fish is worth almost a full point: only "up" moves remain.
        if caught_0 is not None and caught_0 in fish_scores:
            value += self.W_CAUGHT * fish_scores[caught_0]
        if caught_1 is not None and caught_1 in fish_scores:
            value -= self.W_CAUGHT * fish_scores[caught_1]

        # Proximity: the single most attractive reachable fish for each player.
        hook_0 = hooks[0]
        hook_1 = hooks[1]
        best_0 = 0.0
        best_1 = 0.0
        for fish, position in fish_positions.items():
            points = fish_scores[fish]
            if points <= 0:
                # Never steer towards fish that cost points.
                continue
            if fish != caught_0:
                attraction = points * math.exp(
                    -self.DECAY * self._distance(hook_0, position))
                if attraction > best_0:
                    best_0 = attraction
            if fish != caught_1:
                attraction = points * math.exp(
                    -self.DECAY * self._distance(hook_1, position))
                if attraction > best_1:
                    best_1 = attraction

        value += self.W_PROXIMITY * (best_0 - best_1)
        return value

    @staticmethod
    def _distance(hook, fish):
        """Manhattan distance with a wrap-around x axis (the world is round)."""
        dx = abs(hook[0] - fish[0])
        if dx > SPACE_SUBDIVISIONS - dx:
            dx = SPACE_SUBDIVISIONS - dx
        return dx + abs(hook[1] - fish[1])
