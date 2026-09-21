#!/usr/bin/env python3
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
#  Stage 2: fixed-depth minimax with alpha-beta pruning.
#  Targets the C requirement: a correct minimax search combined with a
#  technique that lets it search effectively under the time limit.
# ==========================================================================

class PlayerControllerMinimax(PlayerController):

    # Fixed search depth (root = depth 0, +1 per transition).
    #
    # Measured on 93 positions from the four test scenarios, worst move time:
    #     depth 4: 28 ms    depth 5: 36 ms    depth 6: 97 ms (over 75 ms)
    # Depth 5 is the deepest that stays at about half the budget, leaving a
    # margin for a slower judge machine. Plain minimax without pruning could
    # only afford depth 4 (worst 34 ms), and only after the terminal-test fix
    # below; before that fix it could only afford depth 3.
    #
    # The depth is still chosen for the WORST position, so most moves finish
    # far below the limit (depth 5 averages 11 ms). Stage 3 replaces this
    # constant with iterative deepening under a time budget.
    MAX_DEPTH = 5

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

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

    # ----------------------------------------------------------------------
    #  Entry point. The root is a MAX node: the green player (player 0) moves.
    # ----------------------------------------------------------------------
    def search_best_next_move(self, initial_tree_node):
        """
        Evaluate the children of the root with alpha-beta and play the move
        with the highest value.

        The root is itself a MAX node, so it takes part in the pruning: once
        the first child has been searched, its value becomes alpha, and every
        later child is searched with the window (alpha, +inf). Inside such a
        child, as soon as red finds a reply that holds green to <= alpha, the
        rest of that child is skipped - green already has a move that is at
        least that good.

        beta stays +inf at the root because nothing above the root can
        restrict what MAX is allowed to achieve.

        :param initial_tree_node: game_tree.Node, the root, depth == 0
        :return: one of "stay", "up", "down", "left", "right"
        """
        # Edge case: the root is already terminal, no move matters.
        if self._is_terminal(initial_tree_node):
            return ACTION_TO_STR[0]  # "stay"

        children = initial_tree_node.compute_and_get_children()

        alpha = float("-inf")
        beta = float("inf")
        best_move = children[0].move

        for child in children:
            value = self._alphabeta(child, self.MAX_DEPTH - 1, alpha, beta)
            # Strict '>' keeps the first of several equally good moves. This
            # matters with pruning: a later child that was cut off returns a
            # BOUND (<= alpha), not its exact value, so it must never replace
            # the current best on a tie.
            if value > alpha:
                alpha = value
                best_move = child.move

        return ACTION_TO_STR[best_move]

    # ----------------------------------------------------------------------
    #  Recursion: minimax with alpha-beta pruning
    # ----------------------------------------------------------------------
    def _alphabeta(self, node, depth_left, alpha, beta):
        """
        Return the minimax value of this node from MAX's (green's) point of
        view, skipping branches that provably cannot change the result.

        The two bounds carry information down from the ancestors:
          alpha - the best value MAX is already guaranteed somewhere on the
                  path to the root. MAX will never accept less.
          beta  - the best value MIN is already guaranteed somewhere on the
                  path to the root. MIN will never allow more.

        Only values strictly between alpha and beta can still influence the
        decision at the root. As soon as alpha >= beta the window is empty:
        whichever value the remaining children have, one of the two players
        higher up will avoid this node, so the remaining children are pruned.

        The result is exactly the minimax value whenever that value lies
        inside the original window. When a node is pruned it returns a bound
        instead (fail-soft: the best value seen so far), which is still
        enough for its parent to make the correct choice.

        :param node: the node being evaluated
        :param depth_left: plies still allowed below this node; 0 = cut-off
        :param alpha: lower bound, the value MAX can already force
        :param beta: upper bound, the value MIN can already force
        :return: float
        """
        # (a) Terminal state -> exact utility. Checked before the depth test:
        #     a game that ends exactly at the cut-off has a known result and
        #     must not be scored as if play continued.
        if self._is_terminal(node):
            return self._utility(node.state)

        # (b) Cut-off -> heuristic evaluation.
        if depth_left == 0:
            return self._evaluate(node.state)

        # Children are generated only now, after both stopping tests.
        # Generating them first (as stage 1 did) meant every cut-off leaf
        # built its 5 children just to discover it had some - a whole extra
        # ply of state generation that was then thrown away.
        children = node.compute_and_get_children()

        # (c) Recurse, narrowing the window as better values are found.
        if node.state.get_player() == 0:
            # MAX node: raise alpha.
            value = float("-inf")
            for child in children:
                value = max(value,
                            self._alphabeta(child, depth_left - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    # Beta cut-off: MIN, one level up, already has a reply
                    # that keeps green at <= beta. Green can reach >= beta
                    # here, so MIN will never let the game enter this node.
                    break
            return value
        else:
            # MIN node: lower beta.
            value = float("inf")
            for child in children:
                value = min(value,
                            self._alphabeta(child, depth_left - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    # Alpha cut-off: MAX, one level up, already has a move
                    # worth >= alpha. Red can push green down to <= alpha
                    # here, so MAX will never choose this node.
                    break
            return value

    # ----------------------------------------------------------------------
    #  Terminal test
    # ----------------------------------------------------------------------
    def _is_terminal(self, node):
        """
        The game is over when either
          * every fish has been landed (no fish positions left), or
          * the observation sequence is exhausted (the fishing day is over).

        Both are read directly from the node, without generating children.

        Note the skeleton's compute_and_get_children() only knows about the
        second condition: with no fish left it still returns 5 children.
        Relying on "no children means terminal" would therefore miss the
        first way the game can end, and would also force a full expansion of
        every leaf just to perform the test.
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
    #  Evaluation function used at the cut-off, v(A, s)
    # ----------------------------------------------------------------------
    def _evaluate(self, state):
        """
        The basic evaluation function suggested in the assignment:

            v(A, s) = Score(green boat) - Score(red boat)

        It is reasonable because it is expressed in the same unit as the
        terminal utility, and because it costs O(1) to read straight out of
        the state. An evaluation function is called tens of thousands of
        times per move, so its cost matters as much as its accuracy.

        Its weakness: the value ignores the potential of a position. A fish
        already hooked on the green line looks exactly like the same fish
        still on the sea floor. Stage 4 adds that missing term.

        Deliberately separate from _utility even though the arithmetic is
        currently identical: one states a finished result, the other guesses
        an unfinished one.
        """
        score_p0, score_p1 = state.get_player_scores()
        return score_p0 - score_p1
