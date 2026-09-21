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


class PlayerControllerMinimax(PlayerController):
    MAX_DEPTH = 3

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
    
    def search_best_next_move(self, initial_tree_node):
        children = initial_tree_node.compute_and_get_children()

        # Edge case: the root is already terminal, no legal move exists.
        if not children:
            return ACTION_TO_STR[0]  # "stay"

        best_value = float("-inf")
        best_move = children[0].move

        for child in children:
            # MIN is to move inside a child, so one ply of budget is spent.
            value = self._minimax(child, self.MAX_DEPTH - 1)
            if value > best_value:
                best_value = value
                best_move = child.move

        return ACTION_TO_STR[best_move]

    # ----------------------------------------------------------------------
    #  Recursion (Algorithm 1 of the assignment, plus a depth cut-off)
    # ----------------------------------------------------------------------
    def _minimax(self, node, depth_left):
        
        children = node.compute_and_get_children()

        # (a) Terminal state: no fish left, or the observation sequence is
        #     exhausted (len(observations) == depth). The outcome is now
        #     CERTAIN, so the exact utility is returned rather than an
        #     estimate.

        if not children:
            return self._utility(node.state)

        # (b) Cut-off reached: fall back to the heuristic evaluation.
        if depth_left == 0:
            return self._evaluate(node.state)

        # (c) Recurse. Whose turn it is comes from the state, not from the
        #     parity of the depth. The two agree here, but the transposition
        #     table added in stage 5 must put the player to move in its key,
        #     so both places read it from the same source.
        player = node.state.get_player()

        if player == 0:                      # MAX: green maximises the margin
            best = float("-inf")
            for child in children:
                best = max(best, self._minimax(child, depth_left - 1))
            return best
        else:                                # MIN: red minimises the margin
            best = float("inf")
            for child in children:
                best = min(best, self._minimax(child, depth_left - 1))
            return best

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

        Its weakness is the one the assignment asks about: the value ignores
        the potential of a position entirely. A fish worth 10 points already
        hooked on the green line and one move from the surface looks exactly
        like the same fish still sitting on the sea floor. Stage 4 adds that
        missing term.

        Note that this is deliberately a separate function from _utility even
        though the arithmetic is currently identical. The meaning differs -
        one states a finished result, the other guesses an unfinished one -
        and from stage 4 on the code differs too.
        """
        score_p0, score_p1 = state.get_player_scores()
        return score_p0 - score_p1
