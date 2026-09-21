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
#  Stage 3: iterative deepening alpha-beta under a time budget.
# ==========================================================================

class SearchTimeout(Exception):
    """
    Raised inside the recursion when the time budget is spent.

    An exception is used instead of a return value because the search can be
    many calls deep when time runs out. The exception unwinds all of them at
    once, and no half-finished value can leak into a decision: the result of
    the interrupted iteration is simply never used.
    """
    pass


class PlayerControllerMinimax(PlayerController):

    # Time budget for one move, in seconds. The hard limit is 75 ms, but the
    # budget has to leave room for work outside the search: building the
    # root Node from the message, sending the answer back, the time between
    # the last deadline check and the moment the exception reaches the top,
    # and a judge machine that may be slower than this one.
    TIME_BUDGET = 55e-3

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        # Absolute time (perf_counter) at which the current search must stop.
        self._deadline = 0.0

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Turn off automatic garbage collection for the whole game. The game
        # tree is full of reference cycles (Node.parent <-> Node.children)
        # that only the cyclic collector can free, and left on automatic it
        # starts at arbitrary moments - often in the middle of a search.
        # Instead, search_best_next_move() collects once per move, at a fixed
        # point, and counts that time against the move's budget.
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

        Why this replaces the fixed depth of stage 2:
          * The depth adapts to the position. Simple positions (few fish, a
            hooked fish forcing "up") reach much deeper; hard ones stop early
            instead of overrunning the 75 ms limit.
          * There is always an answer ready. Depth 1 finishes in well under a
            millisecond, and each later iteration only replaces the answer
            once it has completed.

        Why repeating the shallow iterations is cheap:
          * The game tree grows by a factor of about 5 per ply, so all the
            shallower iterations together cost roughly a quarter of the last
            one.
          * Node caches its children (compute_and_get_children), so the next
            iteration walks through states that were already generated and
            only pays for the new bottom layer.
          * The previous iteration tells us which root move is probably best,
            and it is searched first (see _order_root_children).

        :param initial_tree_node: game_tree.Node, the root, depth == 0
        :return: one of "stay", "up", "down", "left", "right"
        """
        self._deadline = time.perf_counter() + self.TIME_BUDGET

        # Free the previous move's game tree. This is done here, AFTER the
        # deadline has been fixed, so its cost is paid out of this move's
        # budget: a slow collection shortens the search instead of pushing
        # the answer past the limit.
        #
        # An earlier version collected right after sending the answer,
        # assuming that time was free. It is not: the judge replies quickly,
        # so the next message arrives while the collection is still running,
        # and that delay counts against the next move. A collection takes
        # 17 ms on average and up to 33 ms here, so search + collection
        # reached 92 ms. That is the likely cause of the one run-time error
        # on Kattis (24/25 passed).
        gc.collect()

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
                # The unfinished iteration is discarded entirely. Its partial
                # result only reflects the root moves it got to, so it could
                # prefer a move merely because the better one was never
                # reached.
                break
            depth += 1

        return ACTION_TO_STR[best_move]

    def _search_root(self, children, depth, previous_best):
        """
        One complete alpha-beta iteration to the given depth.

        :param children: the root's children
        :param depth: search depth of this iteration
        :param previous_best: best move of the previous iteration, searched
                              first
        :return: the best move (int) at this depth
        :raises SearchTimeout: if the budget runs out before it finishes
        """
        alpha = float("-inf")
        beta = float("inf")
        ordered = self._order_root_children(children, previous_best)
        best_move = ordered[0].move

        for child in ordered:
            value = self._alphabeta(child, depth - 1, alpha, beta)
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

        Alpha-beta prunes most when the best move is searched first: its
        value becomes alpha immediately, and every other root move can then
        be refuted by red's first good reply instead of being searched in
        full. The best move at depth d-1 is very often still the best at
        depth d, so it is the cheapest good guess available.

        Ordering never changes the minimax value, only how much is pruned.

        Measured effect in this stage is small (average completed depth 6.75
        with it, 6.73 without, on 93 test positions). Two reasons: it only
        reorders the root, while most of the tree lies below it; and with the
        flat score-difference evaluation most moves tie at the same value, so
        there is rarely a clearly best move to put first. Stage 5 extends the
        idea to every node via the transposition table.
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
        # Abort the whole iteration once the budget is spent. Checked at every
        # node: the check costs far less than generating one state, and a
        # coarser check (e.g. every N nodes) would make the overshoot past
        # the deadline depend on how expensive those N nodes happen to be.
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
