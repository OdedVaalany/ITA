from search import SearchProblem
from board import Board
from typing import Tuple, Literal

Action = Tuple[int, int, Literal["reveal", "mark", "noop"]]


class MinesweeperSearchProblem(SearchProblem):

    def __init__(self, board: Board):
        self.board: Board = board
        self.board_size = board.size

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        return state.is_solved() or state.is_failed()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        successors = []
        for action in state.get_actions():
            new_state = state.apply_action(action)
            cost = self.get_cost_of_action(action)
            successors.append((new_state, action, cost))
        return successors

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        cost = 0
        for action in actions:
            cost += self.get_cost_of_action(action)
        return cost

    def get_cost_of_action(self, action: Action):
        if action[2] == "reveal":
            return -1
        elif action[2] == "mark":
            return -2
        elif action[2] == "noop":
            0
        else:
            return 0


def huristic(state: Board, problem: MinesweeperSearchProblem = None):
    return state.num_of_markers + state.num_of_opens
