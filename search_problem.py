from search import SearchProblem
from board import Board
from typing import Tuple, Literal
import numpy as np
import random

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
        return self.get_next_action(state)

    def __calculate_probabilities(self, vec: np.ndarray):
        """
        Calculate the probability of each cell to be a mine
        Given vector of 8
        """
        num_hidden = np.sum(vec[:, 1:] == Board.HIDDEN_VALUE, axis=1)
        num_markers = np.sum(vec[:, 1:] == Board.MARK_VALUE, axis=1)
        num_mines = vec[:, 0]
        prob = (num_mines - num_markers) / num_hidden
        return prob

    def __process_options(self, option: np.ndarray):
        """
        Given a 5x5 grid, process the options into a list of vectors
        """
        option = option.flatten()
        options = []
        option[6] > 0 and options.append(
            option[[6, 0, 1, 2, 7, 12, 11, 10, 5]])
        option[7] > 0 and options.append(
            option[[7, 1, 2, 3, 8, 13, 12, 11, 6]])
        option[8] > 0 and options.append(
            option[[8, 2, 3, 4, 9, 14, 13, 12, 7]])
        option[11] > 0 and options.append(
            option[[11, 5, 6, 7, 12, 17, 16, 15, 10]])
        option[13] > 0 and options.append(
            option[[13, 7, 8, 9, 14, 19, 18, 17, 12]])
        option[16] > 0 and options.append(
            option[[16, 10, 11, 12, 17, 22, 21, 20, 15]])
        option[17] > 0 and options.append(
            option[[17, 11, 12, 13, 18, 23, 22, 21, 16]])
        option[18] > 0 and options.append(
            option[[18, 12, 13, 14, 19, 24, 23, 22, 17]])
        options = np.array(options)
        return options

    def get_next_action(self, state: Board):
        """
        Given a state, return the next action to take
        """
        if state.num_of_opens == 0:
            random_action = (*random.choice(state.avilable_states), "reveal")
            return [(state.apply_action(random_action), random_action, 0)]
        next_actions = []
        _board = np.full((self.board_size[0]+4, self.board_size[1]+4),
                         Board.OUT_OF_BOUNDS_VALUE)
        _board[2:-2, 2:-2] = state[:, :]
        options = np.array(state.avilable_states)
        for option in options:
            _option = _board[option[0]:option[0]+5, option[1]:option[1]+5]
            if np.all(_option[1:-1, 1:-1] != Board.HIDDEN_VALUE):
                continue
            is_close = _option[1, 2] > 0 or _option[2, 1] > 0 or _option[2,
                                                                         3] > 0 or _option[3, 2] > 0
            _option = self.__process_options(_option)
            if _option.shape[0] == 0:
                continue
            _probs = self.__calculate_probabilities(_option)
            if np.any(_probs == 1):
                action = (option[0], option[1], "mark")
                successur = state.apply_action(action)
                next_actions.append((successur, action, 0))
            elif np.any(_probs == 0):
                action = (option[0], option[1], "reveal")
                successur = state.apply_action(action)
                next_actions.append((successur, action, 1))
            elif is_close:
                action = (option[0], option[1], "reveal")
                successur = state.apply_action(action)
                next_actions.append(
                    (successur, action, np.max(_probs)+1))
        if len(next_actions) == 0:
            price = 0
            if state.num_of_bombs - state.num_of_markers == len(state.avilable_states):
                action = (*random.choice(state.avilable_states), "mark")
                price = 0
            else:
                action = (*random.choice(state.avilable_states), "reveal")
                price = 1
            successur = state.apply_action(action)
            next_actions.append((successur, action, price))
        return next_actions

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
