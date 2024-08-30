from typing import Union, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import *
from utils import EMPTY, FLAG

import copy
import math


class Board(object):

    PRECENTAGE_OF_BOMBS = 0.1

    BOMB_VALUE = 10
    MARK_VALUE = -20
    HIDDEN_VALUE = -1
    OUT_OF_BOUNDS_VALUE = -10

    HIDDEN_MASK_VALUE = 0
    REVEALED_MASK_VALUE = 1
    MARKED_MASK_VALUE = 2

    def __init__(self, size: Union[int, Tuple[int, int]], bomb_density=PRECENTAGE_OF_BOMBS) -> None:
        assert (isinstance(size, int) and size > 0) or (isinstance(size, tuple) and len(
            size) == 2 and size[0] > 0 and size[1] > 0), "Invalid size"
        self.__bomb_density = bomb_density
        self.__size = size if isinstance(size, tuple) else (size, size)
        self.__number_of_bombs = 0
        self.reset()

    def reset(self, keep_the_same=False) -> None:
        """
        Reset the board
        """
        if keep_the_same:
            self.__mask = np.zeros(self.__size, dtype=int)
            return
        self.__board = np.zeros(self.__size, dtype=int)
        self.__mask = np.zeros(self.__size, dtype=int)
        self.__number_of_bombs = 0
        self.__num_of_markers = 0
        self.__num_of_opens = 0
        self.__generate_bomb()
        self.__set_numbers()

    @property
    def avilable_states(self) -> List[Tuple[int, int]]:
        return np.argwhere(self.__mask == 0).tolist()

    def get_actions(self, state: Tuple[int, int]) -> List[str]:
        if self.__mask[state] == 0:
            return ["reveal", "mark"]
        elif self.__mask[state] == 2:
            return ["unmark"]
        return []

    @property
    def size(self) -> Tuple[int, int]:
        return self.__size

    @property
    def num_of_markers(self) -> int:
        return self.__num_of_markers

    @property
    def num_of_opens(self) -> int:
        return np.count_nonzero(self.__mask == 1)

    @property
    def bombs(self) -> Set[Tuple[int, int]]:
        return self.__bombs

    def reveal_all(self) -> None:
        self.__mask = np.ones(self.__size, dtype=int)
        return self

    def __getitem__(self, key: Tuple[int, int]) -> int:
        tmp = self.__board.copy()
        tmp[self.__mask == 0] = -1
        tmp[self.__mask == 2] = -2
        return (tmp[key])

    def __str__(self) -> str:
        return str(self.__board)

    def is_bomb(self, r: int, c: int) -> bool:
        return self.__board[r, c] == Board.BOMB_VALUE

    def is_revealed(self, r: int, c: int) -> bool:
        return self.__mask[r, c] == Board.REVEALED_MASK_VALUE

    def is_marked(self, r: int, c: int) -> bool:
        return self.__mask[r, c] == Board.MARKED_MASK_VALUE

    def __generate_bomb(self, avoid: Tuple[int, int] = None) -> None:
        if self.__bomb_density > 1:
            self.__number_of_bombs = self.__bomb_density
        else:
            self.__number_of_bombs = math.floor(
                int(self.__size[0]*self.__size[1]*self.__bomb_density))
        # Generate all possible cell positions as a list of tuples
        all_positions = [(r, c) for r in range(self.__size[0])
                         for c in range(self.__size[1])]
        if avoid is not None:
            all_positions.remove(avoid)

        # Randomly sample the required number of unique positions
        self.__bombs = []
        while len(self.__bombs) < self.__number_of_bombs:
            bomb = random.choice(all_positions)
            self.__bombs.append(bomb)
            all_positions.remove(bomb)

        # Place the bombs on the board
        for r, c in self.__bombs:
            self.__board[r, c] = Board.BOMB_VALUE

    def __set_numbers(self) -> None:
        x, y, v = [], [], []
        for i in range(self.__size[0]):
            for j in range(self.__size[1]):
                if self.__board[i, j] == Board.BOMB_VALUE:
                    continue
                x.append(i)
                y.append(j)
                v.append(np.count_nonzero(self.__board[max(
                    0, i-1):min(self.__size[0], i+2), max(0, j-1):min(self.__size[1], j+2)]))
        self.__board[x, y] = v

    def apply_action(self, state: Tuple[int, int], action: str) -> None:
        if action in self.get_actions(state):
            if action == "reveal":
                self.reveal(*state)
            elif action == "mark":
                self.mark(*state)
            elif action == "unmark":
                self.mark(*state)
        return self

    def reveal(self, row: int, col: int) -> None:
        assert 0 <= row < self.__size[0] and 0 <= col < self.__size[1], "Invalid position"
        if self.num_of_opens == 0 and self.__board[row, col] == Board.BOMB_VALUE:
            self.__board[:, :] = 0
            self.__generate_bomb((row, col))
            self.__set_numbers()
        if self.__mask[row, col] != 0:
            return
        if self.__board[row, col] == 0:
            self.__mask[row, col] = 1
            for i in range(max(0, row-1), min(self.__size[0], row+2)):
                for j in range(max(0, col-1), min(self.__size[1], col+2)):
                    self.reveal(i, j)
        else:
            self.__mask[row, col] = 1
            self.__num_of_opens += 1
        return self

    def mark(self, row: int, col: int) -> None:
        assert 0 <= row < self.__size[0] and 0 <= col < self.__size[1], "Invalid position"
        if self.__mask[row, col] == 1:
            return
        if self.__mask[row, col] == 2:
            self.__mask[row, col] = 0
            self.__num_of_markers -= 1
        else:
            self.__mask[row, col] = 2
            self.__num_of_markers += 1
        return self

    def plot(self) -> None:
        plt.imshow(self.__board, cmap='hot', interpolation='nearest')
        plt.show()
        plt.imshow(self.__mask, cmap='hot', interpolation='nearest')
        plt.show()
        return self

    def copy(self) -> 'Board':
        """
        This function returns a deep copy of the board
        """
        new_board = Board(self.__size)
        new_board.__board = self.__board.copy()
        new_board.__mask = self.__mask.copy()
        new_board.__num_of_markers = self.__num_of_markers
        new_board.__num_of_opens = self.__num_of_opens
        new_board.__bombs = self.__bombs.copy()
        return new_board

    # Relared to search_problem.py
    def is_solved(self) -> bool:
        """
        This function checks if the board is solved
        """
        return self.num_of_opens == self.__size[0]*self.__size[1] - len(self.__bombs)

    def is_failed(self) -> bool:
        """
        This function checks if the board is failed (i.e. a bomb is revealed)
        """
        return any([self.__mask[b[0], b[1]] == 1 for b in self.__bombs])

    def apply_action(self, action: Tuple[int, int, Literal["reveal", "mark"]], copy: bool = True) -> 'Board':
        """
        This function applies an action to the board

        """
        new_board = self.copy() if copy else self
        if action[2] == "reveal":
            new_board.reveal(action[0], action[1])
        elif action[2] == "mark":
            new_board.mark(action[0], action[1])
        return new_board

    def get_square(self, cell: Tuple[int, int]) -> np.ndarray:
        """
        This function returns the square around a cell
        """
        if cell[0] > 1 and cell[0] + 2 < self.__size[0] and cell[1] > 1 and cell[1]+2 < self.__size[1]:
            return self[cell[0]-2:cell[0]+3, cell[1]-2:cell[1]+3]
        tmp = np.full((self.size[0]+4, self.size[1]+4),
                      Board.OUT_OF_BOUNDS_VALUE)
        tmp[2:-2, 2:-2] = self[:, :]
        return tmp[cell[0]:cell[0]+5, cell[1]:cell[1]+5]

    def what_action_for_this_cell(self, cell: Tuple[int, int]) -> Literal["reveal", "mark", "noop", "optional"]:
        """
        This function returns the best action for a cell
        """
        sq = self.relax_square(self.get_square(cell))
        for i in range(1, 4):
            for j in range(1, 4):
                if i == 2 and j == 2:
                    continue
                # if sq[i, j] == 1:
                #     if np.count_nonzero(sq[i-1:i+2, j-1:j+2] != Board.HIDDEN_VALUE) == 8:
                #         return 'mark'  # The cell is good for flagging
                if sq[i, j] > 0:
                    if np.count_nonzero(sq[i-1:i+2, j-1:j+2] == Board.HIDDEN_VALUE) == sq[i, j]:
                        return 'mark'  # The cell is good for flagging
                if sq[i, j] == 0:
                    return 'reveal'  # The cell is good for revealing
        if np.count_nonzero(sq[1:4, 1:4] != Board.HIDDEN_VALUE) + np.count_nonzero(sq[1:4, 1:4] != Board.OUT_OF_BOUNDS_VALUE) == 9:
            return 'noop'
        return 'optional'

    def get_prob_for_optinal_cells(self, cell: Tuple[int, int]) -> float:
        """
        This function returns the probability of a cell to be a bomb
        """
        sq = self.relax_square(self.get_square(cell))
        if sq[2, 1] == Board.HIDDEN_VALUE and sq[2, 3] == Board.HIDDEN_VALUE and sq[1, 2] == Board.HIDDEN_VALUE and sq[3, 2] == Board.HIDDEN_VALUE:
            return 1
        prob = 1
        for i in range(1, 4):
            for j in range(1, 4):
                if i == 2 and j == 2:
                    continue
                if sq[i, j] == Board.HIDDEN_VALUE:
                    continue
                _prob = (sq[i, j]) / \
                    (np.count_nonzero(sq[i-1:i+2, j-1:j+2]
                     == Board.HIDDEN_VALUE))
                prob *= (1-_prob)
        return 1-prob

    def relax_square(self, square: np.ndarray) -> np.ndarray:
        """
        This function relaxes the square
        """
        for i in range(1, 4):
            for j in range(1, 4):
                if square[i, j] > 0:
                    square[i,
                           j] -= np.count_nonzero(square[i-1:i+2, j-1:j+2] == FLAG)
        return square

    def get_actions(self) -> List[Tuple[int, int, Literal["reveal", "mark", "noop"]]]:
        """
        This function returns the list of actions that can be taken
        """
        actions = []
        noops = []
        for cell in self.avilable_states:
            act = self.what_action_for_this_cell(cell)
            if act != 'optional' and act != 'noop':
                actions.append((*cell, act))
            elif act == 'optional':
                noops.append((*cell, 'reveal'))
        if len(actions) == 0:
            if self.num_of_opens < 1:
                cell = random.choice(self.avilable_states)
                return [(*cell, 'reveal')]
            else:
                scores = [self.get_prob_for_optinal_cells(
                    (x[0], x[1])) for x in noops]
                best_prob = min(scores)
                best_actions = [x for x, y in zip(
                    noops, scores) if y == best_prob]
                return best_actions

        return actions
