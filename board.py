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
    MARK_VALUE = -2
    HIDDEN_VALUE = -1

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

    def reset(self) -> None:
        """
        Reset the board
        """
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

    def __generate_bomb(self):
        if self.__bomb_density > 1:
            self.__number_of_bombs = self.__bomb_density
        else:
            self.__number_of_bombs = math.floor(
                int(self.__size[0]*self.__size[1]*self.__bomb_density))
        # Generate all possible cell positions as a list of tuples
        all_positions = [(r, c) for r in range(self.__size[0])
                         for c in range(self.__size[1])]

        # Randomly sample the required number of unique positions
        self.__bombs = set(random.sample(
            all_positions, self.__number_of_bombs))
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
        num_of_opens = self.__size[0]*self.__size[1] - len(self.__bombs)
        return (self.__num_of_markers ==  len(self.__bombs) and len(self.avilable_states) == 0)

    def is_failed(self) -> bool:
        """
        This function checks if the board is failed (i.e. a bomb is revealed)
        """
        return any([self.__mask[b[0], b[1]] == 1 for b in self.__bombs]) or len(self.avilable_states) == 0

    # def apply_action(self, action: Tuple[int, int, Literal["reveal", "mark"]]) -> 'Board':
    #     """
    #     This function applies an action to the board
    #     """
    #     new_board = self.copy()
    #     if action[2] == "reveal":
    #         new_board.reveal(action[0], action[1])
    #     elif action[2] == "mark":
    #         new_board.mark(action[0], action[1])
    #     return new_board

    def get_square(self, cell: Tuple[int, int]) -> np.ndarray:
        """
        This function returns the square around a cell
        """
        if cell[0] > 1 and cell[0] + 2 < self.__size[0] and cell[1] > 1 and cell[1]+2 < self.__size[1]:
            return self[cell[0]-2:cell[0]+3, cell[1]-2:cell[1]+3]
        tmp = np.full((self.size[0]+4, self.size[1]+4), -10)
        tmp[2:-2, 2:-2] = self[:, :]
        return tmp[cell[0]:cell[0]+5, cell[1]:cell[1]+5]

    def what_action_for_this_cell(self, cell: Tuple[int, int]) -> Literal["reveal", "mark", "noop"]:
        """
        This function returns the best action for a cell
        """
        sq = self.relax_square(self.get_square(cell))
        for i in range(1, 4):
            for j in range(1, 4):
                if i == 2 and j == 2:
                    continue
                if sq[i, j] == 1:
                    if np.count_nonzero(sq[i-1:i+2, j-1:j+2] != EMPTY) == 8:
                        return 'mark'  # The cell is good for flagging
                if sq[i, j] == 2:
                    if np.count_nonzero(sq[i-1:i+2, j-1:j+2] != EMPTY) == 7:
                        return 'mark'  # The cell is good for flagging
                if sq[i, j] == 0:
                    return 'reveal'  # The cell is good for revealing
        return 'noop'

    def relax_square( square: np.ndarray) -> np.ndarray:
        """
        This function relaxes the square
        """
        for i in range(1, 4):
            for j in range(1, 4):
                if square[i, j] > 0:
                    square[i,
                           j] -= np.count_nonzero(square[i-1:i+2, j-1:j+2] == FLAG)
        return square

    # def get_actions(self) -> List[Tuple[int, int, Literal["reveal", "mark", "noop"]]]:
    #     """
    #     This function returns the list of actions that can be taken
    #     """
    #     actions = []
    #     for cell in self.avilable_states:
    #         act = self.what_action_for_this_cell(cell)
    #         if act != 'noop':
    #             actions.append((*cell, act))
    #     if len(actions) == 0:
    #         if self.num_of_opens < 4:
    #             if self.__mask[0, 0] == 0:
    #                 actions += [(0, 0, 'reveal')]
    #             if self.__mask[0, self.size[1]-1] == 0:
    #                 actions += [(0, self.size[1]-1, 'reveal')]
    #             if self.__mask[self.size[0]-1, 0] == 0:
    #                 actions += [(self.size[0]-1, 0, 'reveal')]
    #             if self.__mask[self.size[0]-1, self.size[1]-1] == 0:
    #                 actions += [(self.size[0]-1, self.size[1]-1, 'reveal')]
    #         else:
    #             random_cell = random.choice(self.avilable_states)
    #             actions += [(random_cell[0], random_cell[1], 'reveal')]

    #     return actions
    
    def open_first(self):
        for row in range(self.__size[0]):
            for col in range(self.__size[1]):
                if self.__board[row, col] == 0:
                    self.apply_action((row, col), "reveal")
                    return self
        
        return self
    

    def print_current_board(self):
    
        tmp = self.__board.copy()
        tmp[self.__mask == 0] = -1
        tmp[self.__mask == 2] = -2
        print(tmp)
        return self
    
    def set_board(self, board, mask):
        self.__board = board
        self.__mask = mask
        self.__number_of_bombs = 0
        self.__num_of_markers = 0
        self.__num_of_opens = 0
        self.__set_numbers()
        return self
    def get_board(self):
        tmp = self.__board.copy()
        tmp[self.__mask == 0] = -1
        tmp[self.__mask == 2] = -2
        return tmp
