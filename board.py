from typing import Union, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import *
import copy

class Board(object):
    

    BOMB_VALUE = 10
    MARK_VALUE = 20
    HIDDEN_VALUE = -1

    HIDDEN_MASK_VALUE = 0
    REVEALED_MASK_VALUE = 1
    MARKED_MASK_VALUE = 2

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        assert (isinstance(size, int) and size > 0) or (isinstance(size, tuple) and len(
            size) == 2 and size[0] > 0 and size[1] > 0), "Invalid size"
        self.__size = size if isinstance(size, tuple) else (size, size)
        self.reset()

    def reset(self) -> None:
        """
        Reset the board
        """
        self.__board = np.zeros(self.__size, dtype=int)
        self.__mask = np.zeros(self.__size, dtype=int)
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
        return self.__num_of_opens

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

    def __generate_bomb(self) -> Tuple[int, int]:
        number_of_bombs = int(self.__size[0]*self.__size[1]*0.1)
        x = random.choices(np.arange(self.__size[1]), k=number_of_bombs)
        y = random.choices(np.arange(self.__size[0]), k=number_of_bombs)
        self.__bombs = set([(r, c) for r, c in zip(y, x)])
        self.__board[y, x] = Board.BOMB_VALUE

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
        self.__mask[row, col] = 2 - self.__mask[row, col]
        self.__num_of_markers += 1 if self.__mask[row, col] == 2 else -1
        return self

    def plot(self) -> None:
        plt.imshow(self.__board, cmap='hot', interpolation='nearest')
        plt.show()
        plt.imshow(self.__mask, cmap='hot', interpolation='nearest')
        plt.show()
        return self

    def kernel_n(self, size , cell):
        # give a piece of the board where cell(x,y) is the center and size of the piece is size*size
        x, y = cell
        
        radius = size//2
        board_piece = np.zeros((size, size), dtype=int)
        row = 0   
        for i in range( x-radius, x+radius+1):
            col = 0
            for j in range( y-radius,  y+radius+1):
                if i < 0 or i >= self.__size[0] or j < 0 or j >= self.__size[1]:
                    board_piece[row][col] = Board.HIDDEN_VALUE
                elif self.__mask[i, j] == Board.HIDDEN_MASK_VALUE :
                    board_piece[row][col] = Board.HIDDEN_VALUE
                elif self.__mask[i, j] == Board.REVEALED_MASK_VALUE:
                    board_piece[row][col] = self.__board[i,j]
                elif self.__mask[i, j] == Board.MARKED_MASK_VALUE:
                    board_piece[row][col] = Board.BOMB_VALUE
                col += 1
            row += 1
        return board_piece