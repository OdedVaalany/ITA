from sklearn import tree
from util import getState
from board import Board

import numpy as np
from itertools import product, combinations
from itertools import product

def generate_all_minesweeper_boards(size=4):
    all_boards = []
    set_of_made_boards = set()
    
    # Generate all possible mine placements
    for mines in product([0, 1], repeat=size*size):
        mines = np.array(mines).reshape((size, size))
        board = np.copy(mines)
        
        # Replace 1s with -1 to represent mines
        board[board == 1] = -1
        
        # Calculate numbers based on mine placement
        for i in range(size):
            for j in range(size):
                if board[i][j] == -1:
                    continue  # Skip mines
                mine_count = 0
                # Check all adjacent cells for mines
                for x in range(max(0, i-1), min(size, i+2)):
                    for y in range(max(0, j-1), min(size, j+2)):
                        if board[x][y] == -1:
                            mine_count += 1
                board[i][j] = mine_count
        inner_board = board[1:, 1:]
        if str(inner_board) not in set_of_made_boards:
            set_of_made_boards.add(str(inner_board))
            all_boards.append(inner_board)
    return all_boards

def generate_all_hidden_boards(boards):
    hidden_boards = []
    size = boards[0].shape[0]  # Assuming all boards are square and of the same size
    
    # Generate all possible ways to hide at least one cell
    cell_indices = [(i, j) for i in range(size) for j in range(size)]
    
    for board in boards:
        for num_hidden in range(1, size*size + 1):  # Hide at least 1 cell up to all cells
            for hidden_indices in combinations(cell_indices, num_hidden):
                hidden_board = np.copy(board)
                for i, j in hidden_indices:
                    hidden_board[i][j] = 999  # Use 999 to represent hidden cells
                hidden_boards.append(hidden_board)
    
    return hidden_boards
# Example usage:
boards = generate_all_minesweeper_boards()
boards = generate_all_hidden_boards(boards)
print(f"Number of boards: {len(boards)}")

# Print the first 3 boards
for idx, board in enumerate(boards[:3]):
    print(f"Board {idx + 1}:\n{board}\n")



    

