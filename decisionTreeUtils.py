from board import Board
import random
import pickle
import numpy as np
from DecisionTree import Boards
import concurrent.futures
from board import Board
import itertools
import numpy as np

MARK_TAG = 0
REVEAL_TAG = 1

def random_choice(board):
    done = True
    available_states = board.avilable_states
    random.shuffle(available_states)
    for cell in available_states:
        cell = (cell[0], cell[1])
        if not board.is_revealed(cell[0],cell[1]) or not board.is_marked(cell[0],cell[1]):
            return cell
    return None

def make_one_move(board, agent):
    ##make move if pass the filter (3 revealed cells around the middle cell) or random cell
    for row in range(board.size[0]):
        for col in range(board.size[1]):
            cell = (row, col)
            if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
                continue
            state = board.get_square(cell)
            if filter_states(state):
                state = feature_vector(state)
                prediction = agent.decision(state)
                print(prediction , "prediction")
                if prediction == REVEAL_TAG:
                    board.apply_action(cell, "reveal")

                    if board.is_bomb(cell[0],cell[1]):
                        done = True
                        return done
                    else:
                        done = False
                        return done
                elif prediction == MARK_TAG:
                    board.apply_action(cell, "mark")
                    done = False
                    return done
    
    random_cell = random_choice(board)
    board.apply_action((random_cell[0],random_cell[1]), "reveal")
    if(board.is_bomb(random_cell[0],random_cell[1])):
        done = True
    else:
        done = False
    return done

def agent_play(board , agent):
    done = False
    move_counter = 0
    while not done:
        make_one_move(board, agent)
        move_counter += 1
        
        if(board.is_solved()):
            return True
        if(board.is_failed()):
            done = True
            print(f"Game Over! Total moves: {move_counter}")
            return False
        board.print_current_board() 
    
    return False


def filter_states(state):
        return True
        # size = len(state)
        # middle = size//2
        # count = 0
        # if (state[middle][middle] !=  Board.HIDDEN_VALUE):
        #     return False
        # for i in range(-1,2):
        #     for j in range(-1,2):
        #         if i ==j and i == 0:
        #             continue
        #         if state[middle+i][middle+j] != Board.HIDDEN_VALUE:
        #             count+=1
        #             if count > 0:
        #                 return True
        
        # return False


def feature_vector(state):
    # relaxed = Board.relax_square(state)
    flatten_state = np.array(state).flatten()
    return flatten_state
        
def random_correct(board):
    done = True
    available_states = board.avilable_states
    random.shuffle(available_states)
    for cell in available_states:
        if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
            continue
        if board.is_bomb(cell[0],cell[1]):
            board.apply_action((cell[0],cell[1]), "mark")
            return cell, MARK_TAG
        else:
            board.apply_action((cell[0],cell[1]), "reveal")
            return cell, REVEAL_TAG
    
    return None, None


            

def generate_bomb_arrangements(k):
    # Initialize an empty 3x3 matrix
    base_matrix = np.zeros((3, 3), dtype=int)
    
    # Get all possible positions in the 3x3 grid
    all_positions = [(i, j) for i in range(3) for j in range(3)]
    
    # Generate all combinations of positions where bombs will be placed
    bomb_combinations = itertools.combinations(all_positions, k)
    
    # List to store all possible bomb arrangements
    bomb_arrangements = []
    
    for bomb_positions in bomb_combinations:
        # Create a new matrix from the base
        new_matrix = base_matrix.copy()
        
        # Place bombs (denoted by 1) at the selected positions
        for pos in bomb_positions:
            new_matrix[pos] = 10
        
        bomb_arrangements.append(new_matrix)
    
    return bomb_arrangements

    
def generate_hidden_matrices( k):
    # Start with a 3x3 matrix with all cells visible (0)
    base_matrix = np.ones((3, 3), dtype=int)
    
    # Get all possible positions (9 cells in total)
    all_positions = [(i, j) for i in range(3) for j in range(3)]
    
    # Generate all possible combinations of positions to hide
    hidden_combinations = itertools.combinations(all_positions, k)
    
    # List to hold all matrices
    matrices = []
    
    for hidden_positions in hidden_combinations:
        # Create a new matrix from the base
        new_matrix = base_matrix.copy()
        
        # Hide the selected positions (set them to -1)
        for pos in hidden_positions:
            new_matrix[pos] = 0
        
        matrices.append(new_matrix)
    
    return matrices


def change_confilicting_tags(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Resolves conflicting samples in X where identical samples have different tags in y.
    Changes the tags of such samples to 2.

    Parameters:
    X (np.ndarray): The array of samples (shape: [n_samples, n_features]).
    y (np.ndarray): The array of tags (shape: [n_samples]).

    Returns:
    np.ndarray: The modified array of tags with conflicts resolved.
    """
    # Create a dictionary to track the indices of identical samples
    sample_dict = {}
    
    for i, sample in enumerate(X):
        # Convert the sample to a tuple to use as a dictionary key
        sample_tuple = tuple(sample)
        
        if sample_tuple in sample_dict:
            # Check if there's a conflicting tag
            existing_index = sample_dict[sample_tuple]
            if y[existing_index] != y[i]:
                # print("happens")
                # Change both tags to 2
                y[existing_index] = 2
                y[i] = 2
        else:
            # Store the index of the sample
            sample_dict[sample_tuple] = i
    
    return X, y
    


def make_action_for_Training(board ):
        tag = None
        available_states = board.avilable_states
        random.shuffle(available_states)
        for cell in available_states:
            cell = (cell[0], cell[1])
            if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
                continue
            state = board.get_square(cell)
            if filter_states(state):
                np_state = feature_vector(state)
                if board.is_bomb(cell[0],cell[1]):
                    board.apply_action(cell, "mark")
                    tag = MARK_TAG
                    add_to_data = True
                    return np_state, tag , add_to_data 
                else:
                    board.apply_action(cell, "reveal")
                    tag = REVEAL_TAG
                    add_to_data = True
                    return np_state, tag , add_to_data
        
        #make random move that doesn    t pick bomb if no state passed the filter
        
        if board.is_solved() or board.is_failed():
            return None, None, False      
        cell,tag = random_correct(board)
        
    
        np_state = feature_vector(board.get_square(cell))
        add_to_data = False
        return np_state, tag , add_to_data



def run_episode(episode, board , num_episodes):
    X = []
    y = []
    
    # Pick a random board
    
    win = 0
    for i in range(num_episodes):
        board = board.copy()
        board.reset()
        board.open_first()
        done = False
        
        while not done:
            np_state, tag, add_to_data = make_action_for_Training(board)
            
            # Add to the data only if the state is not in the map or the tag is different
            if add_to_data:
                X.append(np_state)
                y.append(tag)
            
            if board.is_solved() or board.is_failed():
                done = True

    
    
    return X, y , win


def expend_features(X):
    new_X = []
    for j in range(len(X)):
        sample = X[j]
        new_features = np.zeros((len(sample), 12))  # 12 is the number of possible values

        for i in range(len(sample)):
            if sample[i] == -10:
                new_features[i][11] = 1
            elif sample[i] == -1:
                new_features[i][10] = 1
            elif sample[i] == -2:
                new_features[i][9] = 1
            else:
                new_features[i][sample[i]] = 1
        new_X.append(new_features.flatten())
    return np.array(new_X)


def run_parallel_episodes(num_episodes, Boards, file_name):
    X = []
    y = []
    batch_episodes = 1000
    # Use ProcessPoolExecutor to run episodes in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_episodes//batch_episodes):
            
            print("crated" ,i)
                # print(f"Processed {i + 1}/{num_episodes} episodes")
                # print(f"Data size: {len(X)} samples")
            futures.append(executor.submit(run_episode, i, Boards , batch_episodes) )
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            episode_X, episode_y ,episode_wins= future.result()
            X.extend(episode_X)
            y.extend(episode_y)
           
            print(f"Processed {i + 1}/{num_episodes//batch_episodes} episodes")
            print(f"Data size: {len(X)} , {len(y)} samples")
            if i % 1000000 == 0:
                with open(file_name, 'wb') as f:
                    pickle.dump((X, y), f)
            
            
    
    with open(file_name, 'wb') as f:
        pickle.dump((X, y), f)
    
    return len(X), len(y)

def create_data_set(boards , num_episodes,file_name):
    return run_parallel_episodes(num_episodes , boards , file_name)


def create_all_possible_states(a,b):
    states = []
    y = []
    for number_of_bombs in range(10):  # 0 to 9 bombs
        bomb_arrangements = generate_bomb_arrangements(number_of_bombs)
        for matrix in bomb_arrangements:
            matrix = generate_numbers(matrix)
            for number_of_hidden in range(10):  # 0 to 9 hidden cells
                hidden_matrices = generate_hidden_matrices(number_of_hidden)
                for mask in hidden_matrices:
                    state , tag= create_state(matrix, mask , a,b)
                    states.append(np.array(state).flatten())
                    y.append(tag)
    return states,y

def generate_numbers(board):
    x, y, v = [], [], []
    
    for i in range(3):
        for j in range(3):
            if board[i, j] == Board.BOMB_VALUE:
                continue
            x.append(i)
            y.append(j)
            
            # Slicing the submatrix around (i, j)
            submatrix = board[max(0, i-1):min(3, i+2), max(0, j-1):min(3, j+2)]
            
            # Counting the bombs in the submatrix
            bomb_count = np.count_nonzero(submatrix == Board.BOMB_VALUE)
            
            v.append(bomb_count)
    
    # Update the board with the bomb counts
    for i, j, val in zip(x, y, v):
        board[i, j] = val
    
    return board


def create_state(board , mask , a,b):
    state = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if mask[i][j] == 0:
                state[i][j] = -1
            elif mask[i][j] == 1:
                if board[i][j] == Board.BOMB_VALUE:
                    state[i][j] = -2
                else:
                    state[i][j] = board[i][j]
            
    if(board[a][b] == Board.BOMB_VALUE):
        tag = 0
    else:
        tag = 1
    return state ,tag


def create_tags(states,a,b):
    y = []
    for index in range(len(states)):
        state = states[index]
        print(state)
        if(state[a][b] == Board.MARK_VALUE):
            y.append(MARK_TAG)
            print("bomb")
        else:
            y.append(REVEAL_TAG)
            print("safe")

    return y