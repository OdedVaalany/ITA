from board import Board
import random
import pickle
import numpy as np
from DecisionTree import Boards
import concurrent.futures

def random_choice(board):
    done = True
    while(done):
        cell = (random.randint(0, board.size[0]-1), random.randint(0, board.size[1]-1))
        if not board.is_revealed(cell[0],cell[1]):
            return cell
    return None

def make_one_move(board, agent):
    ##make move if pass the filter (3 revealed cells around the middle cell) or random cell
    for row in range((board.size[0])):
        for col in range((board.size[1])):
            cell = (row, col)
            if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
                continue
            state = board.kernel_n(5,cell)
            if filter_states(state):
                state = feature_vector(state)
                prediction = agent.decision(state)
                if prediction == 1:
                    board.apply_action(cell, "reveal")
                    if board.is_bomb(cell[0],cell[1]):
                        done = True
                        return done
                    else:
                        done = False
                        return done
                elif prediction == 0:
                    board.apply_action(cell, "mark")
                    done = False
                    return done
    
    random_cell = random_choice(board)
    print("random" ,random_cell)
    board.apply_action(random_cell, "reveal")
    if(board.is_bomb(random_cell[0],random_cell[1])):
        done = True
    else:
        done = False
    return done

def agent_play(board , agent):
    done = False
    move_counter = 0
    while not done:
        # board.print_current_board()
        make_one_move(board, agent)
        move_counter += 1
        if(board.is_solved() or board.is_failed()):
            done = True
        if(board.is_solved()):
            return True
    print(f"Game Over! Total moves: {move_counter}")
    return False


def filter_states(state):
        size = len(state)
        middle = size//2
        count = 0
        if (state[middle][middle] !=  Board.HIDDEN_VALUE):
            return False
        for i in range(-1,2):
            for j in range(-1,2):
                if i ==j and i == 0:
                    continue
                if state[middle+i][middle+j] != Board.HIDDEN_VALUE:
                    count+=1
                    if count > 0:
                        return True
        
        return False


def feature_vector(state):
    return np.array(state).flatten()
        
def random_correct(board):
    done = True
    while(done):
        cell = (random.randint(0, board.size[0]-1), random.randint(0, board.size[1]-1))
        if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
            continue
        if board.is_bomb(cell[0],cell[1]):
            board.apply_action((cell[0],cell[1]), "mark")
            return cell, 0
        else:
            board.apply_action((cell[0],cell[1]), "reveal")
            return cell, 1
    
    return None, None


def make_action_for_Training(board ):
        tag = None
        available_states = board.available_states
        random.shuffle(available_states)
        for cell in available_states:
            cell = (cell[0], cell[1])
            if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
                continue
            state = board.kernel_n(5,cell)
            if filter_states(state):
                actions = board.get_actions(cell)
                if("reveal" in actions):
                    np_state = feature_vector(state)
                    if board.is_bomb(cell[0],cell[1]):
                        board.apply_action(cell, "mark")
                        tag =  0
                        add_to_data = True
                        return np_state, tag , add_to_data 
                    else:
                        board.apply_action(cell, "reveal")
                        tag = 1
                        add_to_data = True
                        return np_state, tag , add_to_data
        
        #make random move that doesnt pick bomb if no state passed the filter
        
        if board.is_solved() or board.is_failed():
            return None, None, False      
        cell,tag = random_correct(board)
        # print("random correct")
    
        np_state = np.array(board.kernel_n(5,cell)).flatten()
        add_to_data = False
        return np_state, tag , add_to_data



def run_episode(episode, Boards):
    X = []
    y = []
    
    # Pick a random board
    random_key = random.choice(list(Boards.keys()))
    board = Boards[random_key]
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
    
    return X, y

def run_parallel_episodes(num_episodes, Boards, file_name):
    X = []
    y = []
    
    # Use ProcessPoolExecutor to run episodes in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_episodes):
            if i % 1000 == 0:
                print("crated" ,)
                # print(f"Processed {i + 1}/{num_episodes} episodes")
                # print(f"Data size: {len(X)} samples")
            futures.append(executor.submit(run_episode, i, Boards) )
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            episode_X, episode_y = future.result()
            X.extend(episode_X)
            y.extend(episode_y)
            if( i % 1000 ==0):
                print(f"Processed {i + 1}/{num_episodes} episodes")
                print(f"Data size: {len(X)} samples")
            
            
    
    with open(file_name, 'wb') as f:
        pickle.dump((X, y), f)
    
    return len(X), len(y)

def create_data_set(boards , num_episodes,file_name):
    return run_parallel_episodes(num_episodes , boards , file_name)
