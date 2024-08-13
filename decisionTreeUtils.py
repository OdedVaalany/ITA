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
                prediction = agent.decision(state)
                print("prediction" ,prediction)
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
        done = make_one_move(board, agent)
        if(board.win()):
            return True
        move_counter += 1
        board.print_current_board()
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
                if state[middle+i][middle+j] != Board.HIDDEN_VALUE and state[middle+i][middle+j] != Board.OUT_OF_BOUND_VALUE:
                    count+=1
                    if count > 0:
                        return True
        
        return False



        
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
        for cell in board.avilable_states:
            cell = (cell[0], cell[1])
            if board.is_revealed(cell[0],cell[1]) or board.is_marked(cell[0],cell[1]):
                continue
            state = board.kernel_n(5,cell)
            if filter_states(state):
                actions = board.get_actions(cell)
                if("reveal" in actions):
                    np_state = feature_vector(np.array(state).flatten())
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
    
        if board.win() or board.lose():
            return None, None, False      
        cell,tag = random_correct(board)
        
    
        np_state = feature_vector(np.array(board.kernel_n(5,cell)).flatten())
        add_to_data = False
        return np_state, tag , add_to_data


def feature_vector(state):
    return state.flatten()
    feature_vector = np.zeros((len(state), 12))
    for i in range(len(state)):
        if(state[i] == Board.OUT_OF_BOUND_VALUE):
            feature_vector[i][11] = 1
        elif(state[i] == Board.HIDDEN_VALUE):
            feature_vector[i][9] = 1
        elif(state[i] == Board.MARK_VALUE):
            feature_vector[i][10] = 1
        else:
            feature_vector[i][state[i]] = 1
        
    return feature_vector.flatten()



def run_episode(episode, Boards):
    X = []
    y = []
    
    # Pick a random board
    random_key = random.choice(list(Boards.keys()))
    board = Boards[random_key]
    board = Board(board.size, board.bomb_density)
    board.reset()
    board.open_first()
    done = False
    
    while not done:
        np_state, tag, add_to_data = make_action_for_Training(board)
        
        # Add to the data only if the state is not in the map or the tag is different
        if add_to_data:
            X.append(np_state)
            y.append(tag)
        
        if board.win() or board.lose():
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
                print("crated" ,i)
                # print(f"Processed {i + 1}/{num_episodes} episodes")
                # print(f"Data size: {len(X)} samples")
            futures.append(executor.submit(run_episode, i, Boards) )
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            episode_X, episode_y = future.result()
            X.extend(episode_X)
            y.extend(episode_y)
            if( i % 100 ==0):
                print(f"Processed {i + 1}/{num_episodes} episodes")
                print(f"Data size: {len(X)} samples")
            
    # import pandas as pd
    print(len(X) ,len(y))  
    X_updated,  y_updated =process_data(X, y)

    with open(file_name, 'wb') as f:
        pickle.dump((X_updated, y_updated), f)
    
    return len(X_updated), len(y_updated)

def process_data(X, y):
    import pandas as pd
    # Convert X and y into a pandas DataFrame
    df = pd.DataFrame(X)
    df['tag'] = y

    # Group by all feature columns and count the number of unique tags for each group
    unique_tag_count = df.groupby(list(df.columns[:-1]))['tag'].transform('nunique')

    # Create a mask where the number of unique tags is greater than 1 (indicating conflicting tags)
    conflict_mask = unique_tag_count > 1
    
    # Update tags for conflicting samples to 2
    df.loc[conflict_mask, 'tag'] = 2

    # Remove duplicates with the same tag (keeping only the first occurrence)
    df = df.drop_duplicates(subset=list(df.columns[:-1]), keep='first')

    # Extract the updated X and y
    X_updated = df.drop('tag', axis=1).values
    y_updated = df['tag'].values

    return X_updated, y_updated

# def process_data(X, y):
#     import pandas as pd
#     # Convert X and y into a pandas DataFrame
#     df = pd.DataFrame(X)
#     df['tag'] = y

#     # Find duplicates with different tags
#     tag_counts = df.groupby(df.columns.tolist()).size()
#     conflicting_tags = tag_counts[tag_counts > 1].index
    
#     # Create a mask for rows with conflicting tags
#     conflict_mask = df.duplicated(keep=False) & df.groupby(list(df.columns[:-1]))['tag'].transform('nunique') > 1
    
#     # Update tags for conflicting samples to 0
#     df.loc[conflict_mask, 'tag'] = 2

#     # Remove duplicates with the same tag
#     # Drop duplicates, keeping only the first occurrence
#     df = df.drop_duplicates(subset=df.columns.tolist(), keep='first')

#     # Extract the updated X and y
#     X_updated = df.drop('tag', axis=1).values
#     y_updated = df['tag'].values

#     return X_updated, y_updated


def create_data_set(boards , num_episodes,file_name):
    return run_parallel_episodes(num_episodes , boards , file_name)
    