from sklearn import tree
from board import Board

import numpy as np
from itertools import product, combinations
from itertools import product
import copy
from decisionTreeUtils import *
import random
import pickle
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt




class Decision_Tree:
    def __init__(self, board, max_depth=100):
        self.board = board
        self.max_depth = max_depth
        self.model = tree.DecisionTreeClassifier(max_depth=max_depth )

    def train(self,X,y):
        self.model.fit(X, y)

    
    def decision(self, state):
        flattened_state = np.array(state).flatten()
        np_state = feature_vector(flattened_state)
        return self.model.predict([np_state])[0]

class RandomForestAgent:
    def __init__(self, board, n_estimators=25, max_depth=25):
        self.board = board
        self.max_depth = max_depth
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion='entropy', bootstrap = True)

    def train(self, X, y):
        self.model.fit(X, y)

    def decision(self, state):
        
        np_state = feature_vector(np.array(state).flatten())
        return self.model.predict([np_state])

    
Boards = {"Beginer" : Board((9, 9) , 0.125),
              "Intermidate" : Board((16, 16) , 0.158),
              "Expert": Board((30, 16) , 0.207)}

    

if __name__ == "__main__":
    

    difficulty = "Beginer"
    board = Boards[difficulty]
    epoches = 80000
    cleanData = True
   
    dt = Decision_Tree(board,max_depth=25)
    # dt = RandomForestAgent(board , 25 , 25)
    X = []
    y = []
    number_of_games = 1000
    file_name = f"data.extra.{cleanData}.{epoches}.pkl"
    data_size , tags_size = create_data_set(Boards , epoches , file_name)
    
    with open(file_name, 'rb') as f:
        X, y = pickle.load(f)
    winCounter = 0
    for i in y:
        if i == 2:
            print("unkowns")
    
    dt.train(X,y)
    

    # for difficulty in Boards.keys():
    board = Boards[difficulty]
    winCounter = 0
    for game_number in range(number_of_games):

        # board.print_current_board()
        
        board.reset()
        board.open_first()
        win = agent_play(board, dt)
        if win:
            winCounter += 1
        print(f"Win rate: {winCounter}/{number_of_games} , in diffculty {difficulty}")
    


    