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

    def __init__(self, board, max_depth=25):
        self.board = board
        self.max_depth = max_depth
        self.model = [None]*8
        for i in range(8):
            self.model[i] = tree.DecisionTreeClassifier(max_depth=max_depth)

    def train(self, X, y):
        sub_trees_samples = [[] for _ in range(8)]
        sub_tree_tags = [[] for _ in range(8)]
        tag_index = 0
        for sample, tag in zip(X, y):
            sample = sample.reshape((5, 5))
            
            for (x, y) in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
                for index in range(8):
                    sub_trees_samples[index].append(self.get_3_by_3(sample, x, y))
                    sub_tree_tags[index].append(tag)
        for i in range(8):
            sub_trees_samples[i] , sub_tree_tags[i] = change_confilicting_tags(sub_trees_samples[i], sub_tree_tags[i])
        # new_X  = expend_features(new_X)
        # print(X[0])
        # print(new_X[0])
        for i in range(8):
            self.model[i].fit(sub_trees_samples[i],sub_tree_tags[i])

    def decision(self, state):
        # new_state = expend_features([state])[0]
        state = state.reshape((5, 5))
        for (x, y) in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
            for index in range(8):
                if self.model[index].predict([self.get_3_by_3(state, x, y)]) == 1:
                    return 1
                elif self.model[index].predict([self.get_3_by_3(state, x, y)]) == 0:
                    return 0
        return 2

    def get_3_by_3(self , matrix, x, y):
        matrix = matrix.copy()
        return matrix[x-1:x+2,y-1:y+2].flatten()
    

    
class RandomForestAgent:
    def __init__(self, board, n_estimators=25, max_depth=25):
        self.board = board
        self.max_depth = max_depth
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion='entropy', bootstrap = True)

    def train(self, X, y):
        new_X , new_y = change_confilicting_tags(X,y)
        new_X  = expend_features(new_X)
        self.model.fit(new_X,new_y)

    def decision(self, state):
        new_state = expend_features([state])
        return self.model.predict([new_state])

    
Boards = {"Beginer" : Board((9, 9) , 0.125),
              "Intermidate" : Board((16, 16) , 0.158),
              "Expert": Board((30, 16) , 0.207)}

    

if __name__ == "__main__":
    

    difficulty = "Intermidate"
    board = Boards[difficulty]
    epoches = 60000
    dt = Decision_Tree(board,25)
    # dt = RandomForestAgent(board , 25 , 25)
    X = []
    y = []
    number_of_games = 100
    file_name = f"data.{epoches}.pkl"
    data_size , tags_size = create_data_set(board , epoches , file_name)
    
    with open(file_name, 'rb') as f:
        X, y = pickle.load(f)
    winCounter = 0
    print(len(X) ,len(y))
    
    
    dt.train(X,y)
    # plt.figure()
    # tree.plot_tree(dt.model,filled=True)
    # plt.title("Decision tree trained on all the iris features")
    # plt.show()
    for game_number in range(number_of_games):
        print(game_number)
        board.reset()
        board.open_first()
        # board.print_current_board()
        win = agent_play(board, dt)
        if win:
            winCounter += 1
    print(f"Win rate: {winCounter}/{number_of_games}")

    with open('decision_tree_instance.pkl', 'wb') as f:
        pickle.dump(dt, f)
    


    