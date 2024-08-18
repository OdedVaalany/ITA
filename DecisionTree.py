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
        self.__single_mode = False
        for i in range(8):
            self.model[i] = tree.DecisionTreeClassifier()


    def special_train(self):
        index = 0
        for (a,b) in [(2,2),(2,1),(2,0),(1,2),(1,0),(0,2),(0,1),(0,0)]:
            states , tags = create_all_possible_states(a,b)
            processed_X , processed_y = change_confilicting_tags(states,tags)
            self.model[index].fit(processed_X,processed_y)
            index += 1

    def train(self, X, y):
        self.__single_model = True
        self.model[0].fit(X,y)

    def decision(self, state):
        if self.__single_model:
            return self.model[0].predict([state])[0]
        # new_state = expend_features([state])[0]
        state = state.reshape((5, 5))
        index = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == -10:
                    state[i][j] = -1
        for (x, y) in [(1,1),(1,2),(1,3),(2,1),(2,3),(3,1),(3,2),(3,3)]:
            if self.model[index].predict([self.get_3_by_3(state, x, y)]) == 1:
                print("model"   , index)
                print(self.get_3_by_3(state, x, y))

                return 1
            elif self.model[index].predict([self.get_3_by_3(state, x, y)]) == 0:
                print("model"   , index)
                print(self.get_3_by_3(state, x, y))
                return 0
            index += 1


        
        return 2

    def get_3_by_3(self , matrix, x, y):
        matrix = matrix.copy()
        return matrix[x-1:x+2,y-1:y+2].flatten()
    

    
    def __init__(self, board, max_depth=25):
        self.board = board
        self.max_depth = max_depth
        self.model = [None]*8
        for i in range(8):
            self.model[i] = RandomForestClassifier(max_depth=max_depth)


    def special_train(self):
        index = 0
        for (a,b) in [(2,2),(2,1),(2,0),(1,2),(1,0),(0,2),(0,1),(0,0)]:
            states , tags = create_all_possible_states(a,b)
            processed_X , processed_y = change_confilicting_tags(states,tags)
            self.model[index].fit(processed_X,processed_y)
            index += 1

    def train(self, X, y):
        
        # new_X  = expend_features(new_X)
        # print(X[0])
        # print(new_X[0])
        self.model[0].fit(X,y)

    def decision(self, state):
        # new_state = expend_features([state])[0]
        state = state.reshape((5, 5))
        index = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == -10:
                    state[i][j] = -1
        for (x, y) in [(1,1),(1,2),(1,3),(2,1),(2,3),(3,1),(3,2),(3,3)]:
            if self.model[index].predict([self.get_3_by_3(state, x, y)]) == 1:
                print("model"   , index)
                print(self.get_3_by_3(state, x, y))

                return 1
            elif self.model[index].predict([self.get_3_by_3(state, x, y)]) == 0:
                print("model"   , index)
                print(self.get_3_by_3(state, x, y))
                return 0
            index += 1


        
        return 2

    def get_3_by_3(self , matrix, x, y):
        matrix = matrix.copy()
        return matrix[x-1:x+2,y-1:y+2].flatten()

    
Boards = {"Beginer" : Board((9, 9) , 0.125),
              "Intermidate" : Board((16, 16) , 0.158),
              "Expert": Board((30, 16) , 0.207)}

    

if __name__ == "__main__":
    

    difficulty = "Beginer"
    board = Boards[difficulty]
    epoches = 100000
    dt = Decision_Tree(board,None)
    # dt = RandomForestAgent(board , 25 , 25)
    X = []
    y = []
    number_of_games = 100
    file_name = f"data.{epoches}.pkl"
    # data_size , tags_size = create_data_set(board , epoches , file_name)
    # print(data_size , tags_size)
    
    # print(len(states))
    
    with open(file_name, 'rb') as f:
        X, y = pickle.load(f)
    print(len(X) , len(y))




    winCounter = 0
   
    # dt.special_train()



    dt.train(X,y)
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
    


    