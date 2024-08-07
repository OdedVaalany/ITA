import numpy as np
from board import *
from learningAgents import QLearningAgent
# Define the Q-learning agent class




# Define the game class
class Game:
    def __init__(self,gameObj):
        self.gameObj = gameObj
    def play(self , num_episodes=1000):
        # Initialize the Q-learning agent
        agent = QLearningAgent(getLegalActions=self.gameObj.get_leagal_action)

        # Start the game loop
        for episode in range(num_episodes):
            state = self.reset()  # Reset the game state
            done = False

            while not done:
                action = agent.getAction(state)  # Choose an action
                next_state , reward=game.step(state,action) # Take a step in the game
                agent.update(state, action, next_state, reward)  # Update the Q-table
                state = next_state

            # Print the total reward for the episode
            print(f"Episode {episode + 1}: Total reward = {reward}")

    def rewardFunction(self, state, action):
        if self.gameObj.is_bomb(state[0],state[1]) and action == 'reveal':
            return -10000
        if action == 'reveal':
            return 1
        else:
            return 0

    def reset(self):
        # Reset the game state
        self.gameObj.reset()

    def step(self, cell,action):
        # Take a step in the game and return the next state, reward, and done flag
        return self.gameObj.apply_action(cell ,action)
    
    def get_legal_actions(self ,state):
        return self.gameObj.get_legal_actions(state)


# Define the main function
if __name__ == "__main__":
    board = Board(8)
    game = Game(board)
    game.play()
