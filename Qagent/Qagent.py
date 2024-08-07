import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, height=3, width=3, mines=2):
        self.height = height
        self.width = width
        self.mines = mines
        self.board = np.zeros((height, width), dtype=int)
        self.revealed = np.full((height, width), False, dtype=bool)
        self.done = False
        self._place_mines()

    def _place_mines(self):
        mine_positions = random.sample(range(self.height * self.width), self.mines)
        for pos in mine_positions:
            x, y = divmod(pos, self.width)
            self.board[x][y] = -1
            for i in range(max(0, x-1), min(self.height, x+2)):
                for j in range(max(0, y-1), min(self.width, y+2)):
                    if self.board[i][j] != -1:
                        self.board[i][j] += 1

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.revealed = np.full((self.height, self.width), False, dtype=bool)
        self.done = False
        self._place_mines()
        return self.revealed.flatten()

    def step(self, action):
        if self.done:
            return self.revealed.flatten(), 0, True

        x, y = divmod(action, self.width)
        if self.revealed[x][y]:
            return self.revealed.flatten(), -1, False
        
        self.revealed[x][y] = True
        
        if self.board[x][y] == -1:
            self.done = True
            return self.revealed.flatten(), -10, True

        if self._check_win():
            self.done = True
            return self.revealed.flatten(), 10, True
        
        return self.revealed.flatten(), 1, False

    def _check_win(self):
        return np.all(self.revealed | (self.board == -1))

    def render(self):
        display_board = np.where(self.revealed, self.board, np.full(self.board.shape, 'X'))
        print(display_board)

class QAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_rate=0.95, exploration_rate=0.8, exploration_decay=0.999995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Use a dictionary to store Q-values

    def get_state_key(self, state):
        return tuple(state)  # Convert state to a hashable type

    def get_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_rate * self.q_table[next_state_key][best_next_action]
        td_delta = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_delta

    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

# Initialize the environment and agent
env = MinesweeperEnv()
state_size = env.height * env.width
action_size = state_size
agent = QAgent(state_size, action_size)

# Training the agent
episodes = 10000000
wins = 0
for e in range(episodes):
    state = env.reset()
    done = False
    moves = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        moves+=1
        if env._check_win():
            wins = wins + 1
    if e%100000 == 0:
        winrate = wins/100000 *100
        print(f"Episode {e} done")
        print(len(agent.q_table))
        print(agent.exploration_rate)
        print(moves)
        print(winrate)
        env.render()
        wins = 0
    # username = input("Enter username:")
    agent.decay_exploration_rate()

# Play a game with the trained agent
state = env.reset()
env.render()
done = False
while not done:
    action = agent.get_action(state)
    state, reward, done = env.step(action)
env.render()
