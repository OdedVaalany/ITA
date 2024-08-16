from search import greedy_search
from search_problem import MinesweeperSearchProblem


class Agent():
    def __init__(self, board):
        self.board = board

    def run(self):
        raise NotImplementedError


class SearchAgent(Agent):
    def __init__(self, board):
        super().__init__(board)

    def run(self):
        return greedy_search(MinesweeperSearchProblem(self.board))


class DpllAgent(Agent):
    def __init__(self, board):
        super().__init__(board)

    def run(self):
        pass
        # bussiness logic
