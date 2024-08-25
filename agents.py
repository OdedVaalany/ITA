from search import greedy_search
from search_problem import MinesweeperSearchProblem
from game import UI
from dpll_solver import dpll


class Agent():
    def __init__(self, board, name):
        self.board = board
        self.__name__ = name

    def run(self):
        raise NotImplementedError

    def __str__(self):
        return self.__name__


class ManualAgent(Agent):
    def __init__(self, board):
        super().__init__(board, "manual")

    def run(self):
        ui = UI(self.board)
        ui.run()


class SearchAgent(Agent):
    def __init__(self, board):
        super().__init__(board, "search")

    def run(self):
        return greedy_search(MinesweeperSearchProblem(self.board))


class DpllAgent(Agent):
    def __init__(self, board):
        super().__init__(board, "dpll")
        self.__dpll = dpll(board)

    def run(self):
        return self.__dpll.run()
