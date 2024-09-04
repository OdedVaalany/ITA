import random
from time import sleep

import pygame

# Example file showing a basic pygame "game loop"
from board import Board
from util import Counter
from utils import *
from time import time
from itertools import combinations
from collections import defaultdict


def format_time(number: float) -> str:
    return f"{number // 60:2.0f}:{number % 60:2.0f}"


class dpll():

    def __init__(self, board):
        self.rows = board.size[0]
        self.cols = board.size[1]
        self.clauses = set()

        self.board = board
        self.height, self.width = board.size
        # self.start_time = time()
        # self.running = True
        # self.revealed = 0
        # self.satisfied = [[False for _ in range(
        #     self.cols)] for _ in range(self.rows)]

        self.available_edges = [[0, i] for i in range(self.cols)] + \
                              [[self.rows - 1, i] for i in range(self.cols)] + \
                              [[i, 0] for i in range(self.rows)] + \
                              [[i, self.cols - 1] for i in range(1, self.rows)]

        # self.screen = pygame.display.set_mode(
        #     (self.width * BLOCK_SIZE, self.height * BLOCK_SIZE + 50))
        # self.clock = pygame.time.Clock()
        # self.start_time = time()
        # pygame.font.init()
        # self.font = pygame.font.Font(None, 30)
        self.running = True
        self.revealed = 0
        self.satisfied = [[False for _ in range(self.cols)] for _ in range(self.rows)]

        # print(self.available_edges)

    def get_pure_literals(self, clauses):
        res = []
        for clause in clauses:
            if len(clause) == 1:

                res.append(list(clause)[0])
        return res

    def dp_solve(self, clauses, assignment, addition_to_assignment):
        # print("assignment", assignment)
        if not clauses:
            # Base case: no more clauses to satisfy, return the current assignment
            # print(type(assignment))

            return True, assignment
        # Base case: if any clause is empty, the assignment is unsatisfiable
        if any(not clause for clause in clauses):
            return False, []

        pure_literals = self.get_pure_literals(clauses)

        if pure_literals:
            literal = pure_literals.pop()
            new_clauses = self.simplify(clauses, literal)
            addition_to_assignment = [literal]
            assignment = assignment + addition_to_assignment
            sat, assi = self.dp_solve(
                new_clauses, assignment, addition_to_assignment)

            return sat, assi

        # nothing added by unary simplifying by one literal
        if addition_to_assignment == []:
            literal = self.choose_literal(clauses)
            # print("literal ", literal)
            new_clauses = self.simplify(clauses, literal)
            addition_to_assignment = [literal]
            assignment = assignment + addition_to_assignment
            return self.dp_solve(new_clauses, assignment, addition_to_assignment)

        return True, assignment

    def simplify(self, clauses, literal):
        new_clauses = []
        for clause in clauses:
            if literal in clause:
                new_clause = [l for l in clause if l != literal]
                if new_clause:
                    new_clauses.append(frozenset(new_clause))

            elif -literal in clause:
                continue  # Clause not satisfied
            else:
                new_clauses.append(clause)

        return new_clauses

    def clause1_in_clause2(self, cla1, cla2):
        for lit in cla1:
            if lit not in cla2:
                return False
        return True

    def do_binary_simply(self, clauses):
        # first try clauses of len 2:
        for clause in clauses:
            if len(clause) == 2:
                opposite = frozenset([-lit for lit in clause])
                if opposite in clauses:
                    for other in clauses:
                        if other == clause or other == opposite:
                            continue
                        if self.clause1_in_clause2(clause, other):
                            to_check = frozenset(
                                [lit for lit in other if lit not in clause] + [lit for lit in opposite])
                            if to_check in clauses:
                                return [lit for lit in other if lit not in clause]
                        if self.clause1_in_clause2(opposite, other):
                            to_check = frozenset(
                                [lit for lit in other if lit not in opposite] + [lit for lit in clause])
                            if to_check in clauses:
                                return [lit for lit in other if lit not in opposite]
        return None

    def smart_choice(self, clauses):
        for clause in clauses:
            positive_literals = [literal for literal in clause if literal > 0]
            negative_literals = [literal for literal in clause if literal < 0]

            # Check what will happen if we decide that the positive literal it is a bomb.
            if len(positive_literals) == 1 and len(negative_literals) == len(clause) - 1:
                # print("!positive_literals", positive_literals)
                # print("!negative_literals", negative_literals)

                reduction = [lit for lit in clause if lit not in positive_literals] # (-31, -32, -33)
                for other in clauses:
                    if other == clause:
                        continue
                    negative_other_lit = [literal for literal in other if literal < 0]
                    to_pass = False
                    for lit in reduction:
                        if lit not in other and -lit not in other:
                            to_pass = True
                            continue
                    if not to_pass and (len(negative_other_lit) < len(reduction)):
                        return - positive_literals[0]  # cannot be a bomb because cause collision
            elif len(negative_literals) == 1 and len(positive_literals) == len(clause) - 1:
                # print("positive_literals", positive_literals)
                # print("negative_literals", negative_literals)
                # check if assign the single literal , leaving us with valid clauses.
                reduction = [lit for lit in clause if lit not in negative_literals]
                for other in clauses:
                    if other == clause:
                        continue
                    positive_other_lit = [literal for literal in other if literal > 0]
                    to_pass = False
                    for lit in reduction:
                        if lit not in other and -lit not in other:
                            to_pass = True
                            continue
                        if not to_pass and (len(positive_other_lit) < len(reduction)):
                            return - negative_literals[0]  # cannot open because because collision soo its a bomb

        return None


    def simplify_by_clause_longer_than_one_lit(self, clauses):
        clause_to_assign = self.do_binary_simply(clauses)
        if clause_to_assign != None:
            for lit in clause_to_assign:
                return lit
        else:
            to_assign = self.smart_choice(clauses)
            if to_assign != None:
                return to_assign
        return None

    def choose_literal(self, clauses):
        # check the option of multy simplifying to get some smart decision
        l = self.simplify_by_clause_longer_than_one_lit(clauses)
        if l != None:
            return l
        # need to guess

        # try corner first
        avilable = self.board.avilable_states
        if [0, 0] in avilable:
            return -1
        elif [0, self.cols - 1] in avilable:
            return -self.cols
        elif [self.rows - 1, 0] in avilable:
            return -((self.rows - 1) * self.cols + 1)
        elif [self.rows - 1, self.cols - 1] in avilable:
            return - self.cols * self.rows
        for cell in self.available_edges:
            if cell in avilable:
                return -self.cell_to_var(cell)
            else:
                self.available_edges.remove(cell)
        return -self.cell_to_var(random.choice(avilable))
        # for clause in clauses:
        #     for literal in clause:
        #         if literal < 0:
        #             return literal

    def guess(self):
        # random choice that can have an heuristic to a smarter choice.
        # unrevealed_cells = self.board.avilable_states
        # print(unrevealed_cells)
        available = self.board.avilable_states


        # print(self.available_edges)
        for cell in self.available_edges:
            # print("!", cell, "#")
            if cell in available:
                # print((cell[0], cell[1]))
                # print("cell", cell, "end")
                self.available_edges.remove(cell)
                return (cell[0] , cell[1])
        #
        if len(available) == 0:
            return None  # No cells to reveal

            # For simplicity, just return a random choice from unrevealed cells
        res = random.choice(available)
        return (res[0], res[1])

    def hundle_click(self, event_name, x, y):
        click_func = self.board.reveal
        if event_name == "mark":
            click_func = self.board.mark
        if x < self.rows and y < self.cols and y >= 0 and x >= 0:
            click_func(x, y)

    def apply_assignment(self, assignment):
        boards = []
        for i in assignment:
            if i < 0:  # not bomb
                k, j = self.var_to_cell(-i)
                if not self.board.is_revealed(k, j):
                    self.board.apply_action((k, j, "reveal"))
                    # boards.append(self.board.copy())
                    self.hundle_click("open", k, j)

                    # self.board.apply_action((k, j, "open"))
                    # if self.board.is_failed() or self.board.is_solved():
                    #     # print("is a bomb in apply assignment", k, j)
                    #     # self.hundle_click("open", k, j)
                    #
                    #     return [self.board.copy()]
            else:
                k, j = self.var_to_cell(i)
                if not self.board.is_marked(k, j):
                    self.hundle_click("mark", k, j)
                    self.board.apply_action((k, j, "mark"))
                    # boards.append(self.board.copy())
        return [self.board.copy()]

    def run(self):
        boards = []
        need_to_guess = True
        cell = self.guess()
            
        # in the first version we are going to guess just cells to open.
        # next we will add marking.
        # e = len(self.board.avilable_states)

        self.hundle_click("open", cell[0], cell[1])
        self.board.apply_action((cell[0], cell[1], "reveal"))
        boards += self.run_iteration()

        return boards

    def run_iteration(self):
        res = []
        e = len(self.board.avilable_states)

        while len(self.board.avilable_states) > 0:
            # sleep(1.5)
            # self.show()
            # self.show()
            e = len(self.board.avilable_states)

            clauses = self.generate_cnf_clauses()
            if len(clauses) == 0:
                assignment =[]
                for cell in self.board.avilable_states:
                    assignment.append(-self.cell_to_var(cell))
                board = self.apply_assignment(assignment)
                res += board
                this_apply = board[0]
                if this_apply.is_failed() or this_apply.is_solved():
                    # self.show()
                    # sleep(150)
                    return res
            satisfiable, assignments = self.dp_solve(clauses, [], [])
            if satisfiable:
                # print("SATISFIABLE: ", assignments)
                board = self.apply_assignment(assignments)

                this_apply = board[0]
                res += board
                if this_apply.is_failed() or this_apply.is_solved():
                    # self.show()
                    # sleep(150)
                    return res


            # else:
            #     pass
            # print("UNSATISFIABLE")
            # res.append(self.board.copy())
        return res


    def get_neighbors(self, i, j):
        return [(i + x, j + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if
                (0 <= i + x < self.rows) and (0 <= j + y < self.cols) and (x != 0 or y != 0)]

    def generate_cnf_clauses(self):
        self.clauses = set()
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.board.is_revealed(i, j):
                    continue
                if self.satisfied[i][j]:
                    continue
                mine_count = int(self.board.__getitem__((i, j)))
                if mine_count == 0:
                    self.satisfied[i][j] = True
                    continue
                # print(mine_count)
                neighbors = self.get_neighbors(i, j)
                avalable_neighbors, marked_neighbors, unmarked_neighbors = [], [], []
                for n in neighbors:
                    reveald = self.board.is_revealed(*n)
                    marked = self.board.is_marked(*n)
                    if not reveald and not marked:
                        avalable_neighbors.append(n)
                    elif marked:
                        marked_neighbors.append(n)
                marked_count = len(marked_neighbors)
                remaining_mines = mine_count - marked_count
                num_avalable = len(avalable_neighbors)

                if num_avalable == 0:
                    self.satisfied[i][j] = True
                    continue

                # No mines in any remaining unmarked neighboring cells
                if remaining_mines == 0:
                    self.satisfied[i][j] = True
                    for cell in avalable_neighbors:
                        # self.satisfied[cell[0]][cell[1]] = True
                        self.clauses.add(frozenset([-self.cell_to_var(cell)]))
                        # self.board.apply_action((cell[0], cell[1], "reveal"))
                        # self.hundle_click("reveal", cell[0], cell[1])

                    continue
                # All unmarked neighbors must be mines
                if remaining_mines == num_avalable:
                    for cell in avalable_neighbors:
                        # self.satisfied[i][j] = True
                        self.clauses.add(frozenset([self.cell_to_var(cell)]))
                        # self.board.apply_action((cell[0], cell[1], "mark"))

                        self.hundle_click("mark", cell[0], cell[1])
                        # self.board.apply_action(cell, "mark")
                    continue
                # At least 'remaining_mines' mines in unmarked neighboring cells
                if 1 <= remaining_mines <= num_avalable:
                    for comb in combinations(avalable_neighbors, remaining_mines):
                        at_least_clause = [self.cell_to_var(c) for c in comb]
                        for other in avalable_neighbors:
                            if other not in comb:
                                at_least_clause.append(-self.cell_to_var(other))
                        self.clauses.add(frozenset(at_least_clause))
        return self.clauses

    def cell_to_var(self, cell):
        i, j = cell
        return i * self.cols + j + 1

    def var_to_cell(self, var):
        i = var // self.cols
        j = var % self.cols
        j -= 1  # -1 because 0 cannot be classified to - or +
        if j == -1:
            i -= 1
            j = self.cols - 1
        return i, j
    def draw_blocks(self):
        for i in range(self.height):
            for j in range(self.width):
                imp = pygame.image.load(
                    ASSETS_IMAGES[self.board[i, j]]).convert()
                self.screen.blit(imp, (j * BLOCK_SIZE, i * BLOCK_SIZE))

    def draw_text(self):
        text = self.font.render(
            f"Flags: {self.board.num_of_markers} \t Time: {format_time(time() - self.start_time)}", True, "black")
        self.screen.blit(text, (0, self.height * BLOCK_SIZE + 10))

    def show(self):
        self.screen.fill("white")
        self.draw_blocks()
        self.draw_text()
        pygame.display.flip()


if __name__ == "__main__":
    board = Board((16, 16))
    dpll_agent = dpll(board)
    res = dpll_agent.run()