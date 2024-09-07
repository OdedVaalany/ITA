import random
from time import sleep

import pygame

# Example file showing a basic pygame "game loop"
from board import Board
from utils import *
from time import time
from itertools import combinations, product
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
        self.start_time = time()
        self.running = True
        self.revealed = 0
        self.satisfied = [[False for _ in range(
            self.cols)] for _ in range(self.rows)]
        self.available_edges = [[0, i] for i in range(self.cols)] + \
                               [[self.rows - 1, i] for i in range(self.cols)] + \
                               [[i, 0] for i in range(self.rows)] + \
                               [[i, self.cols - 1]
                                   for i in range(1, self.rows)]

    def get_pure_literals(self, clauses):
        res = []
        for clause in clauses:
            if len(clause) == 1:
                res.append(list(clause)[0])
        return res

    def dp_solve(self, clauses, assignment, addition_to_assignment):
        if not clauses:
            # Base case: no more clauses to satisfy, return the current assignment
            return True, assignment
        # Base case: if any clause is empty, the assignment is unsatisfiable
        if any(not clause for clause in clauses):
            return False, []

        pure_literals = self.get_pure_literals(clauses)
        if pure_literals:
            new_clauses = self.simplify(clauses, pure_literals)
            for lit in pure_literals:
                if lit not in assignment:
                    addition_to_assignment.append(lit)
            assignment = assignment + addition_to_assignment
            sat, assi = self.dp_solve(
                new_clauses, assignment, addition_to_assignment)
            return sat, assi

        # nothing added by unary simplifying by one literal
        if addition_to_assignment == []:
            literals = self.choose_literal(clauses)
            new_clauses = self.simplify(clauses, literals)
            addition_to_assignment += literals
            assignment = assignment + addition_to_assignment
            return self.dp_solve(new_clauses, assignment, addition_to_assignment)

        return True, assignment

    def simplify(self, clauses, literals):
        new_clauses = set()
        for clause in clauses:
            new_clause = [l for l in clause if l not in literals]
            add_or_not = True
            for lit in literals:
                if -lit in new_clause:
                    add_or_not = False
            if new_clause and add_or_not:
                new_clauses.add(frozenset(new_clause))

        return new_clauses

    def cla1_contain_cla2_no_sign(self, cla1, cla2):
        for literal in cla2:
            if literal not in cla1 and -literal not in cla1:
                return False
        return True

    def simplify_by_clause_longer_than_one_lit(self, clauses):

        for clause in clauses:
            clause_contain_those = [cla for cla in clauses
                                    if self.cla1_contain_cla2_no_sign(clause, cla) if len(cla) != len(clause)]
            if len(clause_contain_those) == 0:
                continue
            new_clause = []
            for lit in clause:
                new_clause.append(lit)
            for cla in clause_contain_those:
                count_pos = 0
                for lit in cla:
                    if lit in new_clause:
                        new_clause.remove(lit)
                        if lit > 0:
                            count_pos += 1

                    if -lit in new_clause:  # not a good assignment.
                        new_clause.remove(-lit)
                        if lit < 0:
                            count_pos += 1

                if count_pos != sum(1 for lit in cla if lit > 0):
                    break

                pos = [lit for lit in new_clause if lit > 0]
                if len(pos) == len(new_clause) or len(pos) == 0:
                    if len(new_clause) == 0:
                        return None
                    return new_clause
        return None

    def get_corners_or_edges_available(self):
        avilable = self.board.avilable_states
        if [0, 0] in avilable:
            return [-1]
        elif [0, self.cols - 1] in avilable:
            return [-self.cols]
        elif [self.rows - 1, 0] in avilable:
            return [-((self.rows - 1) * self.cols + 1)]
        elif [self.rows - 1, self.cols - 1] in avilable:
            return [- self.cols * self.rows]
        for cell in self.available_edges:
            if cell in avilable:
                return [-self.cell_to_var(cell)]
            else:
                self.available_edges.remove(cell)
        return [-self.cell_to_var(random.choice(avilable))]

    def choose_literal(self, clauses):
        # check the option of multy simplifying to get some smart decision
        to_assign = self.simplify_by_clause_longer_than_one_lit(clauses)
        if to_assign != None:
            return to_assign
        return self.get_corners_or_edges_available()

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
                    self.hundle_click("open", k, j)
            else:
                k, j = self.var_to_cell(i)
                if not self.board.is_marked(k, j):
                    self.hundle_click("mark", k, j)
                    self.board.apply_action((k, j, "mark"))
                    # boards.append(self.board.copy())
        return [self.board.copy()]

    def run(self):
        boards = []
        cell = self.var_to_cell(-self.get_corners_or_edges_available()[0])
        self.board.apply_action((cell[0], cell[1], "reveal"))
        self.hundle_click("open", cell[0], cell[1])
        boards += self.run_game()
        return boards

    def run_game(self):
        res = []
        while len(self.board.avilable_states) > 0:
            # self.show()
            clauses = self.generate_cnf_clauses()
            if len(clauses) == 0:
                guess = self.get_corners_or_edges_available()
                assignments = guess
                satisfiable = True
            else:
                satisfiable, assignments = self.dp_solve(clauses, [], [])
            if satisfiable:
                board = self.apply_assignment(assignments)
                this_apply = board[0]
                res += board
                if this_apply.is_failed() or this_apply.is_solved():
                    if this_apply.is_solved():
                        for cell in self.board.avilable_states:
                            self.board.apply_action(
                                (cell[0], cell[1], "reveal"))
                            self.hundle_click("open", cell[0], cell[1])
                        res += [self.board]
                    return res
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
