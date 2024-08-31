# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
import random


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def a_star_search(problem, heuristic=lambda x, y: 0):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    open_set = util.PriorityQueue()
    inOpenSet = {}
    cameFrom = {}

    gScore = {}
    gScore[problem.get_start_state()] = 0

    fScore = {}
    fScore[problem.get_start_state()] = heuristic(
        problem.get_start_state(), problem)

    open_set.push(problem.get_start_state(), fScore[problem.get_start_state()])
    inOpenSet[problem.get_start_state()] = True
    while not open_set.isEmpty():
        current = open_set.pop()
        inOpenSet[current] = None
        if problem.is_goal_state(current):
            total_path = util.Queue()
            p = current
            while cameFrom.get(p) is not None:
                current, action = cameFrom[current]
                total_path.push(action)
                p = current
            return total_path.list
        for successor, action, stepCost in problem.get_successors(current):
            tentative_gScore = gScore[current] + stepCost
            if tentative_gScore < gScore.get(successor, float('inf')):
                cameFrom[successor] = (current, action)
                gScore[successor] = tentative_gScore
                fScore[successor] = gScore[successor] + \
                    heuristic(successor, problem)
                if inOpenSet.get(successor) is None:
                    open_set.push(successor, fScore[successor])
    return []


def greedy_search(problem):
    states = []
    state = problem.get_start_state()
    for i in range(state.size[0]*state.size[1]):
        states.append(state)
        if problem.is_goal_state(state):
            break
        next_actions = problem.get_successors(state)
        if len(next_actions) == 0:
            break
        successor = min(next_actions,
                        key=lambda x: x[2])
        if successor[2] > 1:
            best_actions = [
                action for action in next_actions if action[2] == successor[2]]
            state = random.choice(best_actions)[0]
        else:
            state = successor[0]
    return states


def random_min(seq, key=lambda x: x):
    min_val = float('inf')
    min_elem = None
    for elem in seq:
        val = key(elem)
        if val < min_val:
            min_val = val
            min_elem = elem
        elif val == min_val:
            if util.flip_coin(0.5):
                min_val = val
                min_elem = elem
    return min_elem
