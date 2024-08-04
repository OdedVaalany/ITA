"""
How we will model the knowledge representation agent
given a 5x5 grid, we want to preduce a list of knowledge
that the agent can use to make decisions

N* represent an open cell with number
B represent a bomb
F represent a flagged cell
H represent a hidden cell

11N1 + 12F => 22N0


0 0 0 0 F
0 0 0 1 0
0 0 H 0 0
0 0 0 0 0
0 0 0 0 0

"""
from typing import List, Tuple, Set
from utils import EMPTY, FLAG


class Psokit:
    def __init__(self, p: Set[str]):
        self.pos = set([w for w in p if not w.startswith("~")])
        self.neg = set([w[1:] for w in p if w.startswith("~")])

    def __add__(self, other):
        pos = self.pos.difference(other.neg).union(
            other.pos.difference(self.neg))
        neg = self.neg.difference(other.pos).union(
            other.neg.difference(self.pos))
        return Psokit(pos.union(set([f"~{w}" for w in neg])))

    def __str__(self):
        return " + ".join(self.pos.union(set(["~" + w for w in self.neg])))

    def __hash__(self) -> int:
        return hash(str(self))

    def isEmpty(self):
        return len(self.pos) == 0 and len(self.neg) == 0

    def __eq__(self, value: object) -> bool:
        return self.pos == value.pos and self.neg == value.neg


class KR_agent:
    def __init__(self) -> None:
        # self.__KB: Set[Psokit] = set()
        # with open("KB.txt", 'r') as f:
        #     for line in f:
        #         self.__KB.add(Psokit(line.strip().split(',')))
        self.__World: Set[Psokit] = set()

    def __str__(self) -> str:
        return str(self.__KB)

    def categorize(self, val) -> str:
        if val == EMPTY:
            return "H"
        if val == FLAG:
            return "F"
        if type(val) == int:
            return "N"
        return val

    def procees(self, board: List[List[int]]) -> List[List[str]]:
        knowledge: Set[Psokit] = set()
        for i in range(1, 4):
            for j in range(1, 4):
                cell = board[i][j]
                if cell == EMPTY:
                    continue
                elif cell == FLAG:
                    knowledge.add(Psokit({f"{i}{j}F"}))
                elif cell > 0:
                    count = cell
                    for r in range(i-1, i+2):
                        for c in range(j-1, j+2):
                            if (r == 0 or c == 0 or r == 4 or c == 4) and board[r][c] == "F":
                                count -= 1
                            if r == 2 and c == 2:
                                continue
                    if count >= 0 and count <= 1:
                        knowledge.add(Psokit({f"{i}{j}N{count}"}))
                        knowledge = knowledge | self.get_cnfs((i, j))
                elif cell == 0:
                    knowledge.add(Psokit({f"{i}{j}N{0}"}))
        self.__World = knowledge
        return knowledge

    def get_cnfs(self, cell: Tuple[int, int]) -> Set[Psokit]:
        cnfs = []
        i, j = cell
        if i == 1 and j == 1:
            cnfs = [Psokit({'~11N1', '~12F', '22N0'}),
                    Psokit({'~11N1', '~21F', '22N0'})]
        elif i == 1 and j == 3:
            cnfs = [Psokit({'~13N1', '~12F', '22N0'}),
                    Psokit({'~13N1', '~31F', '22N0'})]
        elif i == 3 and j == 1:
            cnfs = [Psokit({'~31N1', '~21F', '22N0'}),
                    Psokit({'~31N1', '~32F', '22N0'})]
        elif i == 3 and j == 3:
            cnfs = [Psokit({'~33N1', '~32F', '22N0'}),
                    Psokit({'~33N1', '~23F', '22N0'})]
        elif i == 1 and j == 2:
            cnfs = [Psokit({'~12N1', '~11F', '22N0'}),
                    Psokit({'~12N1', '~13F', '22N0'})]
        elif i == 2 and j == 1:
            cnfs = [Psokit({'~21N1', '~11F', '22N0'}),
                    Psokit({'~21N1', '~31F', '22N0'})]
        elif i == 3 and j == 2:
            cnfs = [Psokit({'~32N1', '~31F', '22N0'}),
                    Psokit({'~32N1', '~33F', '22N0'})]
        elif i == 2 and j == 3:
            cnfs = [Psokit({'~23N1', '~13F', '22N0'}),
                    Psokit({'~23N1', '~33F', '22N0'})]
        return set(cnfs)

    def infer(self, alpha: str) -> bool:
        base = self.__World | {Psokit({f'~{alpha}'})}
        new_knowledge = []
        while True:
            to_pass = [w for w in base]
            n = []
            for i in range(len(to_pass)):
                for j in range(i+1, len(to_pass)):
                    n.append(to_pass[i] + to_pass[j])
                    if n[-1].isEmpty():
                        return True
            if set(n).issubset(base):
                return False
            print('new knowledge:')
            base = base.union(set(n))
