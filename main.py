import argparse
from agents import SearchAgent, DpllAgent, ManualAgent
from game import ShowUI, UI
from board import Board
from typing import *
import os
import time

parser = argparse.ArgumentParser(
    prog="MineSweeper", description="A simple minesweeper game", epilog="Thanks for playing!")
parser.add_argument("--level", "-l", type=str, default="easy",
                    help="The level of the game (easy, medium, hard)")
parser.add_argument("--ui", "-u", action="store_true",
                    help="Run the game with a GUI")
parser.add_argument("--search", "-s", action="store_true",
                    help="Run the game with a search algorithm")
parser.add_argument("--knowledge", "-k", action="store_true",
                    help="Run the game with knowledge representation")
parser.add_argument("--decision", "-d", action="store_true",
                    help="Run the game with decision tree")
parser.add_argument("--manual", "-m", action="store_true",
                    help="Run the game with humen interaction")
parser.add_argument("--num_of_games", "-n", type=int,
                    default=1, help="Number of games to play")
parser.add_argument("--epoch", "-e", type=int, default=100,
                    help="The time for state in ms, when using not humen interaction and allow ui to be True")
parser.add_argument("--output", "-o", type=str, default="output",
                    help="The output folder to save the result")


def get_agent(agent_type: Literal["search", "knowledge", "decision", "manual"]):
    if agent_type == "search":
        return SearchAgent
    if agent_type == "knowledge":
        return DpllAgent
    if agent_type == "decision":
        return DpllAgent
    return ManualAgent


def get_board(level: Literal["easy", "medium", "hard"]):
    if level == "easy":
        return Board((10, 10), 10)
    if level == "medium":
        return Board((16, 16), 40)
    return Board((16, 30), 99)


def check_folder(path: str):
    import os
    return os.path.exists(path) and os.path.isdir(path)


if __name__ == "__main__":
    args = parser.parse_args()

    # First validate the inputs
    if args.manual and not args.ui:
        raise ValueError("Manul agent must be used with ui")

    if args.manual and (args.search or args.knowledge or args.decision):
        raise ValueError("Manul agent cannot be used with other agents")

    agents = []
    if args.search:
        agents.append(get_agent("search"))
    if args.knowledge:
        agents.append(get_agent("knowledge"))
    if args.decision:
        agents.append(get_agent("decision"))

    if len(agents) > 1 and args.ui:
        raise ValueError("Cannot have multiple agents with ui")

    if len(agents) > 1 and args.ui and args.num_of_games > 1:
        raise ValueError(
            "Cannot have multiple agents with ui and multiple games")

    if len(agents) == 0 and not args.manual:
        raise ValueError("At least one agent must be selected")

    if args.output and not check_folder(args.output):
        raise ValueError("Output folder does not exist")

    if args.epoch < 0:
        raise ValueError("Epoch must be positive")

    if args.num_of_games < 0:
        raise ValueError("Number of games must be positive")

    if args.ui:
        if args.num_of_games > 1:
            raise ValueError("Cannot have multiple games with ui")

    if args.level not in ["easy", "medium", "hard"]:
        raise ValueError("Invalid level")

    # Now we can start the game

    if args.ui and args.manual:
        agent = ManualAgent(get_board(args.level))
        agent.run()
    elif args.ui:
        agent = agents[0](get_board(args.level))
        states = agent.run()
        ShowUI(states, args.epoch).run()
    else:
        with open(os.path.join(args.output, 'logs.txt'), "w") as f:
            f.write("Game,Level,Agent,Result,Num of steps,Time\n")
            for i in range(args.num_of_games):
                board = get_board(args.level)
                board.reset()
                for ag in agents:
                    agent = ag(board)
                    start_time = time.time()
                    states = agent.run()
                    delta = time.time() - start_time
                    result = "Success" if states[-1].is_solved() else "Failed"
                    f.write(f"{i},{args.level},{agent},{
                            result},{len(states)},{delta}\n")
                    f.flush()
