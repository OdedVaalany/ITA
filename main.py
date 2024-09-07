import argparse
from agents import SearchAgent, DpllAgent, ManualAgent, Agent
from game import ShowUI
from board import Board
from typing import *
import os
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
parser.add_argument("--output", "-o", type=str, default=None,
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
        return Board((9, 9), 10)
    if level == "medium":
        return Board((16, 16), 40)
    return Board((16, 30), 99)


def check_folder(path: str):
    import os
    return os.path.exists(path) and os.path.isdir(path)


def simulate_single_run(level: str, ag: Agent):
    board = get_board(level)
    agent = ag(board)
    start_time = time.time()
    states = agent.run()
    delta = time.time() - start_time
    result = 0 if states[-1].is_solved() else (
        1 if states[-1].is_failed() else 2)
    return level, result, len(states), delta


def simulate_multirun(args: argparse.Namespace):
    counter = [0, 0, 0]  # Success, Failed, Unknown
    time_counter = [0, 0, 0]  # Success, Failed, Unknown
    steps_size = [0, 0, 0]  # Success, Failed, Unknown
    result_map = ["Success", "Failed", "Unknown"]
    with open(os.path.join(args.output, f'logs_{ag.__name__}_level_{args.level}.txt'), "w") as f:
        f.write("Game,Level,Result,Num of steps,Time\n")
        with ProcessPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(simulate_single_run, args.level, ag)
                       for i in range(args.num_of_games)]
            for i, future in tqdm(enumerate(as_completed(futures)), total=args.num_of_games):
                level, result, steps, time = future.result()
                f.write(f"{i},{level},{result},{steps},{time}\n")
                f.flush()
                counter[result] += 1
                steps_size[result] += steps
                time_counter[result] += time
    print(f"Agent {ag.__name__} for level {args.level}")
    print('\n'*1)
    print("{:10} | {:8} | {:10} | {:10} | {:10} | {:10}".format(
        "Achivement", "total", "total time", "total steps", "avg time", "avg steps"))
    print('-'*100)
    print("{:10} | {:8} | {:10.4f} | {:10} | {:10.4f} | {:10.0f}".format(
        "Won", counter[0], time_counter[0], steps_size[0], 0 if counter[0] == 0 else time_counter[0]/counter[0], 0 if counter[0] == 0 else steps_size[0]/counter[0]))
    print("{:10} | {:8} | {:10.4f} | {:10} | {:10.4f} | {:10.0f}".format(
        "Fail", counter[1], time_counter[1], steps_size[1], 0 if counter[1] == 0 else time_counter[1]/counter[1], 0 if counter[1] == 0 else steps_size[1]/counter[1]))
    print("{:10} | {:8} | {:10.4f} | {:10} | {:10.4f} | {:10.0f}".format(
        "No Op", counter[2], time_counter[2], steps_size[2], 0 if counter[2] == 0 else time_counter[2]/counter[2], 0 if counter[2] == 0 else steps_size[2]/counter[2]))


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
        print("Output folder does not exist")
        exit(0)

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
        if not args.output:
            args.output = os.path.dirname(__file__)
        for ag in agents:
            simulate_multirun(args)
