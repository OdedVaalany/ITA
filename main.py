import argparse
from agents import SearchAgent, DpllAgent
from game import ShowAgent, UI
from board import Board

parser = argparse.ArgumentParser(
    prog="MineSweeper", description="A simple minesweeper game", epilog="Thanks for playing!")
parser.add_argument("--level", "-l", type=str, default="easy",
                    help="The level of the game (easy, medium, hard)")
parser.add_argument("--ui", "-u", type=bool, default=False,
                    help="Run the game with a GUI")
parser.add_argument("--search", "-s", type=bool, default=False,
                    help="Run the game with a search algorithm")
parser.add_argument("--knowledge", "-k", type=bool, default=False,
                    help="Run the game with knowledge representation")
parser.add_argument("--decision", "-d", type=bool,
                    default=False, help="Run the game with decision tree")
parser.add_argument("--manual", "-m", type=bool, default=False,
                    help="Run the game with humen interaction")
parser.add_argument("--num_of_games", "-n", type=int,
                    default=1, help="Number of games to play")
parser.add_argument("--epoch", "-e", type=int, default=100,
                    help="The time for state in ms, when using not humen interaction and allow ui to be True")


def run_game():
    board = Board(10, 10, 10)
    agent = SearchAgent(board)
    states = agent.run()
    ui = ShowAgent(states)
    ui.run()


if __name__ == "__main__":
    args = parser.parse_args()

    print(args.epoch)
    # board = Board(10, 10, 10)
    # ui = UI(board)
    # ui.run()
    # pygame.quit()
    # print("Goodbye!")
