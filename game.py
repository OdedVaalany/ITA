# Example file showing a basic pygame "game loop"
import pygame
from board import Board
from utils import *
from time import time
import numpy as np
from search import a_star_search, greedy_search
from search_problem import MinesweeperSearchProblem, huristic
from random import choice
from typing import List


def format_time(number: float) -> str:
    return f"{number//60:2.0f}:{number % 60:2.0f}"


class UI:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.height, self.width = board.size
        self.screen = pygame.display.set_mode(
            (self.width*BLOCK_SIZE, self.height*BLOCK_SIZE+50))
        self.clock = pygame.time.Clock()
        self.start_time = time()
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)
        self.running = True

    def draw_blocks(self):
        for i in range(self.height):
            for j in range(self.width):
                imp = pygame.image.load(
                    ASSETS_IMAGES[self.board[i, j]]).convert()
                self.screen.blit(imp, (j*BLOCK_SIZE, i*BLOCK_SIZE))

    def draw_text(self):
        text = self.font.render(
            f"Flags: {self.board.num_of_markers} \t Time: {format_time(time() - self.start_time)}", True, "black")
        self.screen.blit(text, (0, self.height*BLOCK_SIZE + 10))

    def handle_click(self, event):
        x, y = pygame.mouse.get_pos()
        x = x // BLOCK_SIZE
        y = y // BLOCK_SIZE
        click_func = self.board.reveal
        if event.button == 3:
            click_func = self.board.mark
        if x < self.width and y < self.height and y >= 0 and x >= 0:
            if (self.board.is_bomb(y, x) and event.button == 1):
                self.board.reveal_all()
            else:
                click_func(y, x)

    def run(self):
        while self.running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event)
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill("white")
            self.draw_blocks()
            self.draw_text()
            pygame.display.flip()

            # agent = KR_agent()
            # for i in range(2, 13):
            #     for j in range(2, 13):
            #         agent.procees(board[i-2:i+3, j-2:j+3])
            #         if agent.infer(f'{i}{j}N0'):
            #             print(f'{i} {j}')

            self.clock.tick(60)


class ShowSearch(UI):
    def __init__(self, states: List[Board], epoch: int = 10) -> None:
        super().__init__(states[0])
        self.states = states
        self.state_index = 0
        self.epoch = epoch

    def run(self):
        while self.running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event)
            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill("white")
            self.draw_blocks()
            self.draw_text()
            pygame.display.flip()
            if self.state_index < len(self.states):
                self.board = self.states[self.state_index]
                self.state_index += 1
            self.clock.tick(1000/self.epoch)


if __name__ == "__main__":
    # pygame setup
    board = Board((15, 15))
    # ui = UI(board)
    # ui.run()
    states = greedy_search(MinesweeperSearchProblem(board), huristic)
    ui = ShowSearch(states, 500)
    ui.run()
