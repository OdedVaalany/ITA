# import tkinter as tk
import utils
from board import Board
import random
import os
import tkinter as tk
from PIL import Image, ImageTk


class Minesweeper:

    master: tk.Tk
    board: Board
    rows: int
    columns: int

    def __init__(self, master, board):
        self.board = board
        self.master = master
        self.rows, self.columns = board.size[0], board.size[1]
        self.buttons = []
        self.define_master()
        self.load_images()
        self.create_widgets()

    def define_master(self):
        self.master.title("Minesweeper")
        self.master.resizable(False, False)
        self.master.geometry(
            f"{utils.BRICK_SIZE*self.columns}x{utils.BRICK_SIZE*self.rows}")
        for i in range(self.rows):
            self.master.grid_rowconfigure(i, weight=1)
        for i in range(self.columns):
            self.master.grid_columnconfigure(i, weight=1)

    def load_images(self):
        self.images = {}
        for key, val in utils.ASSETS_IMAGES.items():
            self.images[key] = ImageTk.PhotoImage(
                Image.open(val).resize((30, 30)))

    def create_widgets(self):
        for r in range(0, self.rows):
            row = []
            for c in range(self.columns):
                btn = tk.Label(self.master, width=1, height=1,
                               bg="SystemButtonFace", image=self.images["H"])
                btn.bind("<Button-1>", lambda e, r=r, c=c: self.on_click(r, c))
                btn.bind("<Button-2>", lambda e, r=r,
                         c=c: self.on_right_click(r, c))
                btn.grid(row=r, column=c, sticky="nsew", padx=0, pady=0)
                row.append(btn)
            self.buttons.append(row)

    def on_click(self, r, c):
        board.reveal(r, c)
        if board.is_bomb(r, c):
            self.buttons[r][c].config(image=self.images["10"])
            self.reveal_mines()
            self.game_over("Game Over! You hit a mine.")
        else:
            self.buttons[r][c].config(image=self.images[str(board[r, c])])
        for i in range(self.board.size[0]):
            for j in range(self.board.size[1]):
                if self.board[i, j] != "H":
                    self.buttons[i][j].config(
                        image=self.images[str(board[i, j])])
                    # Here you can add more logic to reveal adjacent cells if there are no adjacent mines

    def on_right_click(self, r, c):
        board.mark(r, c)
        self.buttons[r][c].config(image=self.images[self.board[r, c]])

    def reveal_mines(self):
        for r, c in self.board.bombs:
            self.buttons[r][c].config(image=self.images["10"])

    def game_over(self, message):
        for row in self.buttons:
            for btn in row:
                btn.bind("<Button-1>", lambda e: None)
                btn.bind("<Button-2>", lambda e: None)
        # Open popup with message and option to start a new game
        self.popup = tk.Toplevel(self.master)
        self.popup.title("Game Over")
        message_label = tk.Label(self.popup, text=message)
        message_label.pack()
        new_game_button = tk.Button(
            self.popup, text="New Game", command=self.start_new_game)
        new_game_button.pack()

    def start_new_game(self):
        # Reset the game by creating a new board and restarting the GUI
        self.popup.destroy()
        self.board.reset()
        for r, row in enumerate(self.buttons):
            for c, btn in enumerate(row):
                btn["image"] = self.images["H"]
                btn.bind("<Button-1>", lambda e, r=r, c=c: self.on_click(r, c))
                btn.bind("<Button-2>", lambda e, r=r,
                         c=c: self.on_right_click(r, c))


if __name__ == "__main__":
    board = Board((10, 20))
    root = tk.Tk()
    root.title("Minesweeper")
    game = Minesweeper(root, board)
    root.mainloop()
