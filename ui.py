import tkinter as tk
from board import Board
from PIL import Image, ImageTk
import utils
import datetime as date
import time


class BoardUI(tk.Frame):

    def __init__(self, master=None, board: Board = None):
        super().__init__(master, bg="yellow", padx=10, pady=10)
        self["width"] = master["width"]
        self["height"] = master["height"]-50
        self.master = master
        self.board = board
        self.size = (board.size[0], board.size[1])
        self.set_background()
        self.create_widgets()
        self.pack()

    def set_background(self):
        image = Image.open(
            "./assets/board_background.png").resize((self["width"], self["height"]))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo)
        label.image = photo
        label.place(x=-self["padx"], y=-self['pady'])

    def create_widgets(self):
        brick_size = min((self["width"]-self["padx"]*2)//self.size[1],
                         (self["height"]-self["pady"]*2)//self.size[0])
        xpad = (self["width"]-self["padx"]*2 -
                brick_size*self.size[1])//2
        ypad = (self["height"]-self["pady"]*2 -
                brick_size*self.size[0])//2

        self.buttons = []
        padding = 1
        brick_size -= padding
        for i in range(self.size[0]):
            self.buttons.append([])
            for j in range(self.size[1]):
                button = tk.Label(self, text="", bg="#c0c0c0",
                                  padx=0, pady=0, font=("Arial", 12))
                button.place(x=xpad+j*(brick_size+padding),
                             y=ypad+i*(brick_size+padding), width=brick_size, height=brick_size)
                button.bind(
                    "<Button-1>", lambda e, i=i, j=j: self.board.reveal(i, j))
                button.bind(
                    "<Button-2>", lambda e, i=i, j=j: self.board.mark(
                        i, j)
                )
                self.buttons[-1].append(button)

    def update(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                val = self.board[i, j]
                self.buttons[i][j].config(**utils.get_config_for_button(val))
        super().update()


class StatsUi(tk.Frame):
    def __init__(self, master: tk.Tk, board: Board):
        super().__init__(master, padx=10, pady=10)
        self["width"] = master["width"]
        self["height"] = 100
        self.master = master
        self.board = board
        self.time_var = tk.StringVar(value="0")
        self.mark_var = tk.StringVar(value="0")
        self.open_cells_var = tk.StringVar(value="0")
        self.time = time.time()
        self.create_widgets()
        self.pack()

    def create_widgets(self):
        font = ("Calibri", 14)
        # Create a label for time
        time_label = tk.Label(self, text="Time: ", font=font)
        time_label.grid(row=0, column=0)
        time_value = tk.Label(
            self, textvariable=self.time_var, font=font)
        time_value.grid(row=0, column=1)

        # Create a label for number of markers
        mark_label = tk.Label(self, text="Markers: ", font=font)
        mark_label.grid(row=0, column=2)
        mark_value = tk.Label(
            self, textvariable=self.mark_var, font=font)
        mark_value.grid(row=0, column=3)

        # Create a label for number of opens cells
        mark_label = tk.Label(self, text="Markers: ", font=font)
        mark_label.grid(row=0, column=4)
        mark_value = tk.Label(
            self, textvariable=self.open_cells_var, font=font)
        mark_value.grid(row=0, column=5)

    def update(self):
        gap = time.time() - self.time
        self.time_var.set(f'{int((gap//1))}')
        self.mark_var.set(str(self.board.num_of_markers))
        self.open_cells_var.set(str(self.board.num_of_opens))
        super().update()


# Create the main window
board = Board((10, 20))
brick_size = 50
window = tk.Tk(screenName="Minesweeper", className="Minesweeper")
window["padx"] = 0
window["pady"] = 0
window["width"] = brick_size*board.size[1]
window["height"] = 50 + brick_size*board.size[0]
# window.geometry(f'{window["width"]}x{window["height"]}')
window.resizable(False, False)

# Create a frame with yellow background
stats = StatsUi(window, board)
frame = BoardUI(window, board)

# Start the main event loop

while window.winfo_exists() == 1:
    frame.update()
    stats.update()
    time.sleep(1/60)
# window.mainloop()
