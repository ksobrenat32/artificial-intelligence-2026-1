import tkinter as tk
from tkinter import messagebox
import argparse
import ast

def draw_maze(canvas, maze, cell_size, path=None):
    rows = len(maze)
    cols = len(maze[0])
    for i in range(rows):
        for j in range(cols):
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            color = 'white'
            if maze[i][j] == '1':
                color = 'black'
            elif maze[i][j] == 'S':
                color = 'green'
            elif maze[i][j] == 'M':
                color = 'red'
            elif maze[i][j] == '.':
                color = 'yellow'
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')
    if path:
        for (i, j) in path:
            if maze[i][j] not in ['S', 'M']:
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                canvas.create_rectangle(x1, y1, x2, y2, fill='blue', outline='gray')

def animate_path(canvas, maze, path, cell_size, delay=200):
    def step(index):
        if index >= len(path):
            return
        i, j = path[index]
        if maze[i][j] not in ['S', 'M']:
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            canvas.create_rectangle(x1, y1, x2, y2, fill='blue', outline='gray')
        canvas.after(delay, step, index + 1)
    step(0)

def read_maze_from_file(filename):
    with open(filename, 'r') as file:
        maze = [list(line.strip()) for line in file.readlines()]
    return maze

def main():
    parser = argparse.ArgumentParser(description="Maze GUI Simulator")
    parser.add_argument('--file', type=str, help='Path to the maze file')
    parser.add_argument('--maze', type=str, help='Maze as a python list of lists')
    parser.add_argument('--path', type=str, help='Path as a python list of (i, j) tuples')
    args = parser.parse_args()

    if args.maze:
        try:
            maze = ast.literal_eval(args.maze)
        except Exception:
            print("Invalid maze format.")
            return
    elif args.file:
        maze = read_maze_from_file(args.file)
    else:
        print("Provide a maze using --file or --maze.")
        return

    if args.path:
        try:
            path = ast.literal_eval(args.path)
        except Exception:
            print("Invalid path format.")
            path = None
    else:
        path = None

    cell_size = 40
    rows = len(maze)
    cols = len(maze[0])
    width = cols * cell_size
    height = rows * cell_size

    root = tk.Tk()
    root.title("Maze Path Simulator")
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()
    draw_maze(canvas, maze, cell_size)
    if path:
        animate_path(canvas, maze, path, cell_size)
    root.mainloop()

if __name__ == "__main__":
    main()
