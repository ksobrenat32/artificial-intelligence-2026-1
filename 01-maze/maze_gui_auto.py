import subprocess
import argparse
import ast
import sys
import os
import tkinter as tk

def draw_maze(canvas, maze, cell_size, path=None, color_path=None, offset_x=0, offset_y=0):
    rows = len(maze)
    cols = len(maze[0])
    for i in range(rows):
        for j in range(cols):
            x1 = offset_x + j * cell_size
            y1 = offset_y + i * cell_size
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
                x1 = offset_x + j * cell_size
                y1 = offset_y + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                canvas.create_rectangle(x1, y1, x2, y2, fill=color_path or 'blue', outline='gray')

def animate_two_mazes(canvas, maze_simple, maze_goal, path_simple, path_goal, cell_size, delay=200, offset=20):
    rows = len(maze_simple)
    cols = len(maze_simple[0])
    offset_x_goal = cols * cell_size + offset
    max_len = max(len(path_simple) if path_simple else 0, len(path_goal) if path_goal else 0)
    def step(index):
        if index >= max_len:
            return
        # Animate simple reflex agent
        if path_simple and index < len(path_simple):
            i, j = path_simple[index]
            if maze_simple[i][j] not in ['S', 'M']:
                maze_simple[i][j] = '.'
        # Animate goal-based agent
        if path_goal and index < len(path_goal):
            i, j = path_goal[index]
            if maze_goal[i][j] not in ['S', 'M']:
                maze_goal[i][j] = '.'
        # Draw both mazes with correct path color for current step
        # Draw simple reflex path up to current index in blue
        draw_maze(canvas, maze_simple, cell_size, path=path_simple[:index+1] if path_simple else None, color_path='blue', offset_x=0, offset_y=0)
        # Draw goal-based path up to current index in orange
        draw_maze(canvas, maze_goal, cell_size, path=path_goal[:index+1] if path_goal else None, color_path='orange', offset_x=offset_x_goal, offset_y=0)
        # Redraw legend below both mazes every step
        legend_y = rows * cell_size + 10
        canvas.delete('legend')
        canvas.create_rectangle(10, legend_y, 30, legend_y+20, fill='blue', outline='gray', tags='legend')
        canvas.create_text(40, legend_y+10, anchor='w', text='Simple Reflex Agent', tags='legend')
        canvas.create_rectangle(offset_x_goal+10, legend_y, offset_x_goal+30, legend_y+20, fill='orange', outline='gray', tags='legend')
        canvas.create_text(offset_x_goal+40, legend_y+10, anchor='w', text='Goal-Based Agent', tags='legend')
        canvas.after(delay, step, index + 1)
    # Draw initial state
    draw_maze(canvas, maze_simple, cell_size, path=None, color_path='blue', offset_x=0, offset_y=0)
    draw_maze(canvas, maze_goal, cell_size, path=None, color_path='orange', offset_x=offset_x_goal, offset_y=0)
    step(0)

def read_maze_from_file(filename):
    with open(filename, 'r') as file:
        maze = [list(line.strip()) for line in file.readlines()]
    return maze

def get_paths_from_agent(agent_path, maze_file):
    # Run agent.py and extract both paths from output
    cmd = [sys.executable, agent_path, '--file', maze_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    path_simple = None
    path_goal = None
    lines = output.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith('PATH:'):
            # Heuristic: first PATH is simple, second is goal-based
            try:
                path = ast.literal_eval(line[5:].strip())
                if path_simple is None:
                    path_simple = path
                else:
                    path_goal = path
            except Exception:
                continue
    return path_simple, path_goal

def main():
    parser = argparse.ArgumentParser(description="Maze GUI Simulator (auto path, both agents)")
    parser.add_argument('--file', type=str, required=True, help='Path to the maze file')
    parser.add_argument('--agentpy', type=str, default='01-maze/agent.py', help='Path to agent.py')
    args = parser.parse_args()

    maze = read_maze_from_file(args.file)
    path_simple, path_goal = get_paths_from_agent(args.agentpy, args.file)
    if not path_simple and not path_goal:
        print("Could not obtain paths from agent.py. Make sure agent.py prints the path as 'PATH: [...]' on a line.")
        return

    cell_size = 40
    rows = len(maze)
    cols = len(maze[0])
    offset = 20
    width = cols * cell_size * 2 + offset
    height = rows * cell_size + 50  # Extra space for legend

    # Create two copies of the maze for each agent
    import copy
    maze_simple = copy.deepcopy(maze)
    maze_goal = copy.deepcopy(maze)

    root = tk.Tk()
    root.title("Maze Path Simulator (side by side)")
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()
    animate_two_mazes(canvas, maze_simple, maze_goal, path_simple, path_goal, cell_size, offset=offset)
    root.mainloop()

if __name__ == "__main__":
    main()
