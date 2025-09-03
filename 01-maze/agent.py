import argparse
from typing import List, Optional
from collections import deque
import ast

def print_maze(maze) -> None:
    for row in maze:
        print(" ".join(row))
    print()

def find_start(maze) -> Optional[tuple]:
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 'S':
                return (i, j)
    return None

def find_goal(maze) -> Optional[tuple]:
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 'M':
                return (i, j)
    return None

def is_valid_move(maze, position) -> bool:
    x, y = position
    if 0 <= x < len(maze) and 0 <= y < len(maze[0]):
        return maze[x][y] in ['0', 'M']
    return False

def mark_path(maze, path) -> None:
    for x, y in path:
        if maze[x][y] != 'M':
            maze[x][y] = '.'

def simple_reflex_agent(maze) -> Optional[List[tuple]]:
    start = find_start(maze)
    goal = find_goal(maze)
    if not start or not goal:
        print("Maze must have a start (S) and a goal (M).")
        return None

    x, y = start
    path = [(x, y)]

    while (x, y) != goal:
        moved = False
        # Try to move right
        if not moved and is_valid_move(maze, (x, y + 1)) and (x, y + 1) not in path:
            y += 1
            path.append((x, y))
            moved = True
        # Try to move down
        if not moved and is_valid_move(maze, (x + 1, y)) and (x + 1, y) not in path:
            x += 1
            path.append((x, y))
            moved = True
        # Try to move left
        if not moved and is_valid_move(maze, (x, y - 1)) and (x, y - 1) not in path:
            y -= 1
            path.append((x, y))
            moved = True
        # Try to move up
        if not moved and is_valid_move(maze, (x - 1, y)) and (x - 1, y) not in path:
            x -= 1
            path.append((x, y))
            moved = True

        if not moved:
            print("Agent is stuck, no valid moves available.")
            break
    
    # Do not mark the start position
    mark_path(maze, path[1:])
    return path

def goal_based_agent(maze) -> Optional[List[tuple]]:
    start = find_start(maze)
    goal = find_goal(maze)
    if not start or not goal:
        print("Maze must have a start (S) and a goal (M).")
        return None

    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        x, y = path[-1]

        if (x, y) == goal:
            mark_path(maze, path[1:]) # Mark path except for start
            return path

        # Explore neighbors: Right, Down, Left, Up
        for move_x, move_y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + move_x, y + move_y
            
            if is_valid_move_bfs(maze, (next_x, next_y), visited):
                visited.add((next_x, next_y))
                new_path = list(path)
                new_path.append((next_x, next_y))
                queue.append(new_path)

    return None # No path found

def is_valid_move_bfs(maze, position, visited) -> bool:
    x, y = position
    if not (0 <= x < len(maze) and 0 <= y < len(maze[0])):
        return False
    if maze[x][y] == '1':
        return False
    if (x, y) in visited:
        return False
    return True

def read_maze_from_file(filename):
    with open(filename, 'r') as file:
        maze = [list(line.strip()) for line in file.readlines()]
    return maze

def main():
    # Parse maze from file on argument
    arg_parser = argparse.ArgumentParser(description="Maze Solver Comparison")
    arg_parser.add_argument('--file', type=str, help='Path to the maze file')
    arg_parser.add_argument('--maze', type=str, help='Maze represented as a python list of lists')
    args = arg_parser.parse_args()

    initial_maze = None
    if args.maze:
        try:
            initial_maze = ast.literal_eval(args.maze)
        except (ValueError, SyntaxError):
            print("Error: Invalid format for --maze argument. Please provide a valid list of lists string.")
            return
    elif args.file:
        initial_maze = read_maze_from_file(args.file)
    else:
        print("Error: Please provide a maze using --file or --maze.")
        return

    if not initial_maze:
        print("Could not load maze.")
        return

    # --- Initial Maze ---
    print("Initial Maze:")
    print_maze(initial_maze)

    # --- Simple Reflex Agent ---
    print("\n--- Running Simple Reflex Agent ---")
    # Create a deep copy of the maze for the simple reflex agent
    maze_simple = [row[:] for row in initial_maze]
    path_simple = simple_reflex_agent(maze_simple)
    
    print("Final Maze with Simple Reflex Agent path:")
    print_maze(maze_simple)

    if path_simple:
        steps_simple = len(path_simple) - 1
        print(f"Steps taken by Simple Reflex Agent: {steps_simple}")
    else:
        print("Simple Reflex Agent could not find a path.")

    # --- Goal-Based Agent (BFS) ---
    print("\n--- Running Goal-Based Agent (BFS) ---")
    # Create a deep copy of the maze for the goal-based agent
    maze_goal = [row[:] for row in initial_maze]
    path_goal = goal_based_agent(maze_goal)

    print("Final Maze with Goal-Based Agent path:")
    print_maze(maze_goal)

    if path_goal:
        steps_goal = len(path_goal) - 1
        print(f"Steps taken by Goal-Based Agent: {steps_goal}")
    else:
        print("Goal-Based Agent could not find a path.")

if __name__ == "__main__":
    main()


