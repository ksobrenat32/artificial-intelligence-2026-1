# Three missionaries and three cannibals are on the left bank of a river. They
# must cross to the right bank using a boat that:
#
# 1. Carries at most two people.
# 2. Never travels empty.
# 3. Never leaves a bank with more cannibals than missionaries if at least
# one missionary is present.
#
# Goal: move everyone to the right bank without violating constraints.

from collections import deque
from typing import Optional

class MissionariesAndCannibals:
    initial_state: tuple[int, int, int]

    def __init__(self, missionaries=3, cannibals=3, capacity=2) -> None:
        self.total_missionaries = missionaries
        self.total_cannibals = cannibals
        self.capacity = capacity
        # boat_position: 1 for left bank, 0 for right bank
        self.initial_state = (missionaries, cannibals, 1)
        self.goal_state = (0, 0, 0)

    def is_valid(self, state) -> bool:
        m_left, c_left, _ = state
        
        if not (0 <= m_left <= self.total_missionaries and 0 <= c_left <= self.total_cannibals):
            return False

        m_right = self.total_missionaries - m_left
        c_right = self.total_cannibals - c_left

        if m_left > 0 and c_left > m_left:
            return False
        
        if m_right > 0 and c_right > m_right:
            return False
            
        return True

    def get_possible_moves(self) -> list[tuple[int, int]]:
        moves = []
        for m in range(self.capacity + 1):
            for c in range(self.capacity + 1):
                if 1 <= m + c <= self.capacity:
                    moves.append((m, c))
        return moves

    def solve(self) -> tuple[Optional[list[tuple[int, int, int]]], int]:
        queue = deque([[self.initial_state]])
        visited = {self.initial_state}
        moves = self.get_possible_moves()

        while queue:
            path = queue.popleft()
            current_state = path[-1]

            if current_state == self.goal_state:
                return path, len(visited)

            m_left, c_left, boat_pos = current_state
            
            # -1 for L -> R, +1 for R -> L
            direction = -1 if boat_pos == 1 else 1

            for m_move, c_move in moves:
                new_m_left = m_left + (direction * m_move)
                new_c_left = c_left + (direction * c_move)
                new_boat_pos = 1 - boat_pos

                next_state = (new_m_left, new_c_left, new_boat_pos)

                if self.is_valid(next_state) and next_state not in visited:
                    visited.add(next_state)
                    new_path = list(path)
                    new_path.append(next_state)
                    queue.append(new_path)
        
        return None, len(visited)
    
    def find_all_paths(self) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]]:
        all_paths = []
        valid_solutions = []
        queue = deque([[self.initial_state]])
        moves = self.get_possible_moves()
        max_depth = 15
        
        while queue:
            path = queue.popleft()
            current_state = path[-1]
            
            # Check if path is too long
            if len(path) > max_depth:
                continue
                
            # Check if current state breaks rules
            if not self.is_valid(current_state):
                all_paths.append(path)
                continue
                
            # Check if we reached the goal
            if current_state == self.goal_state:
                all_paths.append(path)
                valid_solutions.append(path)
                continue
                
            # Generate next states
            m_left, c_left, boat_pos = current_state
            direction = -1 if boat_pos == 1 else 1
            
            for m_move, c_move in moves:
                new_m_left = m_left + (direction * m_move)
                new_c_left = c_left + (direction * c_move)
                new_boat_pos = 1 - boat_pos
                next_state = (new_m_left, new_c_left, new_boat_pos)
                
                # Only explore paths with valid number ranges (no negative numbers)
                if (0 <= new_m_left <= self.total_missionaries and 
                    0 <= new_c_left <= self.total_cannibals and
                    next_state not in path):
                    new_path = path + [next_state]
                    queue.append(new_path)
        
        return all_paths, valid_solutions

def format_state(state, total_m=3, total_c=3) -> str:
    m_left, c_left, boat_pos = state
    m_right = total_m - m_left
    c_right = total_c - c_left
    boat_side = 'L' if boat_pos == 1 else 'R'
    return f"({m_left}, {c_left}, {m_right}, {c_right}, {boat_side})"

def print_solution(path, total_m=3, total_c=3) -> None:
    if not path:
        print("No solution found.")
        return
    
    print(f"Shortest solution found in {len(path) - 1} crossings:")
    for i, state in enumerate(path):
        print(f"  Step {i}: {format_state(state, total_m, total_c)}")

def print_all_paths(solver) -> None:
    all_paths, valid_solutions = solver.find_all_paths()
    
    print(f"\nExploring all possible paths (found {len(all_paths)} paths):\n")
    
    for i, path in enumerate(all_paths, 1):
        print(f"--- Path {i} ---")
        for j, state in enumerate(path):
            print(f"  Step {j}: {format_state(state, solver.total_missionaries, solver.total_cannibals)}")
        
        # Check path result
        final_state = path[-1]
        if not solver.is_valid(final_state):
            print("  --> This path BREAKS THE RULES and ends here.")
        elif final_state == solver.goal_state:
            print(f"  --> This path REACHES THE GOAL in {len(path) - 1} steps.")
        else:
            print("  --> This path was truncated (max depth reached).")
        print()
    
    # Print the best solution
    if valid_solutions:
        best_solution = min(valid_solutions, key=len)
        print("="*50)
        print("BEST SOLUTION:")
        print("="*50)
        print(f"Optimal solution found in {len(best_solution) - 1} crossings:")
        for i, state in enumerate(best_solution):
            print(f"  Step {i}: {format_state(state, solver.total_missionaries, solver.total_cannibals)}")
    else:
        print("="*50)
        print("NO VALID SOLUTION FOUND!")
        print("="*50)


if __name__ == '__main__':
    print("Missionaries and Cannibals - All Possible Paths")
    print("="*50)

    solver = MissionariesAndCannibals(missionaries=3, cannibals=3, capacity=2)
    print_all_paths(solver)

    print("\n" + "="*50)
    print("Missionaries and Cannibals - With a boat capacity of three")
    print("="*50)

    solver = MissionariesAndCannibals(missionaries=3, cannibals=3, capacity=3)
    print_all_paths(solver)
