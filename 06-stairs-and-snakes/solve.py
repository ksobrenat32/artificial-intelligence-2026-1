#! /usr/bin/env python3
import math

def zeros_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def identity_matrix(size):
    matrix = zeros_matrix(size, size)
    for i in range(size):
        matrix[i][i] = 1.0
    return matrix

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    
    # Generate a result matrix
    result = zeros_matrix(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_vector_multiply(matrix, vector):
    result = [0.0] * len(vector)
    for j in range(len(vector)):
        for i in range(len(vector)):
            result[j] += vector[i] * matrix[i][j]
    return result

def subtract_matrices(A, B):
    rows, cols = len(A), len(A[0])
    result = zeros_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result

def matrix_invert(matrix):
    "Invert a matrix with Gauss-Jordan"
    n = len(matrix)
    aug = [row[:] + identity_matrix(n)[i] for i, row in enumerate(matrix)]
    
    # Forward elimination
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        # Create the pivot for row i
        pivot = aug[i][i]
        for j in range(2 * n):
            aug[i][j] /= pivot
        
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]
    
    # Just extract inverse
    return [row[n:] for row in aug]

def build_transition_matrix(n, ladders, snakes):
    P = zeros_matrix(n + 1, n + 1)
    
    # First state
    for dice_roll in range(1, 7):
        destination = dice_roll
        if destination in ladders:
            destination = ladders[destination]
        elif destination in snakes:
            destination = snakes[destination]
        P[0][destination] += 1/6
    
    for i in range(1, n):
        for dice_roll in range(1, 7):
            if i + dice_roll <= n:
                destination = i + dice_roll
                if destination in ladders:
                    destination = ladders[destination]
                elif destination in snakes:
                    destination = snakes[destination]
                P[i][destination] += 1/6
            else:
                # Bounce back
                overshoot = (i + dice_roll) - n
                destination = n - overshoot
                if destination in ladders:
                    destination = ladders[destination]
                elif destination in snakes:
                    destination = snakes[destination]
                P[i][destination] += 1/6
    
    P[n][n] = 1.0
    
    return P

def get_probabilities_after_n_moves(P, n_moves):
    num_states = len(P)
    initial_state = [0.0] * num_states
    initial_state[0] = 1.0
    
    current_state = initial_state
    for _ in range(n_moves):
        current_state = matrix_vector_multiply(P, current_state)
    
    return current_state

def expected_moves_to_win(P):
    n = len(P) - 1
    
    # Extract Q (transitions between transient states)
    Q = [row[:-1] for row in P[:-1]]
    
    # Create I - Q
    I = identity_matrix(n)
    I_minus_Q = subtract_matrices(I, Q)
    
    # Compute N = (I - Q)^(-1)
    N = matrix_invert(I_minus_Q)
    
    # Expected steps = sum of first row of N
    expected_steps = sum(N[0])
    
    return expected_steps

if __name__ == "__main__":
    # Define the ladders and snakes on the board
    # Key: start position, Value: end position
    ladders = {
        11: 39,
        17: 67,
        19: 45,
        21: 56,
        26: 50,
        43: 84,
        52: 76,
        70: 92,
        74: 100
    }
    snakes = {
        16: 6,
        22: 2,
        36: 20,
        62: 14,
        75: 30,
        78: 49,
        83: 8,
        93: 40,
        96: 69
    }

    n = 100
    
    print("=" * 70)
    print("SNAKES AND LADDERS - MARKOV CHAINS")
    print("=" * 70)
    
    P = build_transition_matrix(n, ladders, snakes)
    
    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTION AFTER 25 MOVES")
    print("=" * 70)
    
    prob_25 = get_probabilities_after_n_moves(P, 25)
    
    print(f"\nProbability of winning: {prob_25[100]:.6f}")
    print(f"Probability of still not winning: {1 - prob_25[100]:.6f}")
    print()
    
    print("=" * 70)
    print("\nShowing all states with probability > 0.1%:")
    print("-" * 70)
    
    non_zero_probs = []
    for state in range(len(prob_25)):
        if prob_25[state] > 0.001:
            non_zero_probs.append((state, prob_25[state]))
    
    for state, prob in non_zero_probs:
        print(f"  Square {state:3d}: {prob:.6f} ({prob*100:.2f}%)")
    
    with open('probability_vector_25.txt', 'w') as f:
        f.write("Probability Vector after 25 moves\n")
        f.write("=" * 70 + "\n\n")
        for state in range(len(prob_25)):
            f.write(f"Square {state:3d}: {prob_25[state]:.10f}\n")
    
    top_positions_25 = []
    for i in range(1, 100):
        if prob_25[i] > 0.001:
            top_positions_25.append((i, prob_25[i]))
    top_positions_25.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most likely positions:")
    for i, (pos, prob) in enumerate(top_positions_25[:10], 1):
        print(f"  {i}. Square {pos:3d}: {prob:.6f}")
    
    # Calculate probabilities after 50 moves
    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTION AFTER 50 MOVES")
    print("=" * 70)
    prob_50 = get_probabilities_after_n_moves(P, 50)
    print(f"\nProbability of winning: {prob_50[100]:.6f}")
    print(f"Probability of still not winning: {1 - prob_50[100]:.6f}")
    print()
    
    print("=" * 70)
    print("\nShowing all states with probability > 0.1%:")
    print("-" * 70)
    
    non_zero_probs_50 = []
    for state in range(len(prob_50)):
        if prob_50[state] > 0.001:
            non_zero_probs_50.append((state, prob_50[state]))
    
    for state, prob in non_zero_probs_50:
        print(f"  Square {state:3d}: {prob:.6f} ({prob*100:.2f}%)")
    
    with open('probability_vector_50.txt', 'w') as f:
        f.write("Probability Vector after 50 moves\n")
        f.write("=" * 70 + "\n\n")
        for state in range(len(prob_50)):
            f.write(f"Square {state:3d}: {prob_50[state]:.10f}\n")
    
    top_positions_50 = []
    for i in range(1, 100):
        if prob_50[i] > 0.001:
            top_positions_50.append((i, prob_50[i]))
    top_positions_50.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most likely positions:")
    for i, (pos, prob) in enumerate(top_positions_50[:10], 1):
        print(f"  {i}. Square {pos:3d}: {prob:.6f}")
    
    print("\n" + "=" * 70)
    print("EXPECTED NUMBER OF MOVES TO WIN")
    print("=" * 70)
    expected_moves = expected_moves_to_win(P)
    print(f"\nExpected moves to win: {expected_moves:.4f}")
