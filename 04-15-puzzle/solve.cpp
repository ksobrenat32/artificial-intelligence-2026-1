#include <bits/stdc++.h>

// A more readable type alias for the board
using Board = std::array<std::array<int, 4>, 4>;

void printBoard(const Board& board) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (board[r][c] == 0) {
                std::cout << " . ";
            } else {
                std::cout << " " << board[r][c];
                // Just print extra space if it is a number smaller than 10
                if(board[r][c] < 10){
                    std::cout << " ";
                }
            }
        }
        std::cout << std::endl;
    }
}

class FifteenPuzzleSolver {
private:
    Board initialBoard;
    int startEmptyRow = 0;
    int startEmptyCol = 0;

    const Board goalBoard{{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 0}
    }};

    // Calculates the Manhattan Distance heuristic,
    // the sum of the horizontal and vertical distances of each tile from its goal position.
    int calculateManhattanDistance(const Board& board) const {
        int distance = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                int tileValue = board[r][c];
                if (tileValue != 0) {
                    int goalRow = (tileValue - 1) / 4;
                    int goalCol = (tileValue - 1) % 4;
                    distance += std::abs(r - goalRow) + std::abs(c - goalCol);
                }
            }
        }
        return distance;
    }

    void showCurrentState(const Board& board, char move, int moveNumber) {
        std::cout << "\nMove #" << moveNumber << ": '" << move << "'" << std::endl;
        printBoard(board);
    }

    // The recursive search function, it explores paths up to a certain `depthLimit`.
    bool search(Board& currentBoard, int emptyR, int emptyC, int g_cost, char lastMove, int depthLimit, std::string& path) {
        // If f exceeds the current depth limit, we prune this branch.
        int h_cost = calculateManhattanDistance(currentBoard);
        if (g_cost + h_cost > depthLimit) {
            return false;
        }

        // If already at the goal state, break
        if (currentBoard == goalBoard) {
            return true;
        }

        // Down
        if (emptyR < 3 && lastMove != 'u') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR + 1][emptyC]);
            path += 'd';
            if (search(currentBoard, emptyR + 1, emptyC, g_cost + 1, 'd', depthLimit, path)) return true;

            // Undo the move (backtrack)
            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR + 1][emptyC]);
        }

        // Up
        if (emptyR > 0 && lastMove != 'd') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR - 1][emptyC]);
            path += 'u';
            if (search(currentBoard, emptyR - 1, emptyC, g_cost + 1, 'u', depthLimit, path)) return true;

            // Undo the move (backtrack)
            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR - 1][emptyC]);
        }

        // Right
        if (emptyC < 3 && lastMove != 'l') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC + 1]);
            path += 'r';
            if (search(currentBoard, emptyR, emptyC + 1, g_cost + 1, 'r', depthLimit, path)) return true;

            // Undo the move (backtrack)
            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC + 1]);
        }

        // Left
        if (emptyC > 0 && lastMove != 'r') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC - 1]);
            path += 'l';
            if (search(currentBoard, emptyR, emptyC - 1, g_cost + 1, 'l', depthLimit, path)) return true;

            // Undo the move (backtrack)
            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC - 1]);
        }

        return false; // No solution found
    }

public:
    FifteenPuzzleSolver(Board boardState) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                initialBoard[r][c] = boardState[r][c];
                if (boardState[r][c] == 0) {
                    startEmptyRow = r;
                    startEmptyCol = c;

                }
            }
        }
    }

    void Solve() {
        std::cout << "\nInitial Board State:" << std::endl;
        printBoard(initialBoard);

        int initial_h = calculateManhattanDistance(initialBoard);
        std::string path = "";

        // Depth limit starts from the initial heuristic value
        for (int depthLimit = initial_h; depthLimit < 100; ++depthLimit) {
            Board boardCopy = initialBoard;

            if (search(boardCopy, startEmptyRow, startEmptyCol, 0, ' ', depthLimit, path)) {
                std::cout << "\nSolution found in " << path.length() << " moves: " << path << std::endl;

                std::cout << "\nPrinting solution steps:" << std::endl;
                Board stepBoard = initialBoard;
                int emptyR = startEmptyRow;
                int emptyC = startEmptyCol;

                for (size_t i = 0; i < path.length(); ++i) {
                    char move = path[i];
                    if (move == 'u') {
                        std::swap(stepBoard[emptyR][emptyC], stepBoard[emptyR - 1][emptyC]);
                        emptyR--;
                    } else if (move == 'd') {
                        std::swap(stepBoard[emptyR][emptyC], stepBoard[emptyR + 1][emptyC]);
                        emptyR++;
                    } else if (move == 'l') {
                        std::swap(stepBoard[emptyR][emptyC], stepBoard[emptyR][emptyC - 1]);
                        emptyC--;
                    } else if (move == 'r') {
                        std::swap(stepBoard[emptyR][emptyC], stepBoard[emptyR][emptyC + 1]);
                        emptyC++;
                    }
                    showCurrentState(stepBoard, move, i + 1);
                }
                std::cout << "\nSolution Found! :D" << std::endl;
                return;
            }
        }
        std::cout << "No solution found within reasonable depth :(" << std::endl;
    }
};

int main() {
    Board startingBoard;

    std::cout << "Enter the 4x4 board configuration (use 0 for the empty space):" << std::endl;
    for(int r = 0; r < 4; ++r) {
        for(int c = 0; c < 4; ++c) {
            int value;
            std::cin >> value;
            startingBoard[r][c] = value;
        }
    }

    FifteenPuzzleSolver solver(startingBoard);
    solver.Solve();
    return 0;
}