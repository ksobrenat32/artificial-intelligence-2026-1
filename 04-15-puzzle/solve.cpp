#include <bits/stdc++.h>
#pragma GCC optimize("O3")
#pragma GCC target("avx2")

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
        std::cout << '\n';
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

    // Calculates the Heuristic function,
    // which accounts for the actual moves needed to arrange tiles in rows and columns
    int calculateHeuristic(const Board& board) const {
        int rowConflicts = 0;
        for (int r = 0; r < 4; ++r) {
            int tilesInRow = 0;
            int emptySpots = 0;

            //count tiles belonging to this row and empty spots
            for (int c = 0; c < 4; ++c) {
                int tileValue = board[r][c];
                if (tileValue != 0) {
                    int goalRow = (tileValue - 1) / 4;
                    if (goalRow == r) {
                        tilesInRow++;
                    }
                } else {
                    emptySpots++;
                }
            }

            // count tiles that should be in this row but are elsewhere
            int tilesOutsideRow = 0;
            for (int otherRow = 0; otherRow < 4; ++otherRow) {
                if (otherRow == r) continue;
                for (int c = 0; c < 4; ++c) {
                    int tileValue = board[otherRow][c];
                    if (tileValue != 0) {
                        int goalRow = (tileValue - 1) / 4;
                        if (goalRow == r) {
                            tilesOutsideRow++;
                        }
                    }
                }
            }

            rowConflicts += tilesOutsideRow;
        }

        int colConflicts = 0;
        for (int c = 0; c < 4; ++c) {
            int tilesInCol = 0;
            int emptySpots = 0;
            // count tiles belonging to this column and empty spots
            for (int r = 0; r < 4; ++r) {
                int tileValue = board[r][c];
                if (tileValue != 0) {
                    int goalCol = (tileValue - 1) % 4;
                    if (goalCol == c) {
                        tilesInCol++;
                    }
                } else {
                    emptySpots++;
                }
            }
            // count tiles that should be in this column but are elsewhere
            int tilesOutsideCol = 0;
            for (int otherCol = 0; otherCol < 4; ++otherCol) {
                if (otherCol == c) continue;
                for (int r = 0; r < 4; ++r) {
                    int tileValue = board[r][otherCol];
                    if (tileValue != 0) {
                        int goalCol = (tileValue - 1) % 4;
                        if (goalCol == c) {
                            tilesOutsideCol++;
                        }
                    }
                }
            }
            colConflicts += tilesOutsideCol;
        }
        int manhattanDistance = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                int tileValue = board[r][c];
                if (tileValue != 0) {
                    int goalRow = (tileValue - 1) / 4;
                    int goalCol = (tileValue - 1) % 4;
                    manhattanDistance += std::abs(r - goalRow) + std::abs(c - goalCol);
                }
            }
        }
        return manhattanDistance + 2 * (rowConflicts + colConflicts);
    }

    void showCurrentState(const Board& board, char move, int moveNumber) {
        std::cout << "\nMove #" << moveNumber << ": '" << move << "'" << '\n';
        printBoard(board);
    }

    // Modified IDA* search: returns 0 when solution found, otherwise returns the minimum f that exceeded the depth limit
    int search(Board& currentBoard, int emptyR, int emptyC, int g_cost, char lastMove, int depthLimit, std::string& path) {
        int h_cost = calculateHeuristic(currentBoard);
        int f = g_cost + h_cost;
        if (f > depthLimit) return f;

        if (currentBoard == goalBoard) return 0;

        int minThreshold = INT_MAX;

        // Down
        if (emptyR < 3 && lastMove != 'u') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR + 1][emptyC]);
            path += 'd';
            int t = search(currentBoard, emptyR + 1, emptyC, g_cost + 1, 'd', depthLimit, path);
            if (t == 0) return 0;
            if (t < minThreshold) minThreshold = t;

            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR + 1][emptyC]);
        }

        // Up
        if (emptyR > 0 && lastMove != 'd') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR - 1][emptyC]);
            path += 'u';
            int t = search(currentBoard, emptyR - 1, emptyC, g_cost + 1, 'u', depthLimit, path);
            if (t == 0) return 0;
            if (t < minThreshold) minThreshold = t;

            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR - 1][emptyC]);
        }

        // Right
        if (emptyC < 3 && lastMove != 'l') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC + 1]);
            path += 'r';
            int t = search(currentBoard, emptyR, emptyC + 1, g_cost + 1, 'r', depthLimit, path);
            if (t == 0) return 0;
            if (t < minThreshold) minThreshold = t;

            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC + 1]);
        }

        // Left
        if (emptyC > 0 && lastMove != 'r') {
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC - 1]);
            path += 'l';
            int t = search(currentBoard, emptyR, emptyC - 1, g_cost + 1, 'l', depthLimit, path);
            if (t == 0) return 0;
            if (t < minThreshold) minThreshold = t;

            path.pop_back();
            std::swap(currentBoard[emptyR][emptyC], currentBoard[emptyR][emptyC - 1]);
        }

        return minThreshold;
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
        std::cout << "\nInitial Board State:" << '\n';
        printBoard(initialBoard);

        int initial_h = calculateHeuristic(initialBoard);
        std::string path = "";

        int depthLimit = initial_h;
        const int MAX_ITERS = 1000;
        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            Board boardCopy = initialBoard;
            path.clear();
            int t = search(boardCopy, startEmptyRow, startEmptyCol, 0, ' ', depthLimit, path);
            if (t == 0) {
                std::cout << "\nSolution found in " << path.length() << " moves: " << path << '\n';

                std::cout << "\nPrinting solution steps:" << '\n';
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
                std::cout << "\nSolution Found! :D" << '\n';
                return;
            }

            if (t == INT_MAX) break;
            depthLimit = t;
        }
        std::cout << "No solution found within reasonable depth :(" << '\n';
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    Board startingBoard;

    std::cout << "Enter the 4x4 board configuration (use 0 for the empty space):" << '\n';
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