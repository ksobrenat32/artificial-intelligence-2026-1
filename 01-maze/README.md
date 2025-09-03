# Maze Solver

Write down a program that solves a maze using a simple reflex agent and, separately, a goal-based agent. The program will take a maze represented as a matrix.
The matrix will have the following values in each cell:

    1: A wall
    0: a free path
    S: The start
    M: The goal

Finally, write a report with the following sections:

    Title
    Abstract
    Goals
    Introduction
    Development
    Results
    Conclusions
    Bibliography

The program and the report will be performed in teams of four.

## Testing

### Using a file

1. Create a text file representing the maze, for example `maze.txt`:

```
11111
10001
10101
10001
11111
```

2. Run the program with the `--file` argument:

```sh
python agent.py --file maze.txt
```

### Using a string representation

```sh
python 01-maze/agent.py --maze "[['S','0','1'],['0','1','M'],['0','0','0']]"
```
