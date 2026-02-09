# Sudoku Constraint Satisfaction Lab
**Based on Constraint Satisfaction Problems from AI: A Modern Approach (Chapter 6)**

## Learning Objectives

By the end of this lab, you will be able to:

1. **Understand** the components of a constraint satisfaction problem (variables, domains, constraints) in the context of Sudoku
2. **Distinguish** between different search strategies (backtracking, forward checking, constraint propagation) and their computational trade-offs

## Lab Overview

This lab uses Sudoku puzzles as a testbed for exploring constraint satisfaction problem (CSP) solving techniques. Sudoku is an ideal domain because it has clear variables (cells), well-defined domains (digits 1-9), and explicit constraints (no repeats in rows, columns, or boxes), yet can range from trivial to extremely difficult.

Rather than implementing these algorithms yourself, you'll run complete implementations and observe how different CSP techniques handle puzzles of varying difficulty. You'll see firsthand how intelligent constraint propagation can solve easy puzzles without search, how search strategies differ in their efficiency, and why heuristics matter enormously.

**Why Sudoku for CSP:**
- Clear variable/domain/constraint structure maps directly to CSP theory
- Problem difficulty can be precisely controlled
- Visual output makes algorithm behavior observable
- Different techniques show dramatic performance differences
- Connects to real-world CSP applications (scheduling, configuration, planning)

**Setup Requirements:**
- Python 3.10+
- Libraries: `copy`, `time`, `random`, `z3`

Each exercise demonstrates a different CSP solving technique, building from simple constraint checking to sophisticated arc consistency algorithms. By observing these implementations, you'll understand the theoretical concepts from the textbook in concrete, executable form.

I suggest running the following commands from your base user directory:


```bash
mkdir cs430 
cd cs430 
uv init 
uv add z3
source .venv/bin/activate
touch sudoku.ipynb
```

The last command will create a file such as `sudoku.ipynb`. 

---

#### uv

I highly recommend uv (https://docs.astral.sh/uv/). It (according to their docs):

- 🚀 Is a single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
- ⚡️ Is 10-100x faster than pip.
- 🗂️ Provides comprehensive project management, with a universal lockfile.
- ❇️ Runs scripts, with support for inline dependency metadata.
- 🐍 Installs and manages Python versions.

---

#### Jupyter Notebook

This lab is designed to be run in a Jupyter notebook environment, because the examples build progressively.

Select the virtual environment created by `uv`` (`cs430`) as the kernel for your Jupyter notebook.

Paste the code for each exercise in a new code cell.

If you can't use Jupyter Notebook for whatever reason, just build up a regular Python program, and ignore output from earlier exercises.

#### Submission 

Make sure to record your answers to *all* reflections to submit at the end of the lab!

---

## Exercise 1: Understanding the CSP Formulation

### Description

Before exploring solution techniques, we need to understand how Sudoku maps to the CSP framework. This exercise implements the basic CSP representation and demonstrates how constraints are checked. Understanding this foundation is crucial—every solving technique we'll explore builds on this representation of variables, domains, and constraints.

### Key Concepts

- **Variables**: The cells in a Sudoku grid that need values assigned (81 cells in standard 9×9 Sudoku)
- **Domain**: The set of possible values each variable can take (digits 1-9, reduced as constraints are applied)
- **Constraints**: Rules that restrict which value combinations are valid (no repeats in rows/columns/boxes)
- **Assignment**: A binding of values to variables (partial or complete)
- **Constraint checking**: Verifying that an assignment satisfies all relevant constraints

### Task

Run the code below and observe the CSP structure. Pay attention to:
- How the 9×9 grid is represented as a collection of variables
- How domains are maintained for each cell
- The three types of constraints (row, column, box)
- How constraint violations are detected

```python
class SudokuCSP:
    """Sudoku as a Constraint Satisfaction Problem"""
    
    def __init__(self, puzzle):
        """
        Initialize Sudoku CSP.
        puzzle: 9x9 list where 0 represents empty cell
        """
        self.size = 9
        self.box_size = 3
        self.puzzle = [row[:] for row in puzzle]  # Deep copy
        
        # Variables: each cell (i,j)
        # Domains: possible values for each cell
        self.domains = {}
        for i in range(self.size):
            for j in range(self.size):
                if puzzle[i][j] == 0:
                    # Empty cell: domain is all digits 1-9
                    self.domains[(i, j)] = set(range(1, 10))
                else:
                    # Given cell: domain is fixed value
                    self.domains[(i, j)] = {puzzle[i][j]}
        self.apply_initial_constraints()
    
    def get_row_neighbors(self, row, col):
        """Get all cells in the same row"""
        return [(row, c) for c in range(self.size) if c != col]
    
    def get_col_neighbors(self, row, col):
        """Get all cells in the same column"""
        return [(r, col) for r in range(self.size) if r != row]
    
    def get_box_neighbors(self, row, col):
        """Get all cells in the same 3x3 box"""
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        neighbors = []
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if (i, j) != (row, col):
                    neighbors.append((i, j))
        return neighbors
    
    def get_all_neighbors(self, row, col):
        """Get all cells that constrain this cell"""
        neighbors = set()
        neighbors.update(self.get_row_neighbors(row, col))
        neighbors.update(self.get_col_neighbors(row, col))
        neighbors.update(self.get_box_neighbors(row, col))
        return list(neighbors)
    
    def is_consistent(self, row, col, value):
        """
        Check if assigning value to (row, col) violates constraints.
        A value is consistent if it doesn't appear in any neighbor.
        """
        # Check row constraint
        for c in range(self.size):
            if c != col and self.puzzle[row][c] == value:
                return False
        
        # Check column constraint
        for r in range(self.size):
            if r != row and self.puzzle[r][col] == value:
                return False
        
        # Check box constraint
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if (i, j) != (row, col) and self.puzzle[i][j] == value:
                    return False
        
        return True
    
    def is_complete(self):
        """Check if all cells are assigned"""
        for i in range(self.size):
            for j in range(self.size):
                if self.puzzle[i][j] == 0:
                    return False
        return True
    
    def print_puzzle(self, title="Sudoku"):
        """Pretty print the puzzle"""
        print(f"\n{title}")
        print("─" * 25)
        for i in range(self.size):
            if i > 0 and i % 3 == 0:
                print("─" * 25)
            row_str = ""
            for j in range(self.size):
                if j > 0 and j % 3 == 0:
                    row_str += "│ "
                val = self.puzzle[i][j]
                row_str += str(val) if val != 0 else "."
                row_str += " "
            print(row_str)
        print()

    def apply_initial_constraints(self):
        """Reduce domains based on existing values"""
        for (i, j), domain in self.domains.items():
            if len(domain) == 1:  # Skip given values
                continue
            
            # Remove values that appear in neighbors
            for neighbor in self.get_all_neighbors(i, j):
                ni, nj = neighbor
                if len(self.domains[(ni, nj)]) == 1:
                    assigned_value = next(iter(self.domains[(ni, nj)]))
                    domain.discard(assigned_value)

# Demonstration: Easy 4x4 Sudoku (for clarity)
print("=== CSP Formulation of Sudoku ===\n")

# Simple 4x4 Sudoku for demonstration (2x2 boxes)
# This makes it easier to see all variables and constraints
simple_puzzle_4x4 = [
    [1, 0, 0, 4],
    [0, 4, 1, 0],
    [4, 0, 0, 1],
    [0, 1, 4, 0]
]

print("4x4 Sudoku (2x2 boxes):")
print("Variables: 16 cells")
print("Domains: {1, 2, 3, 4} for each cell")
print("Constraints: No repeats in rows, columns, or 2x2 boxes\n")

# Show the puzzle structure
for i, row in enumerate(simple_puzzle_4x4):
    print(f"Row {i}: {row}")

print("\n--- Constraint Analysis ---\n")

# Demonstrate constraint checking
test_cell = (0, 1)  # Second cell in first row
print(f"Cell {test_cell}: Currently empty (0)")
print(f"Row neighbors: {[(0, c) for c in range(4) if c != test_cell[1]]}")
print(f"Column neighbors: {[(r, 1) for r in range(4) if r != test_cell[0]]}")
print(f"Box neighbors: {[(r, c) for r in range(2) for c in range(2) if (r,c) != test_cell]}")

print(f"\nCurrent row values: {simple_puzzle_4x4[0]}")
print(f"Values 1 and 4 already used in row → cannot assign 1 or 4")
print(f"Therefore, domain for cell {test_cell} = {{2, 3}}")

# Standard 9x9 Sudoku (easy puzzle)
easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

csp = SudokuCSP(easy_puzzle)
csp.print_puzzle("Easy Sudoku Puzzle")

# Show domain information
print("Domain Analysis:")
empty_cells = [(i, j) for i in range(9) for j in range(9) if easy_puzzle[i][j] == 0]
print(f"Total variables: 81")
print(f"Given values: {81 - len(empty_cells)}")
print(f"Empty cells (variables to solve): {len(empty_cells)}")
print(f"\nSample domains (first 5 empty cells):")
for cell in empty_cells[:5]:
    print(f"  Cell {cell}: domain = {sorted(csp.domains[cell])}")
```

### Reflection Questions

1. Sudoku has 81 variables (cells) but only 27 constraints (9 rows + 9 columns + 9 boxes). Explain why each constraint actually represents multiple binary constraints between pairs of cells. How many total binary constraints exist in a 9×9 Sudoku?

2. A Sudoku puzzle is only solvable if it has exactly one solution. From a CSP perspective, what would it mean for a puzzle to have zero solutions versus multiple solutions? How would you detect these conditions?

---

## Exercise 2: Naive Backtracking Search

### Description

The simplest CSP solving approach is naive backtracking: try values one at a time, recursively searching until you find a solution or prove none exists. This exercise implements basic backtracking without any intelligence—no domain reduction, no clever ordering, just brute-force search. Observing its performance establishes a baseline for understanding why sophisticated techniques matter.

### Key Concepts

- **Backtracking search**: Depth-first search that assigns variables one at a time and backtracks on failure
- **Recursive exploration**: Building up partial assignments and undoing them when constraints are violated
- **Search tree**: The implicit tree of all possible assignments explored during search
- **Branching factor**: The average number of choices at each decision point (up to 9 in Sudoku)
- **Thrashing**: Repeatedly failing and backtracking due to conflicts discovered late

### Task

Run the naive backtracking solver and observe:
- How many assignments it tries before finding the solution
- The order in which cells are selected
- How quickly the search tree explodes
- The difference between easy and harder puzzles

```python
import time

class NaiveBacktrackingSolver:
    """Basic backtracking without any optimizations"""
    
    def __init__(self, csp):
        self.csp = csp
        self.assignments = 0  # Counter for performance analysis
        self.backtracks = 0
    
    def solve(self):
        """Find solution using naive backtracking"""
        self.assignments = 0
        self.backtracks = 0
        start_time = time.time()
        
        result = self._backtrack()
        
        elapsed = time.time() - start_time
        return result, elapsed
    
    def _backtrack(self):
        """Recursive backtracking"""
        # Check if assignment is complete
        if self.csp.is_complete():
            return True
        
        # Select first unassigned variable (naive ordering)
        var = self._select_unassigned_variable()
        if var is None:
            return True
        
        row, col = var
        
        # Try each value in domain (naive ordering: 1-9)
        for value in range(1, 10):
            self.assignments += 1
            
            # Check if value is consistent with current assignment
            if self.csp.is_consistent(row, col, value):
                # Assign value
                self.csp.puzzle[row][col] = value
                
                # Recursive call
                result = self._backtrack()
                if result:
                    return True
                
                # Undo assignment (backtrack)
                self.csp.puzzle[row][col] = 0
                self.backtracks += 1
        
        return False
    
    def _select_unassigned_variable(self):
        """Select first empty cell (naive, left-to-right, top-to-bottom)"""
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                if self.csp.puzzle[i][j] == 0:
                    return (i, j)
        return None

# Test on easy puzzle
print("=== Naive Backtracking Search ===\n")

easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

csp = SudokuCSP(easy_puzzle)
csp.print_puzzle("Initial Puzzle")

solver = NaiveBacktrackingSolver(csp)
solved, elapsed = solver.solve()

if solved:
    csp.print_puzzle("Solved Puzzle")
    print(f"Success!")
    print(f"Assignments tried: {solver.assignments}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.4f} seconds")
else:
    print("No solution found")

print("\n--- Performance Analysis ---")
print(f"Search tree nodes explored: {solver.assignments}")
print(f"Wasted effort (backtracks): {solver.backtracks}")
print(f"Efficiency ratio: {solver.backtracks / max(1, solver.assignments):.2%} wasted")

# Compare with slightly harder puzzle
print("\n=== Testing on Medium Puzzle ===\n")

medium_puzzle = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0]
]

csp2 = SudokuCSP(medium_puzzle)
solver2 = NaiveBacktrackingSolver(csp2)
solved2, elapsed2 = solver2.solve()

if solved2:
    csp.print_puzzle("Solved Puzzle")
    print(f"Success!")
    print(f"Assignments tried: {solver2.assignments}")
    print(f"Backtracks: {solver2.backtracks}")
    print(f"Time: {elapsed2:.4f} seconds")
else:
    print("No solution found")

print("\n--- Performance Analysis ---")
print(f"Search tree nodes explored: {solver2.assignments}")
print(f"Wasted effort (backtracks): {solver2.backtracks}")
print(f"Efficiency ratio: {solver2.backtracks / max(1, solver2.assignments):.2%} wasted")
```

### Reflection Questions

3. Naive backtracking explores the search tree depth-first without any lookahead. Describe a scenario where this leads to significant wasted work—specifically, where the algorithm makes many assignments before discovering a conflict that could have been detected earlier.

4. The solver selects variables in a fixed order (left-to-right, top-to-bottom). Explain why this ordering is arbitrary and potentially inefficient. What information about the CSP state would be useful for choosing which variable to assign next?

5. Notice that the number of assignments grows dramatically for harder puzzles. Using the branching factor concept, explain why CSP search complexity is exponential. If each empty cell has an average of 5 possible values, approximately how many nodes would be explored in the worst case for a puzzle with 50 empty cells?

---

## Exercise 3: Forward Checking

### Description

Forward checking adds intelligence to backtracking by maintaining domain consistency. After each assignment, it immediately removes inconsistent values from the domains of unassigned neighbors. This detects conflicts much earlier than naive backtracking, often preventing futile search branches before they're explored. This is the first major optimization in practical CSP solvers.

### Key Concepts

- **Forward checking**: After assigning a variable, remove inconsistent values from neighboring variable domains
- **Domain reduction**: Shrinking the set of legal values based on current assignments
- **Early conflict detection**: Discovering dead ends before making additional assignments
- **Domain wipeout**: When a variable's domain becomes empty, indicating a conflict
- **Pruning**: Eliminating search branches that cannot lead to solutions

### Task

Run the forward checking solver and observe:
- How domains shrink after each assignment
- When domain wipeout occurs (triggering backtracking)
- The reduction in assignments compared to naive backtracking
- How much faster it solves the same puzzles

```python
from copy import deepcopy

class ForwardCheckingSolver:
    """Backtracking with forward checking"""
    
    def __init__(self, csp):
        self.csp = csp
        self.assignments = 0
        self.backtracks = 0
        self.domain_reductions = 0
    
    def solve(self):
        """Solve using forward checking"""
        self.assignments = 0
        self.backtracks = 0
        self.domain_reductions = 0
        start_time = time.time()
        
        # Initialize domains by removing values inconsistent with givens
        self._initial_forward_check()
        
        result = self._backtrack()
        elapsed = time.time() - start_time
        return result, elapsed
    
    def _initial_forward_check(self):
        """Remove values from domains based on initial assignments"""
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                if self.csp.puzzle[i][j] != 0:
                    value = self.csp.puzzle[i][j]
                    # Remove this value from all neighbors
                    for neighbor in self.csp.get_all_neighbors(i, j):
                        ni, nj = neighbor
                        if value in self.csp.domains[(ni, nj)]:
                            self.csp.domains[(ni, nj)].discard(value)
    
    def _backtrack(self):
        """Recursive backtracking with forward checking"""
        if self.csp.is_complete():
            return True
        
        # Select unassigned variable
        var = self._select_unassigned_variable()
        if var is None:
            return True
        
        row, col = var
        
        # Try each value in this variable's domain
        for value in list(self.csp.domains[(row, col)]):
            self.assignments += 1
            
            if self.csp.is_consistent(row, col, value):
                # Assign value
                self.csp.puzzle[row][col] = value
                
                # Save domains before forward checking
                saved_domains = deepcopy(self.csp.domains)
                
                # Forward check: remove value from neighbor domains
                if self._forward_check(row, col, value):
                    # No domain wipeout, continue search
                    result = self._backtrack()
                    if result:
                        return True
                
                # Restore domains and undo assignment
                self.csp.domains = saved_domains
                self.csp.puzzle[row][col] = 0
                self.backtracks += 1
        
        return False
    
    def _forward_check(self, row, col, value):
        """
        Remove value from all neighbor domains.
        Returns False if any domain becomes empty (wipeout).
        """
        for neighbor in self.csp.get_all_neighbors(row, col):
            ni, nj = neighbor
            if self.csp.puzzle[ni][nj] == 0:  # Unassigned
                if value in self.csp.domains[(ni, nj)]:
                    self.csp.domains[(ni, nj)].discard(value)
                    self.domain_reductions += 1
                    
                    # Check for domain wipeout
                    if len(self.csp.domains[(ni, nj)]) == 0:
                        return False
        
        return True
    
    def _select_unassigned_variable(self):
        """Select first unassigned variable (naive ordering)"""
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                if self.csp.puzzle[i][j] == 0:
                    return (i, j)
        return None

# Test forward checking
print("=== Forward Checking ===\n")

easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

csp = SudokuCSP(easy_puzzle)
print("Domains after initialization:")
empty_cells = [(i, j) for i in range(9) for j in range(9) if easy_puzzle[i][j] == 0]
print(f"Sample cell (0,2): domain = {sorted(csp.domains[(0, 2)])}")

solver = ForwardCheckingSolver(csp)
solved, elapsed = solver.solve()

if solved:
    csp.print_puzzle("Solved with Forward Checking")
    print(f"Assignments: {solver.assignments}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Domain reductions: {solver.domain_reductions}")
    print(f"Time: {elapsed:.4f} seconds")

# Compare with naive backtracking
print("\n--- Comparison: Forward Checking vs Naive Backtracking ---\n")

csp_naive = SudokuCSP(easy_puzzle)
solver_naive = NaiveBacktrackingSolver(csp_naive)
_, time_naive = solver_naive.solve()

print(f"{'Metric':<25} {'Naive':<15} {'Forward Checking':<15} {'Improvement'}")
print("─" * 70)
print(f"{'Assignments':<25} {solver_naive.assignments:<15} {solver.assignments:<15} {solver_naive.assignments/max(1,solver.assignments):.1f}x fewer")
print(f"{'Time (seconds)':<25} {time_naive:<15.4f} {elapsed:<15.4f} {time_naive/max(0.0001,elapsed):.1f}x faster")
print(f"{'Backtracks':<25} {solver_naive.backtracks:<15} {solver.backtracks:<15} {solver_naive.backtracks/max(1,solver.backtracks):.1f}x fewer")
```

### Reflection Questions

6. Forward checking maintains arc consistency between the current variable and its neighbors. Explain what happens when a neighbor's domain becomes empty (domain wipeout). Why is this more efficient than discovering the conflict through multiple future assignments?

7. Notice the "domain reductions" metric. Each reduction eliminates a potential branch of the search tree. Calculate approximately how many search nodes were pruned by forward checking in the example. How does this relate to the speedup observed?

---

## Exercise 4: Constraint Propagation (Arc Consistency AC-3)

### Description

While forward checking only enforces consistency between the current variable and its neighbors, arc consistency (AC-3) propagates constraints more thoroughly. It ensures that for every variable X and neighbor Y, each value in X's domain has at least one compatible value in Y's domain. This powerful preprocessing can solve easy puzzles without any search at all.

### Key Concepts

- **Arc consistency**: For every pair of constrained variables (X, Y), each value in X's domain must have a compatible value in Y's domain
- **AC-3 algorithm**: A queue-based algorithm that iteratively enforces arc consistency across all constraint arcs
- **Constraint propagation**: Automatically deriving new constraints from existing ones
- **Fixed point**: When no more domain reductions are possible through consistency checking
- **Inference**: Deducing new facts (reduced domains) from current knowledge

### Task

Run the AC-3 implementation and observe:
- How constraints propagate through the puzzle
- Which cells get solved by pure inference (no search)
- The queue operations showing which arcs are processed
- How AC-3 reduces domains more aggressively than forward checking

```python
from collections import deque

class AC3Solver:
    """Constraint propagation using AC-3 algorithm"""
    
    def __init__(self, csp):
        self.csp = csp
        self.assignments = 0
        self.backtracks = 0
        self.arc_revisions = 0
        self.propagation_rounds = 0
    
    def solve(self):
        """Solve using AC-3 + backtracking"""
        self.assignments = 0
        self.backtracks = 0
        self.arc_revisions = 0
        self.propagation_rounds = 0
        
        start_time = time.time()
        
        # Run AC-3 as preprocessing
        print("Running AC-3 constraint propagation...")
        consistent = self.ac3()
        
        if not consistent:
            print("Puzzle is inconsistent!")
            return False, time.time() - start_time
        
        print(f"AC-3 completed: {self.arc_revisions} arc revisions\n")
        
        # Check if AC-3 alone solved it
        if self.csp.is_complete():
            print("Solved by AC-3 alone (no search needed)!")
            elapsed = time.time() - start_time
            return True, elapsed
        
        # Need search for remaining cells
        print("AC-3 insufficient, starting backtracking search...\n")
        result = self._backtrack()
        elapsed = time.time() - start_time
        return result, elapsed
    
    def ac3(self, arcs=None):
        """
        AC-3 algorithm for enforcing arc consistency.
        Returns False if inconsistency detected, True otherwise.
        """
        # Initialize queue with all arcs
        if arcs is None:
            queue = deque()
            for i in range(self.csp.size):
                for j in range(self.csp.size):
                    if self.csp.puzzle[i][j] == 0:
                        for neighbor in self.csp.get_all_neighbors(i, j):
                            queue.append(((i, j), neighbor))
        else:
            queue = deque(arcs)
        
        while queue:
            self.propagation_rounds += 1
            (Xi, Xj) = queue.popleft()
            
            # Revise domain of Xi based on Xj
            if self._revise(Xi, Xj):
                # Domain of Xi was reduced
                if len(self.csp.domains[Xi]) == 0:
                    return False  # Inconsistency detected
                
                # If domain changed, re-check all neighbors of Xi
                for Xk in self.csp.get_all_neighbors(*Xi):
                    if Xk != Xj:
                        queue.append((Xk, Xi))
        
        return True
    
    def _revise(self, Xi, Xj):
        """
        Make Xi arc-consistent with Xj.
        Remove values from Xi's domain that have no support in Xj's domain.
        Returns True if domain was revised.
        """
        self.arc_revisions += 1
        revised = False
        
        # If Xj is assigned, remove that value from Xi's domain
        xi_row, xi_col = Xi
        xj_row, xj_col = Xj
        
        if self.csp.puzzle[xj_row][xj_col] != 0:
            # Xj is assigned - remove its value from Xi
            assigned_value = self.csp.puzzle[xj_row][xj_col]
            if assigned_value in self.csp.domains[Xi]:
                self.csp.domains[Xi].discard(assigned_value)
                revised = True
        else:
            # Xj is unassigned - check for support
            values_to_remove = set()
            for value in self.csp.domains[Xi]:
                # Check if value has support in Xj's domain
                has_support = False
                for xj_value in self.csp.domains[Xj]:
                    if value != xj_value:  # Different values (constraint)
                        has_support = True
                        break
                
                if not has_support:
                    values_to_remove.add(value)
                    revised = True
            
            self.csp.domains[Xi] -= values_to_remove
        
        # Singleton domain: assign it immediately
        if len(self.csp.domains[Xi]) == 1:
            value = list(self.csp.domains[Xi])[0]
            if self.csp.puzzle[xi_row][xi_col] == 0:
                self.csp.puzzle[xi_row][xi_col] = value
                revised = True
        
        return revised
    
    def _backtrack(self):
        """Backtracking with AC-3 after each assignment"""
        if self.csp.is_complete():
            return True
        
        var = self._select_unassigned_variable()
        if var is None:
            return True
        
        row, col = var
        
        for value in list(self.csp.domains[(row, col)]):
            self.assignments += 1
            
            if self.csp.is_consistent(row, col, value):
                self.csp.puzzle[row][col] = value
                saved_domains = deepcopy(self.csp.domains)
                
                # Update domain and run AC-3
                self.csp.domains[(row, col)] = {value}
                
                # Propagate constraints
                arcs = [((n[0], n[1]), (row, col)) for n in self.csp.get_all_neighbors(row, col)]
                if self.ac3(arcs):
                    result = self._backtrack()
                    if result:
                        return True
                
                self.csp.domains = saved_domains
                self.csp.puzzle[row][col] = 0
                self.backtracks += 1
        
        return False
    
    def _select_unassigned_variable(self):
        """Select unassigned variable"""
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                if self.csp.puzzle[i][j] == 0:
                    return (i, j)
        return None

# Test AC-3
print("=== AC-3 Constraint Propagation ===\n")

easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

csp = SudokuCSP(easy_puzzle)
csp.print_puzzle("Initial Puzzle")

# Show domains before AC-3
print("Sample domains before AC-3:")
empty_before = [(i, j) for i in range(9) for j in range(9) if easy_puzzle[i][j] == 0]
for cell in empty_before[:3]:
    print(f"  Cell {cell}: {sorted(csp.domains[cell])}")

solver = AC3Solver(csp)
solved, elapsed = solver.solve()

if solved:
    csp.print_puzzle("Solved with AC-3")
    print(f"Arc revisions: {solver.arc_revisions}")
    print(f"Assignments needed: {solver.assignments}")
    print(f"Time: {elapsed:.4f} seconds")

# Show the power of AC-3
print("\n--- AC-3 Performance Analysis ---")
print(f"Propagation rounds: {solver.propagation_rounds}")
print(f"Search assignments: {solver.assignments}")
if solver.assignments == 0:
    print("✓ Puzzle solved by pure constraint propagation!")
else:
    print(f"Remaining search needed: {solver.assignments} assignments")
```

### Reflection Questions

8. The AC-3 algorithm uses a queue to process arcs. Explain why arcs must be re-added to the queue when domains change. What would happen if we only processed each arc once without re-queuing?

9. Some easy Sudoku puzzles can be solved by AC-3 alone without any search. Explain what this reveals about the puzzle's structure. What property must a puzzle have for AC-3 to solve it completely through constraint propagation?

---

## Exercise 5: Variable and Value Ordering Heuristics

### Description

The order in which we choose variables and values dramatically affects search efficiency. This exercise implements two key heuristics: Minimum Remaining Values (MRV) for variable selection and Least Constraining Value (LCV) for value ordering. These heuristics embody the "fail-first" and "succeed-first" principles of CSP solving.

### Key Concepts

- **Minimum Remaining Values (MRV)**: Choose the variable with the fewest legal values (most constrained)
- **Degree heuristic**: Tie-breaker that chooses the variable involved in most constraints
- **Least Constraining Value (LCV)**: Order values by how many options they leave for neighbors
- **Fail-first principle**: Detect failures as early as possible by tackling hard decisions first
- **Succeed-first principle**: Make choices that preserve maximum flexibility for future decisions

### Task

Run the heuristic solver and observe:
- Which cells are selected first (highly constrained ones)
- How value ordering affects backtracking
- The dramatic reduction in assignments compared to naive ordering
- Why "most constrained first" leads to faster solving

```python
class HeuristicSolver:
    """Backtracking with MRV and LCV heuristics"""
    
    def __init__(self, csp):
        self.csp = csp
        self.assignments = 0
        self.backtracks = 0
        self.mrv_selections = []
    
    def solve(self):
        """Solve using intelligent heuristics"""
        self.assignments = 0
        self.backtracks = 0
        self.mrv_selections = []
        
        start_time = time.time()
        
        # Initial AC-3
        ac3_solver = AC3Solver(self.csp)
        if not ac3_solver.ac3():
            return False, 0
        
        result = self._backtrack()
        elapsed = time.time() - start_time
        return result, elapsed
    
    def _select_unassigned_variable_mrv(self):
        """
        Select variable with Minimum Remaining Values.
        Ties broken by degree heuristic.
        """
        min_domain_size = float('inf')
        best_vars = []
        
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                if self.csp.puzzle[i][j] == 0:
                    domain_size = len(self.csp.domains[(i, j)])
                    
                    if domain_size < min_domain_size:
                        min_domain_size = domain_size
                        best_vars = [(i, j)]
                    elif domain_size == min_domain_size:
                        best_vars.append((i, j))
        
        if not best_vars:
            return None
        
        # Tie-breaker: degree heuristic (most constraining variable)
        if len(best_vars) > 1:
            max_degree = -1
            best_var = best_vars[0]
            for var in best_vars:
                # Count unassigned neighbors
                degree = sum(1 for n in self.csp.get_all_neighbors(*var) 
                           if self.csp.puzzle[n[0]][n[1]] == 0)
                if degree > max_degree:
                    max_degree = degree
                    best_var = var
            return best_var
        
        self.mrv_selections.append((best_vars[0], min_domain_size))
        return best_vars[0]
    
    def _order_domain_values(self, var):
        """
        Order values using Least Constraining Value heuristic.
        Prefer values that rule out fewer choices for neighbors.
        """
        row, col = var
        domain = list(self.csp.domains[var])
        
        # Count how many neighbor values each value would eliminate
        value_constraints = []
        for value in domain:
            eliminated_count = 0
            for neighbor in self.csp.get_all_neighbors(row, col):
                if self.csp.puzzle[neighbor[0]][neighbor[1]] == 0:
                    if value in self.csp.domains[neighbor]:
                        eliminated_count += 1
            value_constraints.append((value, eliminated_count))
        
        # Sort by ascending constraint count (least constraining first)
        value_constraints.sort(key=lambda x: x[1])
        return [v for v, _ in value_constraints]
    
    def _backtrack(self):
        """Backtracking with MRV and LCV"""
        if self.csp.is_complete():
            return True
        
        # MRV variable selection
        var = self._select_unassigned_variable_mrv()
        if var is None:
            return True
        
        row, col = var
        
        # LCV value ordering
        ordered_values = self._order_domain_values(var)
        
        for value in ordered_values:
            self.assignments += 1
            
            if self.csp.is_consistent(row, col, value):
                self.csp.puzzle[row][col] = value
                saved_domains = deepcopy(self.csp.domains)
                
                # Forward check
                self.csp.domains[(row, col)] = {value}
                if self._forward_check(row, col, value):
                    result = self._backtrack()
                    if result:
                        return True
                
                self.csp.domains = saved_domains
                self.csp.puzzle[row][col] = 0
                self.backtracks += 1
        
        return False
    
    def _forward_check(self, row, col, value):
        """Forward checking"""
        for neighbor in self.csp.get_all_neighbors(row, col):
            ni, nj = neighbor
            if self.csp.puzzle[ni][nj] == 0:
                if value in self.csp.domains[(ni, nj)]:
                    self.csp.domains[(ni, nj)].discard(value)
                    if len(self.csp.domains[(ni, nj)]) == 0:
                        return False
        return True

# Test heuristic solver
print("=== Variable and Value Ordering Heuristics ===\n")

medium_puzzle = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0]
]

csp = SudokuCSP(medium_puzzle)
csp.print_puzzle("Medium Difficulty Puzzle")

solver = HeuristicSolver(csp)
solved, elapsed = solver.solve()

if solved:
    csp.print_puzzle("Solved with Heuristics")
    print(f"Assignments: {solver.assignments}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.4f} seconds")
    
    print("\n--- MRV Selection Trace (first 5) ---")
    for i, (cell, domain_size) in enumerate(solver.mrv_selections[:5]):
        print(f"{i+1}. Selected cell {cell} with domain size {domain_size}")

# Compare all approaches
print("\n=== Performance Comparison on Medium Puzzle ===\n")

approaches = {
    'Naive Backtracking': NaiveBacktrackingSolver,
    'Forward Checking': ForwardCheckingSolver,
    'MRV + LCV Heuristics': HeuristicSolver
}

for name, SolverClass in approaches.items():
    csp_test = SudokuCSP(medium_puzzle)
    solver_test = SolverClass(csp_test)
    solved_test, time_test = solver_test.solve()
    print(f"{name}:")
    print(f"  Assignments: {solver_test.assignments}")
    print(f"  Time: {time_test:.4f}s")
    print()
```

### Reflection Questions

10. The MRV heuristic chooses the "most constrained" variable first, which seems counterintuitive—why tackle the hardest decisions first? Explain the "fail-first" principle and how detecting failures early reduces total search effort.

11. The LCV heuristic orders values by how little they constrain neighbors. Describe the philosophical difference between MRV (fail-first for variables) and LCV (succeed-first for values). Why do these opposite strategies work well together?

12. The degree heuristic breaks ties in MRV by choosing variables with the most unassigned neighbors. Explain the reasoning: why would constraining more variables make a variable selection better? Consider both immediate effects and future search reduction.

---

## Exercise 6: Comparing CSP Techniques on Hard Puzzles

### Description

To truly understand each technique's strengths, we need to test them on genuinely difficult puzzles. This exercise benchmarks all approaches on hard Sudoku instances, revealing which optimizations matter most as problem difficulty increases.

### Key Concepts

- **Problem difficulty**: Measured by minimum number of givens or search tree size
- **Scaling behavior**: How algorithm performance changes with problem difficulty
- **Algorithmic bottlenecks**: Which operations dominate runtime for each approach
- **Practical complexity**: Real-world performance beyond theoretical analysis
- **Algorithm selection**: Choosing techniques based on problem characteristics

### Task

Run the comprehensive benchmark and observe:
- How performance gaps widen on harder puzzles
- Which techniques scale better
- The relationship between initial givens and solving difficulty
- When sophisticated techniques become necessary

```python
def benchmark_solvers(puzzles, puzzle_names):
    """Compare all solving techniques on multiple puzzles"""
    
    solvers = {
        'Naive': NaiveBacktrackingSolver,
        'Forward Checking': ForwardCheckingSolver,
        'AC-3': AC3Solver,
        'Heuristics (MRV+LCV)': HeuristicSolver
    }
    
    results = {name: [] for name in solvers}
    
    for puzzle_name, puzzle in zip(puzzle_names, puzzles):
        print(f"\n{'='*60}")
        print(f"Testing: {puzzle_name}")
        print(f"{'='*60}\n")
        
        # Count givens
        givens = sum(1 for row in puzzle for val in row if val != 0)
        empty = 81 - givens
        print(f"Givens: {givens}, Empty cells: {empty}\n")
        
        for solver_name, SolverClass in solvers.items():
            print(f"Running {solver_name}...", end=" ")
            
            csp = SudokuCSP(puzzle)
            solver = SolverClass(csp)
            
            try:
                solved, elapsed = solver.solve()
                
                if solved:
                    print(f"✓ {elapsed:.4f}s, {solver.assignments} assignments")
                    results[solver_name].append({
                        'puzzle': puzzle_name,
                        'time': elapsed,
                        'assignments': solver.assignments,
                        'solved': True
                    })
                else:
                    print("✗ Failed")
                    results[solver_name].append({
                        'puzzle': puzzle_name,
                        'solved': False
                    })
            except Exception as e:
                print(f"✗ Error: {e}")
                results[solver_name].append({
                    'puzzle': puzzle_name,
                    'solved': False
                })
    
    return results

# Define test puzzles of increasing difficulty
easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

medium_puzzle = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0]
]

hard_puzzle = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 5],
    [0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 7, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 1, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 7, 3],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 9]
]

print("=== Comprehensive Performance Benchmark ===")

puzzles = [easy_puzzle, medium_puzzle, hard_puzzle]
names = ["Easy (36 givens)", "Medium (27 givens)", "Hard (21 givens)"]

results = benchmark_solvers(puzzles, names)

# Summary table
print("\n" + "="*60)
print("SUMMARY: Average Assignments by Difficulty")
print("="*60 + "\n")

print(f"{'Solver':<25} {'Easy':<12} {'Medium':<12} {'Hard':<12}")
print("─" * 60)

for solver_name in results:
    row = f"{solver_name:<25}"
    for puzzle_name in names:
        matching = [r for r in results[solver_name] if r['puzzle'] == puzzle_name and r['solved']]
        if matching:
            avg_assigns = matching[0]['assignments']
            row += f"{avg_assigns:<12}"
        else:
            row += f"{'FAILED':<12}"
    print(row)
```

### Reflection Questions

13. Observe how the performance gap between naive backtracking and intelligent techniques grows with puzzle difficulty. Explain why the benefit of heuristics and constraint propagation is more pronounced on harder puzzles. What does this reveal about the structure of the search space?

14. The "hard" puzzle has only 21 givens compared to 36 for the easy puzzle. Discuss why fewer givens doesn't always mean harder—the pattern and distribution of givens matters. What makes a Sudoku puzzle truly "hard" from a CSP perspective?

15. If you were building a production Sudoku solver, which combination of techniques would you choose? Consider the trade-off between implementation complexity, computational cost, and solving power. Would you use the same configuration for all puzzles or adapt based on puzzle characteristics?

---

## Exercise 7: Local Search Methods and Their Limitations

### Description

Unlike systematic search (backtracking), local search starts with a complete (possibly invalid) assignment and iteratively improves it. This exercise implements a simple local search approach for Sudoku and reveals why local search struggles with constraint satisfaction despite succeeding in optimization problems. Understanding these limitations clarifies when different paradigms apply.

### Key Concepts

- **Local search**: Iterative improvement starting from a complete assignment
- **Objective function**: A measure to minimize (e.g., number of constraint violations)
- **Local optimum**: A state where no small change improves the objective
- **Plateaus**: Regions of the search space where many states have equal objective values
- **Constraint satisfaction vs. optimization**: CSPs require exact solutions; optimization accepts approximate ones

### Task

Run the local search implementation and observe:
- How it gets stuck in local optima
- The difference between reducing conflicts and eliminating them
- Why even "close" solutions aren't acceptable for CSPs
- The contrast with backtracking's systematic guarantees

```python
import random

class LocalSearchSolver:
    """Min-conflicts local search for Sudoku"""
    
    def __init__(self, csp, max_iterations=10000):
        self.csp = csp
        self.max_iterations = max_iterations
        self.iterations = 0
        self.conflict_history = []
    
    def solve(self):
        """Attempt to solve using local search"""
        self.iterations = 0
        self.conflict_history = []
        
        start_time = time.time()
        
        # Initialize: fill all empty cells with random valid values
        self._random_initialization()
        
        initial_conflicts = self._count_conflicts()
        print(f"Initial conflicts: {initial_conflicts}\n")
        
        # Iterative improvement
        for i in range(self.max_iterations):
            self.iterations += 1
            
            conflicts = self._count_conflicts()
            self.conflict_history.append(conflicts)
            
            if conflicts == 0:
                elapsed = time.time() - start_time
                print(f"✓ Solution found after {i+1} iterations!")
                return True, elapsed
            
            if i % 100 == 0:
                print(f"Iteration {i}: {conflicts} conflicts")
            
            # Select variable with conflicts
            conflicted = self._get_conflicted_variables()
            if not conflicted:
                break
            
            var = random.choice(conflicted)
            
            # Try to reduce conflicts by changing this variable's value
            self._min_conflicts_move(var)
        
        elapsed = time.time() - start_time
        final_conflicts = self._count_conflicts()
        print(f"\n✗ Failed to find solution")
        print(f"Final conflicts: {final_conflicts}")
        return False, elapsed
    
    def _random_initialization(self):
        """Fill empty cells with random values"""
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                if self.csp.puzzle[i][j] == 0:
                    # Random value from domain
                    self.csp.puzzle[i][j] = random.randint(1, 9)
    
    def _count_conflicts(self):
        """Count total constraint violations"""
        conflicts = 0
        
        # Check rows
        for i in range(self.csp.size):
            row_vals = [self.csp.puzzle[i][j] for j in range(self.csp.size)]
            conflicts += len(row_vals) - len(set(row_vals))
        
        # Check columns
        for j in range(self.csp.size):
            col_vals = [self.csp.puzzle[i][j] for i in range(self.csp.size)]
            conflicts += len(col_vals) - len(set(col_vals))
        
        # Check boxes
        for box_i in range(3):
            for box_j in range(3):
                box_vals = []
                for i in range(box_i * 3, (box_i + 1) * 3):
                    for j in range(box_j * 3, (box_j + 1) * 3):
                        box_vals.append(self.csp.puzzle[i][j])
                conflicts += len(box_vals) - len(set(box_vals))
        
        return conflicts
    
    def _get_conflicted_variables(self):
        """Get list of variables involved in conflicts"""
        conflicted = []
        
        for i in range(self.csp.size):
            for j in range(self.csp.size):
                # Skip given cells
                if (i, j) not in self.csp.domains or len(self.csp.domains[(i, j)]) == 1:
                    if self.csp.puzzle[i][j] != 0 and (i, j) in [(r, c) for r in range(9) for c in range(9) if self.csp.puzzle[r][c] != 0]:
                        continue
                
                # Check if this cell is in conflict
                value = self.csp.puzzle[i][j]
                if self._has_conflict(i, j, value):
                    conflicted.append((i, j))
        
        return conflicted
    
    def _has_conflict(self, row, col, value):
        """Check if this cell/value creates conflicts"""
        # Check row
        for c in range(self.csp.size):
            if c != col and self.csp.puzzle[row][c] == value:
                return True
        
        # Check column
        for r in range(self.csp.size):
            if r != row and self.csp.puzzle[r][col] == value:
                return True
        
        # Check box
        box_row, box_col = (row // 3) * 3, (col // 3) * 3
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if (i, j) != (row, col) and self.csp.puzzle[i][j] == value:
                    return True
        
        return False
    
    def _min_conflicts_move(self, var):
        """Change variable to value with minimum conflicts"""
        row, col = var
        current_value = self.csp.puzzle[row][col]
        
        # Try all values and count conflicts
        min_conflicts = float('inf')
        best_value = current_value
        
        for value in range(1, 10):
            self.csp.puzzle[row][col] = value
            conflicts = self._count_conflicts()
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_value = value
        
        self.csp.puzzle[row][col] = best_value

# Test local search
print("=== Local Search (Min-Conflicts) ===\n")

easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

csp = SudokuCSP(easy_puzzle)
csp.print_puzzle("Attempting with Local Search")

solver = LocalSearchSolver(csp, max_iterations=1000)
solved, elapsed = solver.solve()

print(f"\nTime: {elapsed:.4f}s")
print(f"Iterations: {solver.iterations}")

# Show conflict trajectory
if len(solver.conflict_history) > 10:
    print("\nConflict trajectory (every 100 iterations):")
    for i in range(0, min(len(solver.conflict_history), 1000), 100):
        print(f"  Iteration {i}: {solver.conflict_history[i]} conflicts")

# Compare with systematic search
print("\n--- Comparison: Local Search vs. Backtracking ---\n")

csp_bt = SudokuCSP(easy_puzzle)
solver_bt = HeuristicSolver(csp_bt)
solved_bt, time_bt = solver_bt.solve()

print(f"Local Search: {' Solved' if solved else 'Failed'} in {elapsed:.4f}s, {solver.iterations} iterations")
print(f"Backtracking: {'Solved' if solved_bt else 'Failed'} in {time_bt:.4f}s, {solver_bt.assignments} assignments")
print(f"\nConclusion: Local search struggles with CSPs that require exact solutions!")
```

### Reflection Questions

16. The min-conflicts heuristic chooses values that minimize immediate conflicts. Compare this greedy, local decision-making to backtracking's systematic exploration. Why does backtracking's ability to undo multiple decisions give it an advantage for CSPs?

17. Local search performs well for some CSPs (like N-queens) but poorly for Sudoku. Hypothesize about what properties make a CSP amenable to local search versus requiring systematic search. Consider factors like constraint density, solution density, and landscape topology.

---

## Exercise 7: SMT Solving with Z3

### Description

While the previous exercises explored search-based CSP solving techniques, modern constraint solvers can use fundamentally different approaches. Z3 is an SMT (Satisfiability Modulo Theories) solver that uses sophisticated reasoning techniques including DPLL (Davis-Putnam-Logemann-Loveland), conflict-driven clause learning, and theory-specific decision procedures. This exercise demonstrates how declarative constraint specification with Z3 can solve Sudoku efficiently without explicit search code.

### Key Concepts

- **SMT Solving**: Determining satisfiability of logical formulas with respect to background theories (integers, arrays, etc.)
- **Declarative specification**: Describing what constraints must hold rather than how to find solutions
- **SAT/SMT techniques**: DPLL search with clause learning, backjumping, and unit propagation
- **Distinct constraints**: Built-in support for all-different constraints common in CSPs
- **Theory reasoning**: Leveraging specialized decision procedures for integer arithmetic

### Task

Run the Z3-based solver below and compare its approach to the search techniques from previous exercises. Install Z3 with: `pip install z3-solver`

```python
from z3 import *
import time

def solve_sudoku_z3(puzzle):
    """
    Solve Sudoku using Z3 SMT solver.
    
    Rather than implementing search ourselves, we declare constraints
    and let Z3's sophisticated solver find a satisfying assignment.
    """
    
    # Create 9x9 grid of integer variables
    grid = [[Int(f"cell_{i}_{j}") for j in range(9)] for i in range(9)]
    
    s = Solver()
    
    # Constraint 1: Each cell is between 1 and 9
    print("Adding domain constraints...")
    for i in range(9):
        for j in range(9):
            s.add(And(grid[i][j] >= 1, grid[i][j] <= 9))
    
    # Constraint 2: Rows are distinct (no duplicates)
    print("Adding row constraints...")
    for i in range(9):
        s.add(Distinct(grid[i]))
    
    # Constraint 3: Columns are distinct
    print("Adding column constraints...")
    for j in range(9):
        s.add(Distinct([grid[i][j] for i in range(9)]))
    
    # Constraint 4: 3x3 boxes are distinct
    print("Adding box constraints...")
    for box_i in range(3):
        for box_j in range(3):
            cells = [grid[i][j] 
                     for i in range(box_i*3, box_i*3+3)
                     for j in range(box_j*3, box_j*3+3)]
            s.add(Distinct(cells))
    
    # Constraint 5: Fix given cells
    print("Adding given values...")
    givens = 0
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                s.add(grid[i][j] == puzzle[i][j])
                givens += 1
    
    print(f"Total givens: {givens}")
    print(f"Total constraints added: {len(s.assertions())}")
    print("\nSolving with Z3...")
    
    # Solve
    start = time.time()
    result = s.check()
    elapsed = time.time() - start
    
    if result == sat:
        m = s.model()
        print(f"✓ Solution found in {elapsed:.4f}s")
        
        # Extract solution
        solution = [[0]*9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                solution[i][j] = m.evaluate(grid[i][j]).as_long()
        
        return solution, elapsed, True
    else:
        print(f"✗ No solution exists")
        return None, elapsed, False

def print_sudoku(puzzle, title="Sudoku"):
    """Pretty print the puzzle"""
    print(f"\n{title}")
    print("─" * 25)
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("─" * 25)
        row_str = ""
        for j in range(9):
            if j > 0 and j % 3 == 0:
                row_str += "│ "
            val = puzzle[i][j]
            row_str += str(val) if val != 0 else "."
            row_str += " "
        print(row_str)
    print()

# Test on puzzles of varying difficulty
print("=== Z3 SMT Solver for Sudoku ===\n")

# Easy puzzle
easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

print_sudoku(easy_puzzle, "Easy Puzzle (36 givens)")
solution, time_easy, solved = solve_sudoku_z3(easy_puzzle)
if solved:
    print_sudoku(solution, "Solution")

# Hard puzzle
hard_puzzle = [
    [0, 0, 0, 0, 0, 0, 6, 8, 0],
    [0, 0, 0, 0, 7, 3, 0, 0, 9],
    [3, 0, 9, 0, 0, 0, 0, 4, 5],
    [4, 9, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 3, 0, 5, 0, 9, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 3, 6],
    [9, 6, 0, 0, 0, 0, 3, 0, 8],
    [7, 0, 0, 6, 8, 0, 0, 0, 0],
    [0, 2, 8, 0, 0, 0, 0, 0, 0]
]

print("\n" + "="*50 + "\n")
print_sudoku(hard_puzzle, "Hard Puzzle (21 givens)")
solution, time_hard, solved = solve_sudoku_z3(hard_puzzle)
if solved:
    print_sudoku(solution, "Solution")

# Analysis
print("\n=== Z3 Performance Summary ===\n")
print(f"Easy puzzle: {time_easy:.4f}s")
print(f"Hard puzzle: {time_hard:.4f}s")
print(f"Ratio: {time_hard/time_easy:.2f}x")

print("\n--- Key Observations ---\n")
print("1. Z3 handles both puzzles efficiently despite difficulty difference")
print("2. Declarative specification: we state constraints, not search strategy")
print("3. Z3 uses sophisticated techniques (clause learning, backjumping)")
print("4. Built-in Distinct() constraint is optimized for all-different")
print("5. No explicit variable ordering or value selection needed")
```

### Reflection Questions

18. Z3 uses a declarative approach where you specify constraints and let the solver find solutions. Compare this to the procedural backtracking approaches from earlier exercises. What are the advantages and disadvantages of each paradigm for solving CSPs?

19. The Distinct() constraint in Z3 is a built-in primitive that handles "all different" efficiently. Earlier exercises checked constraints pairwise manually. Explain how a solver could optimize the Distinct() constraint internally using techniques beyond pairwise checking. Consider arc consistency and conflict analysis.

20. Z3 uses techniques like conflict-driven clause learning (CDCL) which learns from failures to avoid repeating similar mistakes. Compare this to plain backtracking which forgets why branches failed. How does learning from conflicts relate to the constraint propagation and heuristics you've explored in previous exercises?

---

## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including all code from this lab and:

Create `lab_ch6_results.md`:

```markdown
# Names: Your names here
# Lab: lab3 (CSP)
# Date: Today's date
```

And your answers to all reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the fourth lab!

---