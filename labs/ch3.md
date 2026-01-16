# Chapter 3: Solving Problems by Searching - Lab
**Textbook Reference:** Russell & Norvig, "Artificial Intelligence: A Modern Approach" - Chapter 3

## Learning Objectives

By the end of this lab, you will be able to:

1. **Identify** the key components of a problem formulation (initial state, actions, transition model, goal test, path cost)
2. **Distinguish** between uninformed and informed search strategies based on their use of problem-specific knowledge
3. **Explain** how different search algorithms explore the search space and maintain their frontier data structures

## Lab Overview

This lab uses an **observation-based learning approach**: you will run pre-written code and carefully analyze what happens, rather than implementing algorithms from scratch. This approach helps you understand the theoretical concepts by seeing them in action.

**Domain:**
We'll use classic AI problems including:
- The 8-puzzle (sliding tile puzzle)
- Grid-based pathfinding
- Simple route-finding problems

Each exercise demonstrates core concepts from Chapter 3 through runnable code with detailed output.

---

I suggest running the following commands from your base user directory if necessary:


```bash
mkdir cs430 
cd cs430 
uv init 
uv sync
source .venv/bin/activate
touch search.ipynb
```

The last command will create a file such as `search.ipynb`. 

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

## Exercise 1: Problem Formulation

### Description
This exercise demonstrates how to formulate a search problem by defining its five key components. Understanding problem formulation is the foundation for applying any search algorithm. We'll use the 8-puzzle as our example domain.

### Key Concepts
- **State**: A configuration of the environment (e.g., positions of tiles in the 8-puzzle)
- **Actions**: Possible moves from a given state (e.g., slide tile up, down, left, right)
- **Transition Model**: The result of applying an action to a state
- **Goal Test**: A function that determines if a state is the goal
- **Path Cost**: The cost of reaching a state (e.g., number of moves)

### Task
Run the code below and observe how a problem is formally defined. Pay attention to:
- How states are represented (what data structure is used?)
- How the `get_actions()` method determines valid moves
- What information the `result()` method returns
- How the goal test works

```python
class EightPuzzle:
    """Represents an 8-puzzle problem."""
    
    def __init__(self, initial_state, goal_state):
        self.initial = initial_state
        self.goal = goal_state
    
    def get_actions(self, state):
        """Returns list of valid actions from this state."""
        actions = []
        blank_pos = state.index(0)  # Find the blank tile
        row, col = blank_pos // 3, blank_pos % 3
        
        if row > 0: actions.append('UP')
        if row < 2: actions.append('DOWN')
        if col > 0: actions.append('LEFT')
        if col < 2: actions.append('RIGHT')
        
        return actions
    
    def result(self, state, action):
        """Returns the state that results from executing action."""
        new_state = list(state)
        blank_pos = state.index(0)
        row, col = blank_pos // 3, blank_pos % 3
        
        # Calculate new position based on action
        if action == 'UP':
            new_pos = (row - 1) * 3 + col
        elif action == 'DOWN':
            new_pos = (row + 1) * 3 + col
        elif action == 'LEFT':
            new_pos = row * 3 + (col - 1)
        elif action == 'RIGHT':
            new_pos = row * 3 + (col + 1)
        
        # Swap blank with tile in new position
        new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
        return tuple(new_state)
    
    def is_goal(self, state):
        """Tests if state is the goal."""
        return state == self.goal
    
    def display_state(self, state):
        """Pretty print a state."""
        for i in range(0, 9, 3):
            print(f"  {state[i]} {state[i+1]} {state[i+2]}")

# Create a problem instance
initial = (1, 2, 3, 4, 0, 5, 6, 7, 8)  # 0 represents blank
goal = (0, 1, 2, 3, 4, 5, 6, 7, 8)

problem = EightPuzzle(initial, goal)

print("PROBLEM FORMULATION DEMONSTRATION")
print("=" * 50)
print("\nInitial State:")
problem.display_state(initial)

print("\nGoal State:")
problem.display_state(goal)

print("\nAvailable actions from initial state:", problem.get_actions(initial))

print("\nApplying action 'UP':")
new_state = problem.result(initial, 'UP')
problem.display_state(new_state)

print("\nIs new state the goal?", problem.is_goal(new_state))
print("Is goal state the goal?", problem.is_goal(goal))

print("\nPath cost: Each move costs 1 (implicit in our formulation)")
```

### Reflection Questions

**Question 1:** Why is it important to have a formal problem formulation before applying a search algorithm? What would happen if any of the five components (initial state, actions, transition model, goal test, path cost) were missing or incorrectly defined?

**Question 2:** In the 8-puzzle, we represent states as tuples of 9 numbers. What are the advantages of this representation compared to using a 2D array or other data structures? Consider both computational efficiency and ease of use.

**Question 3:** The transition model in our implementation checks whether moves are valid before executing them. Why is it important for the `get_actions()` method to only return valid actions rather than returning all possible actions and letting the search algorithm handle invalid ones?

---

## Exercise 2: Breadth-First Search (BFS)

### Description
This exercise demonstrates uninformed breadth-first search, which explores all nodes at depth *d* before exploring nodes at depth *d+1*. BFS uses a FIFO (first-in-first-out) queue to manage the frontier.

### Key Concepts
- **Uninformed Search**: Strategies that have no problem-specific knowledge beyond the problem definition
- **Frontier**: The set of nodes that have been generated but not yet expanded
- **FIFO Queue**: Data structure where the first element added is the first one removed
- **Completeness**: A search algorithm is complete if it always finds a solution when one exists
- **Optimality**: An algorithm is optimal if it finds a solution with the lowest path cost

### Task
Run the code and observe the search process. Focus on:
- The order in which nodes are explored (printed as "Exploring:")
- How the frontier size changes over time
- How many nodes are expanded before finding the goal
- The path returned by BFS

```python
from collections import deque

def bfs_search(problem):
    """Breadth-first search implementation."""
    
    # Node: (state, path_from_initial, cost)
    initial_node = (problem.initial, [], 0)
    
    if problem.is_goal(problem.initial):
        return []
    
    frontier = deque([initial_node])  # FIFO queue
    explored = set()
    nodes_expanded = 0
    
    print("BREADTH-FIRST SEARCH")
    print("=" * 50)
    
    while frontier:
        print(f"\nFrontier size: {len(frontier)}, Explored: {len(explored)}")
        
        # Remove first node from frontier (FIFO)
        state, path, cost = frontier.popleft()
        nodes_expanded += 1
        
        print(f"Exploring node {nodes_expanded} (depth {len(path)}):")
        problem.display_state(state)
        
        explored.add(state)
        
        # Expand the node - show what actions are available
        actions = problem.get_actions(state)
        print(f"  Available actions: {actions}")
        
        for action in actions:
            child_state = problem.result(state, action)
            print(f"  Applying '{action}' →", end=" ")
            
            if child_state not in explored and child_state not in [n[0] for n in frontier]:
                child_path = path + [action]
                child_cost = cost + 1
                
                if problem.is_goal(child_state):
                    print("GOAL!")
                    print(f"\n{'='*50}")
                    print(f"✓ Goal found! Total nodes expanded: {nodes_expanded}")
                    print(f"\nGoal state reached:")
                    problem.display_state(child_state)
                    print(f"\nSolution path: {child_path}")
                    print(f"Path length: {len(child_path)}")
                    return child_path
                else:
                    print("added to frontier")
                
                frontier.append((child_state, child_path, child_cost))
            else:
                if child_state in explored:
                    print("already explored")
                else:
                    print("already in frontier")
    
    return None  # No solution

initial = (1, 2, 3, 4, 5, 6, 0, 7, 8)
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Problem requiring 4 moves for more interesting exploration
# initial = (1, 2, 3, 4, 0, 5, 7, 8, 6)  # Blank at position 4 (center)
# goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)      # Standard goal (blank at bottom-right)

problem = EightPuzzle(initial, goal)
solution = bfs_search(problem)
```

### Reflection Questions

**Question 4:** BFS explores nodes level by level (all nodes at depth 1, then all at depth 2, etc.). How does this exploration strategy guarantee that BFS finds the optimal solution for problems where all actions have the same cost?

**Question 5:** Observe the frontier size as the search progresses. Why does the frontier grow so rapidly in BFS, and what does this tell you about the space complexity of this algorithm? How would this affect solving larger problems?

**Question 6:** Compare the number of nodes expanded to the length of the solution path. Why does BFS typically expand many more nodes than are in the final solution? What does this reveal about the efficiency of uninformed search?

---

## Exercise 3: Depth-First Search (DFS)

### Description
This exercise demonstrates depth-first search, which always expands the deepest node in the frontier using a LIFO (last-in-first-out) stack. Comparing DFS to BFS reveals important tradeoffs between different uninformed strategies.

### Key Concepts
- **LIFO Stack**: Data structure where the last element added is the first one removed
- **Depth-First Exploration**: Exploring as deeply as possible before backtracking
- **Space Complexity Advantage**: DFS stores fewer nodes in memory than BFS
- **Non-Optimal**: DFS may find a solution but not the shortest one
- **Depth Limiting**: Preventing infinite loops by limiting maximum depth

### Task
Run the code and compare DFS to BFS from Exercise 2. Notice:
- How the exploration order differs from BFS
- The maximum frontier size compared to BFS
- Whether the solution found is optimal (shortest)
- How the depth limit prevents infinite exploration

```python
def dfs_search(problem, max_depth=20):
    """Depth-first search with depth limiting."""
    
    initial_node = (problem.initial, [], 0)
    
    if problem.is_goal(problem.initial):
        return []
    
    frontier = [initial_node]  # LIFO stack (use list as stack)
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 1
    
    print("DEPTH-FIRST SEARCH")
    print("=" * 50)
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        # Remove last node from frontier (LIFO)
        state, path, cost = frontier.pop()
        
        if len(path) > max_depth:
            continue  # Depth limit reached
        
        nodes_expanded += 1
        print(f"\nExploring node {nodes_expanded} (depth {len(path)}):")
        problem.display_state(state)
        
        if problem.is_goal(state):
            print(f"\n{'='*50}")
            print(f"✓ Goal found! Total nodes expanded: {nodes_expanded}")
            print(f"Maximum frontier size: {max_frontier_size}")
            print(f"\nGoal state reached:")
            problem.display_state(state)
            print(f"\nSolution path: {path}")
            print(f"Path length: {len(path)}")
            return path
        
        explored.add(state)
        
        # Show available actions
        actions = problem.get_actions(state)
        print(f"  Available actions: {actions}")
        
        # Add children to frontier (in reverse order for consistent behavior)
        for action in reversed(actions):
            child_state = problem.result(state, action)
            print(f"  Applying '{action}' →", end=" ")
            
            if child_state not in explored and child_state not in [n[0] for n in frontier]:
                child_path = path + [action]
                child_cost = cost + 1
                frontier.append((child_state, child_path, child_cost))
                print("added to frontier")
            else:
                if child_state in explored:
                    print("already explored")
                else:
                    print("already in frontier")
    
    return None

initial = (1, 2, 3, 4, 5, 6, 0, 7, 8)
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Problem requiring 4 moves for more interesting exploration
# initial = (1, 2, 3, 4, 0, 5, 7, 8, 6)  # Blank at position 4 (center)
# goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)      # Standard goal (blank at bottom-right)

problem = EightPuzzle(initial, goal)
solution = dfs_search(problem)
```

### Reflection Questions

**Question 7:** Compare the maximum frontier size between DFS and BFS. Why does DFS use significantly less memory, and in what situations would this memory advantage be crucial for solving a problem?

**Question 8:** DFS found a solution, but was it the optimal (shortest) solution? Explain why DFS is not guaranteed to find the optimal solution even when one exists, and describe a scenario where DFS might find a very poor solution.

**Question 9:** We implemented a depth limit to prevent DFS from exploring infinitely deep paths. What problems could arise without this limit, and how does this relate to the concept of completeness in search algorithms?

---

## Exercise 4: Uniform-Cost Search (UCS)

### Description
This exercise demonstrates uniform-cost search, which expands nodes in order of their path cost. UCS uses a priority queue and is optimal for problems with varying action costs.

### Key Concepts
- **Priority Queue**: Data structure where elements are removed in order of priority (lowest cost first)
- **Path Cost**: The total cost of reaching a node from the initial state
- **Uniform-Cost Property**: Always expanding the lowest-cost node guarantees optimality
- **Action Cost Variation**: Different actions may have different costs
- **Completeness with Optimality**: UCS is both complete and optimal

### Task
Run the code and observe how UCS differs from BFS. Pay attention to:
- How nodes are ordered by path cost rather than depth
- The role of the priority queue in selecting which node to expand
- How UCS handles varying action costs
- Why the solution is guaranteed to be optimal

```python
import heapq

class GridWorld:
    """A grid world with varying terrain costs."""
    
    def __init__(self, grid, start, goal):
        self.grid = grid  # 2D list where values are terrain costs
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
    
    def get_actions(self, state):
        """Returns valid moves from state."""
        row, col = state
        actions = []
        
        if row > 0: actions.append(('UP', row - 1, col))
        if row < self.rows - 1: actions.append(('DOWN', row + 1, col))
        if col > 0: actions.append(('LEFT', row, col - 1))
        if col < self.cols - 1: actions.append(('RIGHT', row, col + 1))
        
        return actions
    
    def step_cost(self, state, action):
        """Returns cost of moving to new position."""
        _, new_row, new_col = action
        return self.grid[new_row][new_col]
    
    def is_goal(self, state):
        return state == self.goal

def ucs_search(problem):
    """Uniform-cost search implementation."""
    
    # Priority queue: (cost, counter, state, path)
    counter = 0  # For tie-breaking
    initial_node = (0, counter, problem.start, [])
    frontier = [initial_node]
    explored = set()
    nodes_expanded = 0
    
    print("UNIFORM-COST SEARCH")
    print("=" * 50)
    print("\nTerrain costs (lower is better):")
    for row in problem.grid:
        print(" ", row)
    print(f"\nStart: {problem.start}, Goal: {problem.goal}\n")
    
    while frontier:
        cost, _, state, path = heapq.heappop(frontier)  # Get lowest-cost node
        
        if state in explored:
            continue
        
        nodes_expanded += 1
        print(f"Expanding node {nodes_expanded}: {state}, Cost: {cost}")
        
        if problem.is_goal(state):
            print(f"\n✓ Goal found! Total nodes expanded: {nodes_expanded}")
            print(f"Optimal path: {path}")
            print(f"Total cost: {cost}")
            return path, cost
        
        explored.add(state)
        
        for action in problem.get_actions(state):
            action_name, new_row, new_col = action
            child_state = (new_row, new_col)
            
            if child_state not in explored:
                step_cost = problem.step_cost(state, action)
                child_cost = cost + step_cost
                child_path = path + [action_name]
                counter += 1
                
                heapq.heappush(frontier, (child_cost, counter, child_state, child_path))
    
    return None, float('inf')

# Grid where numbers represent terrain difficulty (cost to enter)
grid = [
    [1, 1, 1, 1, 1],
    [1, 5, 5, 5, 1],
    [1, 5, 1, 5, 1],
    [1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1]
]

problem = GridWorld(grid, start=(0, 0), goal=(4, 4))
solution, cost = ucs_search(problem)
```

### Reflection Questions

**Question 10:** How does UCS decide which node to expand next, and why does this strategy guarantee finding the optimal solution? Compare this to how BFS makes its expansion decisions.

**Question 11:** Observe the path found by UCS through the grid. Does it go directly toward the goal, or does it take a longer route? Explain why UCS chose this path in terms of path cost versus path length.

**Question 12:** In what types of problems would UCS perform identically to BFS? When would UCS clearly outperform BFS in terms of solution quality? Give specific examples of problem domains.

---

## Exercise 5: Greedy Best-First Search

### Description
This exercise introduces informed search by using a heuristic function that estimates the distance to the goal. Greedy best-first search always expands the node that appears closest to the goal.

### Key Concepts
- **Heuristic Function**: An estimate of the cost from a state to the goal (denoted h(n))
- **Informed Search**: Using problem-specific knowledge to guide search
- **Greedy Strategy**: Always choosing the option that looks best immediately
- **Non-Optimal**: May find a solution quickly but not the best one
- **Manhattan Distance**: A common heuristic for grid problems (sum of horizontal and vertical distances)

### Task
Run the code and observe how the heuristic guides the search. Notice:
- How nodes are prioritized based on estimated distance to goal
- Whether the search goes directly toward the goal
- How many nodes are expanded compared to uninformed search
- Whether the solution found is optimal

```python
def manhattan_distance(state, goal):
    """Heuristic: Manhattan distance in grid."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def greedy_search(problem, heuristic):
    """Greedy best-first search using heuristic."""
    
    counter = 0
    h_value = heuristic(problem.start, problem.goal)
    initial_node = (h_value, counter, problem.start, [], 0)
    frontier = [initial_node]
    explored = set()
    nodes_expanded = 0
    
    print("GREEDY BEST-FIRST SEARCH")
    print("=" * 50)
    print("\nTerrain costs:")
    for row in problem.grid:
        print(" ", row)
    print(f"\nStart: {problem.start}, Goal: {problem.goal}")
    print("Using Manhattan distance heuristic\n")
    
    while frontier:
        h_val, _, state, path, cost = heapq.heappop(frontier)
        
        if state in explored:
            continue
        
        nodes_expanded += 1
        print(f"Expanding: {state}, h={h_val}, actual_cost={cost}")
        
        if problem.is_goal(state):
            print(f"\n✓ Goal found! Nodes expanded: {nodes_expanded}")
            print(f"Path: {path}")
            print(f"Cost: {cost} (not necessarily optimal)")
            return path, cost
        
        explored.add(state)
        
        for action in problem.get_actions(state):
            action_name, new_row, new_col = action
            child_state = (new_row, new_col)
            
            if child_state not in explored:
                step_cost = problem.step_cost(state, action)
                child_cost = cost + step_cost
                child_path = path + [action_name]
                child_h = heuristic(child_state, problem.goal)
                counter += 1
                
                heapq.heappush(frontier, (child_h, counter, child_state, child_path, child_cost))
    
    return None, float('inf')

# Same grid as UCS
grid = [
    [1, 1, 1, 1, 1],
    [1, 5, 5, 5, 1],
    [1, 5, 1, 5, 1],
    [1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1]
]

problem = GridWorld(grid, start=(0, 0), goal=(4, 4))
solution, cost = greedy_search(problem, manhattan_distance)
```

### Reflection Questions

**Question 13:** Compare the number of nodes expanded by greedy best-first search versus UCS (Exercise 4). Why does the heuristic function allow greedy search to expand fewer nodes, and what risk does this efficiency create?

**Question 14:** Look at the actual cost of the solution found by greedy search versus UCS. If they differ, explain why greedy search failed to find the optimal solution. What does this reveal about the greedy strategy?

**Question 15:** The Manhattan distance heuristic estimates distance but ignores terrain costs. How does this limitation affect the search behavior? Describe how you might design a better heuristic for this specific grid world.

---

## Exercise 6: A* Search

### Description
This exercise demonstrates A* search, which combines the optimality of uniform-cost search with the efficiency of greedy search. A* uses an evaluation function f(n) = g(n) + h(n), where g(n) is the actual cost to reach n and h(n) is the estimated cost to the goal.

### Key Concepts
- **Evaluation Function f(n)**: The estimated total cost of the cheapest solution through node n
- **g(n)**: The actual cost to reach node n from the start
- **h(n)**: The heuristic estimate of cost from n to the goal
- **Admissibility**: A heuristic is admissible if it never overestimates the true cost
- **Optimal with Admissible Heuristic**: A* finds the optimal solution when h(n) is admissible

### Task
Run the code and observe how A* balances actual and estimated costs. Focus on:
- How the f(n) = g(n) + h(n) evaluation guides search
- The difference in expansion order compared to UCS and greedy search
- Why A* finds the optimal solution
- How many nodes A* expands compared to other algorithms

```python
def astar_search(problem, heuristic):
    """A* search implementation."""
    
    counter = 0
    g = 0  # Cost to reach start
    h = heuristic(problem.start, problem.goal)
    f = g + h
    initial_node = (f, counter, problem.start, [], g)
    frontier = [initial_node]
    explored = set()
    nodes_expanded = 0
    
    print("A* SEARCH")
    print("=" * 50)
    print("\nTerrain costs:")
    for row in problem.grid:
        print(" ", row)
    print(f"\nStart: {problem.start}, Goal: {problem.goal}")
    print("Using f(n) = g(n) + h(n) where h is Manhattan distance\n")
    
    while frontier:
        f_val, _, state, path, g_val = heapq.heappop(frontier)
        
        if state in explored:
            continue
        
        nodes_expanded += 1
        h_val = heuristic(state, problem.goal)
        print(f"Expanding: {state}, g={g_val}, h={h_val}, f={f_val}")
        
        if problem.is_goal(state):
            print(f"\n✓ Optimal goal found! Nodes expanded: {nodes_expanded}")
            print(f"Path: {path}")
            print(f"Optimal cost: {g_val}")
            return path, g_val
        
        explored.add(state)
        
        for action in problem.get_actions(state):
            action_name, new_row, new_col = action
            child_state = (new_row, new_col)
            
            if child_state not in explored:
                step_cost = problem.step_cost(state, action)
                child_g = g_val + step_cost
                child_path = path + [action_name]
                child_h = heuristic(child_state, problem.goal)
                child_f = child_g + child_h
                counter += 1
                
                heapq.heappush(frontier, (child_f, counter, child_state, child_path, child_g))
    
    return None, float('inf')

# Same grid
grid = [
    [1, 1, 1, 1, 1],
    [1, 5, 5, 5, 1],
    [1, 5, 1, 5, 1],
    [1, 1, 1, 5, 1],
    [1, 1, 1, 1, 1]
]

problem = GridWorld(grid, start=(0, 0), goal=(4, 4))
solution, cost = astar_search(problem, manhattan_distance)

print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print("\nRun all three searches to compare:")
print("- UCS: Expands by cost, finds optimal, may expand many nodes")
print("- Greedy: Expands by heuristic, fast but may not be optimal")
print("- A*: Expands by cost+heuristic, finds optimal efficiently")
```

### Reflection Questions

**Question 16:** Explain how A* combines the strengths of UCS and greedy best-first search. How does the f(n) = g(n) + h(n) evaluation function achieve this balance, and why does this lead to both optimality and efficiency?

**Question 17:** Compare the number of nodes expanded by A* to both UCS and greedy search. Where does A* fall on this spectrum, and what does this tell you about A*'s practical performance characteristics?

**Question 18:** The Manhattan distance heuristic is admissible for this grid world. Explain what admissibility means and why it's crucial for A* to guarantee finding the optimal solution. What would happen if we used a heuristic that sometimes overestimated?

---

## Exercise 7: Comparing Heuristics

### Description
This exercise demonstrates how different heuristic functions affect A* performance. We'll compare Manhattan distance with a weaker heuristic (always returns 0) and explore the concept of heuristic dominance.

### Key Concepts
- **Heuristic Dominance**: h2 dominates h1 if h2(n) ≥ h1(n) for all n and both are admissible
- **Null Heuristic**: h(n) = 0 for all n (reduces A* to UCS)
- **Informed vs Uninformed**: Better heuristics expand fewer nodes
- **Consistency (Monotonicity)**: h(n) ≤ c(n, a, n') + h(n') for all actions
- **Effective Branching Factor**: Measure of search efficiency

### Task
Run the code and observe how heuristic quality affects performance. Notice:
- How many nodes each heuristic causes A* to expand
- The relationship between heuristic values and nodes expanded
- Why a better heuristic leads to more efficient search
- The behavior when h(n) = 0 (should match UCS)

```python
def null_heuristic(state, goal):
    """Heuristic that provides no information."""
    return 0

def euclidean_distance(state, goal):
    """Straight-line distance heuristic."""
    return ((state[0] - goal[0])**2 + (state[1] - goal[1])**2)**0.5

def astar_with_stats(problem, heuristic, heuristic_name):
    """A* search that tracks statistics."""
    
    counter = 0
    g = 0
    h = heuristic(problem.start, problem.goal)
    f = g + h
    initial_node = (f, counter, problem.start, [], g)
    frontier = [initial_node]
    explored = set()
    nodes_expanded = 0
    max_frontier = 1
    
    print(f"\nA* with {heuristic_name}")
    print("-" * 40)
    
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        f_val, _, state, path, g_val = heapq.heappop(frontier)
        
        if state in explored:
            continue
        
        nodes_expanded += 1
        
        if problem.is_goal(state):
            print(f"✓ Solution found!")
            print(f"  Nodes expanded: {nodes_expanded}")
            print(f"  Max frontier size: {max_frontier}")
            print(f"  Solution cost: {g_val}")
            print(f"  Solution length: {len(path)}")
            return nodes_expanded, g_val
        
        explored.add(state)
        
        for action in problem.get_actions(state):
            action_name, new_row, new_col = action
            child_state = (new_row, new_col)
            
            if child_state not in explored:
                step_cost = problem.step_cost(state, action)
                child_g = g_val + step_cost
                child_path = path + [action_name]
                child_h = heuristic(child_state, problem.goal)
                child_f = child_g + child_h
                counter += 1
                
                heapq.heappush(frontier, (child_f, counter, child_state, child_path, child_g))
    
    return None, float('inf')

print("HEURISTIC COMPARISON")
print("=" * 50)

# Larger grid for more interesting comparison
grid = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 5, 5, 5, 5, 5, 1],
    [1, 5, 1, 1, 1, 5, 1],
    [1, 5, 1, 5, 1, 5, 1],
    [1, 5, 1, 5, 1, 5, 1],
    [1, 5, 5, 5, 5, 5, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

problem = GridWorld(grid, start=(0, 0), goal=(6, 6))

print("Grid layout:")
for row in grid:
    print(" ", row)
print(f"\nStart: {problem.start}, Goal: {problem.goal}\n")

# Test each heuristic
stats = []
stats.append(astar_with_stats(problem, null_heuristic, "Null Heuristic (h=0)"))
stats.append(astar_with_stats(problem, manhattan_distance, "Manhattan Distance"))
stats.append(astar_with_stats(problem, euclidean_distance, "Euclidean Distance"))

print("\n" + "=" * 50)
print("ANALYSIS")
print("=" * 50)
print("\nAll heuristics found optimal solution (cost={})".format(stats[0][1]))
print("\nNodes expanded comparison:")
print(f"  Null (A* = UCS):     {stats[0][0]} nodes")
print(f"  Manhattan:           {stats[1][0]} nodes")
print(f"  Euclidean:           {stats[2][0]} nodes")

print("\nKey insight: Better heuristics expand fewer nodes!")
print("Manhattan dominates null heuristic (always ≥ 0)")
print("Both are admissible (never overestimate true cost)")
```

### Reflection Questions

**Question 19:** When h(n) = 0, A* expands the same nodes as UCS. Explain why this happens by considering what the f(n) = g(n) + h(n) evaluation function reduces to. What does this reveal about the relationship between A* and UCS?

**Question 20:** Compare the nodes expanded using Manhattan distance versus Euclidean distance. Which is a better heuristic for grid movement, and why? Consider what types of moves are allowed in the grid world.

**Question 21:** We say Manhattan distance "dominates" the null heuristic because it provides more information while remaining admissible. Explain how providing more accurate information leads to expanding fewer nodes, and why this doesn't compromise optimality as long as the heuristic remains admissible.

---

## Summary and Key Takeaways

### Problem Formulation
You've learned that every search problem requires five components: initial state, actions, transition model, goal test, and path cost. Proper formulation is essential before applying any search algorithm.

### Uninformed Search Strategies
- **BFS**: Explores level-by-level, guarantees optimal solutions for uniform costs, but uses exponential space
- **DFS**: Explores deeply first, uses linear space, but may find suboptimal solutions and is incomplete without depth limiting
- **UCS**: Expands by path cost, guarantees optimal solutions for variable costs, but may expand many nodes

### Informed Search Strategies
- **Greedy Best-First**: Uses heuristic to expand seemingly closest nodes, efficient but not optimal
- **A***: Combines actual cost g(n) and heuristic h(n), guarantees optimal solutions with admissible heuristics

### Heuristic Functions
- Admissibility (never overestimate) is crucial for A* optimality
- Better heuristics (higher values while remaining admissible) expand fewer nodes
- Consistency is a stronger property that provides additional benefits
- Heuristic quality dramatically affects search efficiency

### Algorithm Selection
The choice of search algorithm depends on:
- Whether you know the cost structure (uniform vs. variable)
- Space constraints (DFS for memory-limited scenarios)
- Whether an admissible heuristic exists (A* when available)
- Optimality requirements (UCS or A* for guaranteed optimal solutions)

---

## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including the `agent.ipynb` file from this lab and:

A Markdown document called `reflections.md` containing this header

Create `lab_ch3_results.md`:

```markdown
# Names: Your names here
# Lab: lab1 (Intelligent Agents)
# Date: Today's date
```

And your answers to all reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the second lab!

---