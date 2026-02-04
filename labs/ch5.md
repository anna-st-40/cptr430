# Adversarial Game AI Lab: Tron Light Cycles
**Textbook Reference:** Russell & Norvig, "Artificial Intelligence: A Modern Approach" - Chapters 5 (Adversarial Search), 17 (Making Complex Decisions)

## Learning Objectives

By the end of this lab, you will be able to:

1. **Distinguish** between different adversarial search strategies (random, heuristic, minimax, MCTS) based on their decision-making processes
2. **Identify** the strengths and weaknesses of different AI approaches when applied to the same adversarial problem
3. **Evaluate** the role of simulation and lookahead in game-playing agents

## Lab Overview

This lab uses **Tron Light Cycles** (a grid-based adversarial game) as a concrete domain to explore different AI approaches to competitive decision-making. In Tron, two players move simultaneously on a grid, leaving permanent trails behind them. If a player hits a wall, border, or trail, they lose. The last player alive wins.

You will observe and analyze implementations of:
- Random baseline agents
- Heuristic-based greedy agents
- Minimax with alpha-beta pruning
- Monte Carlo Tree Search (MCTS)
- LLM-based reasoning (using Ollama)

**Setup Requirements:**
- Python 3.10+
- Libraries: `numpy`, `random`, `copy`, `requests` (for Ollama API), `pygame` (for visualization)
- CS Lab network access with `llama3.2` model pulled (for Exercise 5 only)
- Install pygame: ``

I suggest running the following commands from your base user directory if necessary:

```bash
mkdir cs430 
cd cs430 
uv init 
uv add numpy
uv add pygame
uv add requests
source .venv/bin/activate
```

---

**Code Organization:**
- **Exercise 1** creates `tron_base.py` - save this file (contains TronGame, RandomAgent, flood_fill)
- **Exercises 2-5** add new agents to separate files
- **Exercises 6-8** import from all modules and use all agents

**Pedagogical Approach:** You will run complete implementations and observe their behavior, output, and decision patterns. Focus on understanding *why* each approach makes the decisions it does, not on writing the code yourself.

---

## Exercise 1: The Tron Environment and Random Baseline

### Description
This exercise establishes the core Tron game environment and implements the simplest possible AI: random action selection. Understanding the baseline helps us appreciate what intelligence adds to decision-making.

### Key Concepts
- **Zero-sum game**: One player's gain is exactly another player's loss (winner gets +1, loser gets -1)
- **Game state**: Complete information about board configuration, player positions, and valid moves
- **Action space**: The set of legal moves available at any given state (UP, DOWN, LEFT, RIGHT)
- **Terminal state**: A game configuration where no further moves are possible (someone has crashed)

### Task
Run the code below and observe multiple games between two random agents. Pay attention to:
- How game length varies between matches
- The pattern of "who wins" (is it roughly 50/50?)
- What kinds of positions lead to quick losses
- **Watch the visualization at the end to see the game in action!!**

**Important:** Save this code as `tron_base.py` - you'll import from it in later exercises.

```python
# tron_base.py - Save this file, you'll import from it later!

import numpy as np
import random
from copy import deepcopy
import pygame
import time

def flood_fill(board, start_pos, player_id):
    """Count empty cells reachable from start position"""
    visited = set()
    stack = [start_pos]
    count = 0
    height, width = board.shape
    
    while stack:
        pos = stack.pop()
        if pos in visited:
            continue
        
        y, x = pos
        if not (0 <= y < height and 0 <= x < width):
            continue
        if board[y, x] != 0 and board[y, x] != player_id:
            continue  # Hit a wall or opponent trail
        
        visited.add(pos)
        count += 1
        
        # Add neighbors
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            stack.append((y + dy, x + dx))
    
    return count

class TronGame:
    """Tron Light Cycles game environment"""
    
    def __init__(self, width=12, height=12, visualize=False, cell_size=40):
        self.width = width
        self.height = height
        self.visualize = visualize
        self.cell_size = cell_size
        
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
            pygame.display.set_caption("Tron AI Battle")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.P1_COLOR = (0, 191, 255)  # Deep sky blue
            self.P2_COLOR = (255, 69, 0)   # Red-orange
            self.GRID_COLOR = (30, 30, 30)
        
        self.reset()
    
    def reset(self):
        """Initialize new game"""
        self.board = np.zeros((self.height, self.width), dtype=int)
        # Place players in opposite corners
        self.p1_pos = (1, 1)
        self.p2_pos = (self.height - 2, self.width - 2)
        self.board[self.p1_pos] = 1  # Player 1 trail
        self.board[self.p2_pos] = 2  # Player 2 trail
        self.game_over = False
        self.winner = None
        
        if self.visualize:
            self.draw()
        
        return self.get_state()
    
    def draw(self):
        """Render the game state using Pygame"""
        self.screen.fill(self.BLACK)
        
        # Draw grid lines
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height * self.cell_size))
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width * self.cell_size, y))
        
        # Draw trails
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] == 1:
                    color = self.P1_COLOR
                    pygame.draw.rect(self.screen, color, 
                                   (x * self.cell_size + 2, y * self.cell_size + 2, 
                                    self.cell_size - 4, self.cell_size - 4))
                elif self.board[y, x] == 2:
                    color = self.P2_COLOR
                    pygame.draw.rect(self.screen, color, 
                                   (x * self.cell_size + 2, y * self.cell_size + 2, 
                                    self.cell_size - 4, self.cell_size - 4))
        
        # Draw player heads (larger circles)
        pygame.draw.circle(self.screen, self.WHITE, 
                          (self.p1_pos[1] * self.cell_size + self.cell_size // 2,
                           self.p1_pos[0] * self.cell_size + self.cell_size // 2), 
                          self.cell_size // 3)
        pygame.draw.circle(self.screen, self.WHITE, 
                          (self.p2_pos[1] * self.cell_size + self.cell_size // 2,
                           self.p2_pos[0] * self.cell_size + self.cell_size // 2), 
                          self.cell_size // 3)
        
        # Display player labels
        font = pygame.font.Font(None, 24)
        p1_text = font.render("P1", True, self.BLACK)
        p2_text = font.render("P2", True, self.BLACK)
        self.screen.blit(p1_text, (self.p1_pos[1] * self.cell_size + self.cell_size // 2 - 10,
                                   self.p1_pos[0] * self.cell_size + self.cell_size // 2 - 8))
        self.screen.blit(p2_text, (self.p2_pos[1] * self.cell_size + self.cell_size // 2 - 10,
                                   self.p2_pos[0] * self.cell_size + self.cell_size // 2 - 8))
        
        pygame.display.flip()
        
        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.visualize = False
    
    def get_valid_moves(self, pos):
        """Return list of valid moves from position"""
        moves = []
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 
                     'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        for action, (dy, dx) in directions.items():
            new_y, new_x = pos[0] + dy, pos[1] + dx
            # Check bounds and empty cell
            if (0 <= new_y < self.height and 
                0 <= new_x < self.width and 
                self.board[new_y, new_x] == 0):
                moves.append(action)
        return moves
    
    def step(self, p1_action, p2_action):
        """Execute both players' moves simultaneously"""
        if self.game_over:
            return self.get_state(), 0, True
        
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 
                     'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        # Calculate new positions
        dy1, dx1 = directions.get(p1_action, (0, 0))
        dy2, dx2 = directions.get(p2_action, (0, 0))
        new_p1 = (self.p1_pos[0] + dy1, self.p1_pos[1] + dx1)
        new_p2 = (self.p2_pos[0] + dy2, self.p2_pos[1] + dx2)
        
        # Check collisions
        p1_valid = (0 <= new_p1[0] < self.height and 
                   0 <= new_p1[1] < self.width and 
                   self.board[new_p1] == 0)
        p2_valid = (0 <= new_p2[0] < self.height and 
                   0 <= new_p2[1] < self.width and 
                   self.board[new_p2] == 0)
        
        # Determine winner
        if not p1_valid and not p2_valid:
            self.game_over = True
            self.winner = 0  # Draw
            if self.visualize:
                self.draw()
                pygame.time.wait(500)
            return self.get_state(), 0, True
        elif not p1_valid:
            self.game_over = True
            self.winner = 2
            if self.visualize:
                self.draw()
                pygame.time.wait(500)
            return self.get_state(), -1, True
        elif not p2_valid:
            self.game_over = True
            self.winner = 1
            if self.visualize:
                self.draw()
                pygame.time.wait(500)
            return self.get_state(), 1, True
        
        # Update positions and board
        self.p1_pos = new_p1
        self.p2_pos = new_p2
        self.board[new_p1] = 1
        self.board[new_p2] = 2
        
        if self.visualize:
            self.draw()
            self.clock.tick(10)  # 10 FPS for visualization
        
        return self.get_state(), 0, False
    
    def close(self):
        """Clean up Pygame resources"""
        if self.visualize:
            pygame.quit()
            self.visualize = False
    
    def get_state(self):
        """Return current game state"""
        return {
            'board': self.board.copy(),
            'p1_pos': self.p1_pos,
            'p2_pos': self.p2_pos,
            'p1_moves': self.get_valid_moves(self.p1_pos),
            'p2_moves': self.get_valid_moves(self.p2_pos)
        }

class RandomAgent:
    """Agent that selects random valid moves"""
    
    def get_action(self, state, player):
        """Return random valid action"""
        moves = state['p1_moves'] if player == 1 else state['p2_moves']
        return random.choice(moves) if moves else None

# Tournament and visualization functions
def run_random_tournament(num_games=10):
    """Run tournament between two random agents"""
    print("=== RANDOM vs RANDOM ({} games) ===\n".format(num_games))
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    results = {'p1': 0, 'p2': 0, 'draw': 0}
    
    for game_num in range(num_games):
        game = TronGame(width=12, height=12)
        state = game.reset()
        moves = 0
        
        while not game.game_over:
            a1 = agent1.get_action(state, 1)
            a2 = agent2.get_action(state, 2)
            state, reward, done = game.step(a1, a2)
            moves += 1
        
        if game.winner == 1:
            results['p1'] += 1
        elif game.winner == 2:
            results['p2'] += 1
        else:
            results['draw'] += 1
        
        print(f"Game {game_num + 1}: Winner = Player {game.winner if game.winner else 'Draw'}, Moves = {moves}")
    
    print(f"\nResults: P1 wins={results['p1']}, P2 wins={results['p2']}, Draws={results['draw']}")
    return results

def visualize_random_game():
    """Run one visualized game between random agents"""
    print("\n" + "="*60)
    print("BONUS: Watch one game with visualization!")
    print("="*60)
    print("Running Random vs Random with Pygame visualization...")
    
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    game_viz = TronGame(width=12, height=12, visualize=True, cell_size=50)
    state = game_viz.reset()
    
    while not game_viz.game_over:
        a1 = agent1.get_action(state, 1)
        a2 = agent2.get_action(state, 2)
        state, reward, done = game_viz.step(a1, a2)
    
    winner = f"Player {game_viz.winner}" if game_viz.winner else "Draw"
    print(f"\nVisualized game complete! Winner: {winner}")
    game_viz.close()

# Run if executed directly
if __name__ == "__main__":
    run_random_tournament(10)
    visualize_random_game()
```

### Reflection Questions

**Question 1:** Why is the win rate between two random agents approximately equal, and what does this tell us about the relationship between starting position and strategy in Tron?

**Question 2:** Random agents occasionally win by "accidentally" making good moves. Explain why this approach cannot scale to more complex games and what properties an intelligent agent needs that random selection lacks.

**Question 3:** How does the average game length between random agents compare to what you might expect from intelligent play, and what does this reveal about the relationship between lookahead and survival time?

**Question 3b (Visualization):** After watching the visualized game, describe how observing the spatial patterns of random movement helps you understand why random agents crash quickly. What visual patterns emerge that text-based statistics don't capture?

---

## Exercise 2: Greedy Heuristic Agent (Space Control)

### Description
This exercise introduces the concept of a **heuristic evaluation function**—a way to estimate how "good" a position is without looking ahead. The greedy agent uses flood-fill to count reachable empty spaces and moves toward the direction with the most available space.

### Key Concepts
- **Heuristic function**: A rule-of-thumb that estimates position quality without perfect knowledge
- **Flood fill**: Algorithm that counts connected empty cells reachable from a position
- **Greedy strategy**: Always choose the action that looks best right now, without considering future consequences
- **Spatial control**: In Tron, controlling more empty space correlates with longer survival

### Task
Run the code and observe how the greedy agent behaves against random. Notice:
- Does the greedy agent consistently beat random?
- What happens when both agents get trapped in small spaces?
- How does the agent decide between equally-sized spaces?

**Important:** Save this code as `greedy.py`.

```python
# greedy.py - Add to this file

# Import the base game and random agent from Exercise 1
from tron_base import TronGame, RandomAgent, flood_fill

class GreedyAgent:
    """Agent that maximizes immediate space control"""
    
    def get_action(self, state, player):
        """Select action leading to most available space"""
        pos = state['p1_pos'] if player == 1 else state['p2_pos']
        moves = state['p1_moves'] if player == 1 else state['p2_moves']
        
        if not moves:
            return None
        
        best_action = None
        best_space = -1
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 
                     'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        # Evaluate each move
        for action in moves:
            dy, dx = directions[action]
            new_pos = (pos[0] + dy, pos[1] + dx)
            space = flood_fill(state['board'], new_pos, player)
            
            if space > best_space:
                best_space = space
                best_action = action
        
        return best_action

# Tournament - Test the greedy agent
# Run if executed directly
if __name__ == "__main__":
    print("\n=== GREEDY vs RANDOM (10 games) ===\n")
    greedy = GreedyAgent()
    random_agent = RandomAgent()
    results = {'greedy': 0, 'random': 0, 'draw': 0}

    for game_num in range(10):
        game = TronGame(width=12, height=12)
        state = game.reset()
        moves = 0
        
        while not game.game_over:
            a1 = greedy.get_action(state, 1)
            a2 = random_agent.get_action(state, 2)
            state, reward, done = game.step(a1, a2)
            moves += 1
        
        winner_name = "Greedy" if game.winner == 1 else ("Random" if game.winner == 2 else "Draw")
        if game.winner == 1:
            results['greedy'] += 1
        elif game.winner == 2:
            results['random'] += 1
        else:
            results['draw'] += 1
        
        print(f"Game {game_num + 1}: Winner = {winner_name}, Moves = {moves}")

    print(f"\nResults: Greedy={results['greedy']}, Random={results['random']}, Draws={results['draw']}")

    # Optional: Visualize one match
    print("\n" + "="*60)
    print("Watch Greedy vs Random with visualization!")
    print("="*60)

    game_viz = TronGame(width=12, height=12, visualize=True, cell_size=50)
    state = game_viz.reset()

    while not game_viz.game_over and state:
        a1 = greedy.get_action(state, 1)
        a2 = random_agent.get_action(state, 2)
        state, reward, done = game_viz.step(a1, a2)

    winner = "Greedy" if game_viz.winner == 1 else ("Random" if game_viz.winner == 2 else "Draw")
    print(f"Visualized game: {winner} wins!")
    game_viz.close()
```

### Reflection Questions

**Question 4:** Explain why the greedy agent's flood-fill heuristic is effective against random play, and what assumption about survival it makes that proves generally correct in Tron.

**Question 5:** Describe a scenario where greedy space-maximization could lead to a losing position, demonstrating the difference between local optimality and global strategy.

**Question 6:** How does the computational cost of flood-fill (which explores many cells) compare to random selection, and why might this cost be acceptable for a real-time game?

**Question 6b (Visualization):** After watching the greedy vs random visualization, describe what strategic patterns you noticed in the greedy agent's movement. How does visualizing the space-control heuristic in action deepen your understanding compared to just reading the code?

---

## Exercise 3: Minimax with Alpha-Beta Pruning

### Description
This exercise implements **minimax search**—the classic adversarial search algorithm that assumes optimal play from both players. The agent looks ahead several moves, considers all possible move combinations, and chooses the action that guarantees the best worst-case outcome.

### Key Concepts
- **Minimax principle**: Maximize your minimum guaranteed outcome assuming opponent plays optimally
- **Alpha-beta pruning**: Optimization that skips evaluating branches that cannot affect the final decision
- **Depth-limited search**: Stop searching after a fixed number of moves and use heuristic evaluation
- **Game tree**: Tree structure where nodes are game states and edges are actions

### Task
Run the code and observe minimax vs greedy. Watch for:
- How does search depth affect playing strength?
- How many nodes does alpha-beta prune compared to full minimax?
- Does minimax consistently beat greedy at depth-3, or is the advantage modest?
- **Critical observation:** If minimax doesn't dominate, why might that be?

**Important:** Save this code as `minimax.py`.

```python
# minimax.py - Add to this file
# Import base game and agents from previous exercises
from tron_base import TronGame, flood_fill
from greedy import GreedyAgent
from copy import deepcopy

# Minimax agent implementation
class MinimaxAgent:
    """Agent using minimax with alpha-beta pruning"""
    
    def __init__(self, depth=5):
        self.depth = depth
        self.nodes_evaluated = 0
    
    def evaluate_state(self, board, p1_pos, p2_pos):
        """Heuristic: difference in reachable space"""
        p1_space = flood_fill(board, p1_pos, 1)
        p2_space = flood_fill(board, p2_pos, 2)
        return p1_space - p2_space
    
    def minimax(self, state, depth, alpha, beta, maximizing_player):
        """Minimax with alpha-beta pruning"""
        self.nodes_evaluated += 1
        
        # Terminal conditions
        if depth == 0 or not state['p1_moves'] or not state['p2_moves']:
            return self.evaluate_state(state['board'], state['p1_pos'], state['p2_pos'])
        
        if maximizing_player:
            max_eval = float('-inf')
            for action in state['p1_moves']:
                # Simulate move
                new_state = self.simulate_move(state, action, None, 1)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for action in state['p2_moves']:
                new_state = self.simulate_move(state, None, action, 2)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def simulate_move(self, state, p1_action, p2_action, player):
        """Create new state after hypothetical move"""
        new_state = deepcopy(state)
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 
                     'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        if player == 1 and p1_action:
            dy, dx = directions[p1_action]
            new_pos = (state['p1_pos'][0] + dy, state['p1_pos'][1] + dx)
            new_state['board'][new_pos] = 1
            new_state['p1_pos'] = new_pos
            new_state['p1_moves'] = self.get_valid_moves_from_board(new_state['board'], new_pos)
        
        if player == 2 and p2_action:
            dy, dx = directions[p2_action]
            new_pos = (state['p2_pos'][0] + dy, state['p2_pos'][1] + dx)
            new_state['board'][new_pos] = 2
            new_state['p2_pos'] = new_pos
            new_state['p2_moves'] = self.get_valid_moves_from_board(new_state['board'], new_pos)
        
        return new_state
    
    def get_valid_moves_from_board(self, board, pos):
        """Helper to get valid moves from board state"""
        moves = []
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 
                     'LEFT': (0, -1), 'RIGHT': (0, 1)}
        height, width = board.shape
        
        for action, (dy, dx) in directions.items():
            new_y, new_x = pos[0] + dy, pos[1] + dx
            if (0 <= new_y < height and 0 <= new_x < width and board[new_y, new_x] == 0):
                moves.append(action)
        return moves
    
    def get_action(self, state, player):
        """Select best action using minimax"""
        self.nodes_evaluated = 0
        moves = state['p1_moves'] if player == 1 else state['p2_moves']
        
        if not moves:
            return None
        
        best_action = moves[0]
        best_value = float('-inf') if player == 1 else float('inf')
        
        for action in moves:
            new_state = self.simulate_move(state, action if player == 1 else None, 
                                          action if player == 2 else None, player)
            value = self.minimax(new_state, self.depth - 1, float('-inf'), float('inf'), 
                               player == 2)
            
            if player == 1 and value > best_value:
                best_value = value
                best_action = action
            elif player == 2 and value < best_value:
                best_value = value
                best_action = action
        
        return best_action

# Tournament and visualization functions
def run_minimax_tournament(num_games=5, depth=5):
    """Run tournament between minimax and greedy agents"""
    print("\n=== MINIMAX (depth={}) vs GREEDY ({} games) ===\n".format(depth, num_games))
    
    minimax = MinimaxAgent(depth=depth)
    greedy = GreedyAgent()
    results = {'minimax': 0, 'greedy': 0, 'draw': 0}
    
    for game_num in range(num_games):
        game = TronGame(width=10, height=10)  # Smaller for speed
        state = game.reset()
        moves = 0
        
        while not game.game_over and moves < 100:
            a1 = minimax.get_action(state, 1)
            a2 = greedy.get_action(state, 2)
            state, reward, done = game.step(a1, a2)
            moves += 1
        
        winner_name = "Minimax" if game.winner == 1 else ("Greedy" if game.winner == 2 else "Draw")
        if game.winner == 1:
            results['minimax'] += 1
        elif game.winner == 2:
            results['greedy'] += 1
        else:
            results['draw'] += 1
        
        print(f"Game {game_num + 1}: Winner = {winner_name}, Moves = {moves}, Nodes = {minimax.nodes_evaluated}")
    
    print(f"\nResults: Minimax={results['minimax']}, Greedy={results['greedy']}, Draws={results['draw']}")
    return results

def visualize_minimax_game(depth=5):
    """Run one visualized game between minimax and greedy"""
    print("\n" + "="*60)
    print("Watch Minimax vs Greedy - notice the planning!")
    print("="*60)
    
    minimax = MinimaxAgent(depth=depth)
    greedy = GreedyAgent()
    game_viz = TronGame(width=10, height=10, visualize=True, cell_size=50)
    state = game_viz.reset()
    move_count = 0
    
    while not game_viz.game_over and move_count < 100:
        a1 = minimax.get_action(state, 1)
        a2 = greedy.get_action(state, 2)
        state, reward, done = game_viz.step(a1, a2)
        move_count += 1
    
    winner = "Minimax" if game_viz.winner == 1 else ("Greedy" if game_viz.winner == 2 else "Draw")
    print(f"Visualized game: {winner} wins! ({move_count} moves)")
    game_viz.close()


def run_comparison(num_games=20, board_size=12, depths=[2, 3, 4, 5]):
    """Compare greedy vs minimax at different depths"""
    print(f"\n{'='*70}")
    print(f"BASELINE: Greedy vs Minimax on {board_size}x{board_size} board")
    print(f"{'='*70}\n")
    
    greedy = GreedyAgent()
    results = {}
    
    for depth in depths:
        print(f"\n--- Minimax Depth-{depth} vs Greedy ({num_games} games) ---")
        minimax = MinimaxAgent(depth=depth)
        
        wins_minimax = 0
        wins_greedy = 0
        draws = 0
        total_time = 0
        total_moves = 0
        
        for game_num in range(num_games):
            game = TronGame(width=board_size, height=board_size)
            state = game.reset()
            moves = 0
            
            import time
            start = time.time()
            
            while not game.game_over and moves < 200:
                a1 = minimax.get_action(state, 1)
                a2 = greedy.get_action(state, 2)
                state, reward, done = game.step(a1, a2)
                moves += 1
            
            elapsed = time.time() - start
            total_time += elapsed
            total_moves += moves
            
            if game.winner == 1:
                wins_minimax += 1
            elif game.winner == 2:
                wins_greedy += 1
            else:
                draws += 1
        
        win_rate = wins_minimax / num_games * 100
        avg_time = total_time / num_games
        avg_moves = total_moves / num_games
        
        results[depth] = {
            'minimax_wins': wins_minimax,
            'greedy_wins': wins_greedy,
            'draws': draws,
            'win_rate': win_rate,
            'avg_time': avg_time,
            'avg_moves': avg_moves
        }
        
        print(f"  Minimax: {wins_minimax}, Greedy: {wins_greedy}, Draws: {draws}")
        print(f"  Minimax Win Rate: {win_rate:.1f}%")
        print(f"  Avg Time/Game: {avg_time:.2f}s")
        print(f"  Avg Game Length: {avg_moves:.1f} moves")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Depth':<8} {'Win Rate':<12} {'Avg Time':<12} {'Avg Moves':<12}")
    print("-"*70)
    for depth, stats in results.items():
        print(f"{depth:<8} {stats['win_rate']:>10.1f}% {stats['avg_time']:>10.2f}s {stats['avg_moves']:>10.1f}")
    
    return results

if __name__ == "__main__":
    results = run_comparison(num_games=20, board_size=12, depths=[2, 3, 4, 5])

if __name__ == "__main__":
    run_minimax_tournament(5, depth=5)
    visualize_minimax_game(depth=5)
```

### Reflection Questions

**Question 7:** Explain why minimax requires an evaluation function at depth limits rather than computing exact game outcomes, and what trade-off this represents between accuracy and computational feasibility. Based on the results, at what depth does minimax start to significantly outperform greedy (if at all)? What does this suggest about the "lookahead horizon" needed in Tron?

**Question 8:** Analyze how alpha-beta pruning reduces the search space without affecting the final decision, and describe a scenario where pruning would be most effective (many cutoffs vs few cutoffs).

**Question 9:** Compare minimax's assumption of optimal opponent play to the greedy agent's behavior. When would this assumption hurt minimax's performance, and when would it help?

**Question 9b (Critical Thinking):** If minimax with depth-3 only wins 50-60% of games against greedy (rather than 80-90%), what does this suggest about the relationship between lookahead and the quality of the evaluation function? Consider that both algorithms use the same space-difference heuristic at their search horizon.

---

## Exercise 4: Monte Carlo Tree Search (MCTS)

### Description
MCTS takes a fundamentally different approach: instead of exhaustively searching to a fixed depth, it runs many random simulations and uses statistics to identify promising moves. This exercise demonstrates how probabilistic sampling can guide decision-making.

### Key Concepts
- **UCB1 (Upper Confidence Bound)**: Formula balancing exploitation (choose known-good moves) vs exploration (try uncertain moves)
- **Rollout/simulation**: Playing out a game randomly from a position to estimate its value
- **Selection-Expansion-Simulation-Backpropagation**: The four phases of MCTS iteration
- **Monte Carlo methods**: Using randomness to approximate solutions to deterministic problems

### Task
Run the code and compare MCTS to minimax. Observe:
- How does simulation count affect performance?
- Are MCTS decisions more "random-looking" than minimax?
- Which approach handles time pressure better?

**Important:** Save this code as `mcts.py`.

```python
# mcts.py - Add to this file

from tron_base import TronGame, flood_fill
from greedy import GreedyAgent
import math
import random
from copy import deepcopy

# MCTS implementation
import math
import random
import copy


class MCTSNode:
    def __init__(self, state, player, parent=None, move=None):
        self.state = state
        self.player = player            # player TO MOVE at this node
        self.parent = parent
        self.move = move

        self.children = []
        self.visits = 0
        self.value = 0.0

        self.untried_actions = self.get_moves()

    def get_moves(self):
        return self.state['p1_moves'][:] if self.player == 1 else self.state['p2_moves'][:]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=math.sqrt(2)):
        return max(
            self.children,
            key=lambda child: (
                child.value / child.visits
                + c * math.sqrt(math.log(self.visits) / child.visits)
            )
        )


class MCTSAgent:
    def __init__(self, simulations=200):
        self.simulations = simulations

    def search(self, root_state, player):
        self.root_player = player
        root = MCTSNode(copy.deepcopy(root_state), player)

        if self.is_terminal(root_state):
            return None

        for _ in range(self.simulations):
            node = self.select(root)
            result = self.simulate(node.state, node.player)
            self.backpropagate(node, result)

        if not root.children:
            moves = root_state['p1_moves'] if player == 1 else root_state['p2_moves']
            return random.choice(moves) if moves else None

        return max(root.children, key=lambda c: c.visits).move



    def get_action(self, state, player):
        return self.search(state, player)


    # ------------------------
    # Selection + Expansion
    # ------------------------

    def select(self, node):
        while True:
            # If terminal, return immediately
            terminal_value = self.is_terminal(node.state)
            if terminal_value is not None:
                return node

            # If node has moves to expand, expand one
            if not node.is_fully_expanded():
                return self.expand(node)

            # Node fully expanded: go to best child if it exists
            if node.children:
                node = node.best_child()
            else:
                # Node has no children and no moves → treat as terminal
                return node



    def expand(self, node):
        move = node.untried_actions.pop()
        next_state = copy.deepcopy(node.state)
        self.apply_move(next_state, move, node.player)

        next_player = 3 - node.player
        child = MCTSNode(
            state=next_state,
            player=next_player,
            parent=node,
            move=move
        )
        node.children.append(child)
        return child

    # ------------------------
    # Simulation (Rollout)
    # ------------------------

    def simulate(self, state, player, max_depth=200):
        current_state = copy.deepcopy(state)
        current_player = player
        depth = 0

        while depth < max_depth:
            terminal_value = self.is_terminal(current_state)
            if terminal_value is not None:
                return terminal_value

            moves = (
                current_state['p1_moves'] if current_player == 1 else current_state['p2_moves']
            )
            if not moves:  # No moves → current player loses
                return -1 if current_player == self.root_player else 1

            move = random.choice(moves)
            self.apply_move(current_state, move, current_player)
            current_player = 3 - current_player
            depth += 1

        return 0  # Timeout = draw


    # ------------------------
    # Backpropagation
    # ------------------------

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

    # ------------------------
    # Terminal Evaluation
    # ------------------------

    def is_terminal(self, state):
        if 'loser' in state:
            winner = 3 - state['loser']
            return 1 if winner == self.root_player else -1
        return None

        
    def apply_move(self, state, action, player):
        directions = {
            'UP': (-1, 0), 'DOWN': (1, 0),
            'LEFT': (0, -1), 'RIGHT': (0, 1)
        }

        pos = state['p1_pos'] if player == 1 else state['p2_pos']
        dy, dx = directions[action]
        new_pos = (pos[0] + dy, pos[1] + dx)

        h, w = state['board'].shape
        if not (0 <= new_pos[0] < h and 0 <= new_pos[1] < w):
            state['p1_moves'] = []
            state['p2_moves'] = []
            state['loser'] = player
            return

        if state['board'][new_pos] != 0:
            state['p1_moves'] = []
            state['p2_moves'] = []
            state['loser'] = player
            return

        state['board'][new_pos] = player

        if player == 1:
            state['p1_pos'] = new_pos
        else:
            state['p2_pos'] = new_pos

        state['p1_moves'] = self.get_valid_moves(state['board'], state['p1_pos'])
        state['p2_moves'] = self.get_valid_moves(state['board'], state['p2_pos'])

    
    def get_valid_moves(self, board, pos):
        """Get valid moves from position"""
        moves = []
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 
                     'LEFT': (0, -1), 'RIGHT': (0, 1)}
        height, width = board.shape
        
        for action, (dy, dx) in directions.items():
            new_y, new_x = pos[0] + dy, pos[1] + dx
            if (0 <= new_y < height and 0 <= new_x < width and board[new_y, new_x] == 0):
                moves.append(action)
        return moves

if __name__ == "__main__":
    # Tournament
    print("\n=== MCTS (500 sims) vs GREEDY (5 games) ===\n")
    mcts = MCTSAgent(simulations=500)
    greedy = GreedyAgent()
    results = {'mcts': 0, 'greedy': 0, 'draw': 0}

    for game_num in range(5):
        game = TronGame(width=10, height=10)
        state = game.reset()
        moves = 0
        
        while not game.game_over and moves < 100:
            a1 = mcts.get_action(state, 1)
            a2 = greedy.get_action(state, 2)
            state, reward, done = game.step(a1, a2)
            moves += 1
        
        winner_name = "MCTS" if game.winner == 1 else ("Greedy" if game.winner == 2 else "Draw")
        if game.winner == 1:
            results['mcts'] += 1
        elif game.winner == 2:
            results['greedy'] += 1
        else:
            results['draw'] += 1
        
        print(f"Game {game_num + 1}: Winner = {winner_name}, Moves = {moves}")

    print(f"\nResults: MCTS={results['mcts']}, Greedy={results['greedy']}, Draws={results['draw']}")

    # Optional: Visualize MCTS's probabilistic decision-making
    print("\n" + "="*60)
    print("Watch MCTS vs Greedy - see the exploration!")
    print("="*60)

    game_viz = TronGame(width=10, height=10, visualize=True, cell_size=50)
    state = game_viz.reset()
    move_count = 0

    while not game_viz.game_over and move_count < 100:
        a1 = mcts.get_action(state, 1)
        a2 = greedy.get_action(state, 2)
        state, reward, done = game_viz.step(a1, a2)
        move_count += 1

    winner = "MCTS" if game_viz.winner == 1 else ("Greedy" if game_viz.winner == 2 else "Draw")
    print(f"Visualized game: {winner} wins!")
    game_viz.close()
```

### Reflection Questions

**Question 10:** Explain why MCTS can make good decisions without explicitly evaluating position quality, and how the law of large numbers ensures convergence to optimal play.

**Question 11:** Compare the UCB1 exploration parameter's role in MCTS to the depth parameter in minimax. How do they both address the exploration-exploitation trade-off, and what makes their approaches fundamentally different?

**Question 12:** Describe why MCTS might outperform minimax in games with high branching factors or deep game trees, referencing the computational complexity of each approach.

---

## Exercise 5: LLM-Based Reasoning Agent (Ollama)

### Description
This exercise uses a Large Language Model (via Ollama's llama3.2) to make decisions through natural language reasoning. The agent receives a textual description of the game state and returns an action based on strategic reasoning.

### Key Concepts
- **Prompt engineering**: Structuring input to guide AI reasoning toward desired outputs
- **Few-shot learning**: Providing examples to teach the model appropriate response format
- **Chain-of-thought reasoning**: Having the model explain its thinking process
- **Symbol grounding**: Converting spatial/numeric game state to language and back

### Task
Run the code and observe the LLM agent. Note:
- What kinds of reasoning does the model articulate?
- Are there patterns in when it succeeds vs fails?
- How does response time compare to other agents?

**Important:** Save this code as `ollamatron.py`.

```python
# ollamatron.py - Add to this file

from greedy import GreedyAgent
import requests
import json

# LLM Agent implementation
class OllamaAgent:
    """Agent using LLM reasoning via Ollama"""
    
    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://ollama.cs.wallawalla.edu:11434/api/generate"
    
    def state_to_text(self, state, player):
        """Convert game state to text description"""
        board = state['board']
        my_pos = state['p1_pos'] if player == 1 else state['p2_pos']
        opp_pos = state['p2_pos'] if player == 1 else state['p1_pos']
        my_moves = state['p1_moves'] if player == 1 else state['p2_moves']
        
        desc = f"You are Player {player} in a Tron game on a {board.shape[0]}x{board.shape[1]} grid.\n"
        desc += f"Your position: row {my_pos[0]}, col {my_pos[1]}\n"
        desc += f"Opponent position: row {opp_pos[0]}, col {opp_pos[1]}\n"
        desc += f"Your available moves: {', '.join(my_moves)}\n"
        desc += f"Empty cells near you: {self.count_nearby_space(board, my_pos)}\n"
        desc += f"Empty cells near opponent: {self.count_nearby_space(board, opp_pos)}\n"
        desc += "\nChoose ONE move (UP/DOWN/LEFT/RIGHT) that maximizes your survival space."
        return desc
    
    def count_nearby_space(self, board, pos):
        """Count empty cells in 3x3 area"""
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = pos[0] + dy, pos[1] + dx
                if (0 <= y < board.shape[0] and 0 <= x < board.shape[1] and board[y, x] == 0):
                    count += 1
        return count
    
    def get_action(self, state, player):
        """Query Ollama for action"""
        my_moves = state['p1_moves'] if player == 1 else state['p2_moves']
        if not my_moves:
            return None
        
        prompt = self.state_to_text(state, player)
        
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').upper()
                
                # Extract first valid move from response
                for move in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    if move in text and move in my_moves:
                        return move
            
            # Fallback to greedy if LLM fails
            return self.greedy_fallback(state, player)
        
        except Exception as e:
            print(f"  [LLM Error: {e}]")
            return self.greedy_fallback(state, player)
    
    def greedy_fallback(self, state, player):
        """Fallback greedy action"""
        pos = state['p1_pos'] if player == 1 else state['p2_pos']
        moves = state['p1_moves'] if player == 1 else state['p2_moves']
        
        best_action, best_space = None, -1
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        for action in moves:
            dy, dx = directions[action]
            new_pos = (pos[0] + dy, pos[1] + dx)
            space = flood_fill(state['board'], new_pos, player)
            if space > best_space:
                best_space, best_action = space, action
        
        return best_action

# Tournament and visualization functions
def run_ollama_tournament(num_games=3, model="llama3.2"):
    """Run tournament between LLM and greedy agents"""
    print("\n=== OLLAMA-LLM vs GREEDY ({} games) ===\n".format(num_games))
    print("(Note: This may take 30-60 seconds due to LLM inference time)\n")
        
    ollama = OllamaAgent(model=model)
    greedy = GreedyAgent()
    results = {'ollama': 0, 'greedy': 0, 'draw': 0}
    
    for game_num in range(num_games):
        game = TronGame(width=10, height=10)
        state = game.reset()
        moves = 0
        
        print(f"Game {game_num + 1} starting...")
        while not game.game_over and moves < 50:
            a1 = ollama.get_action(state, 1)
            a2 = greedy.get_action(state, 2)
            state, reward, done = game.step(a1, a2)
            moves += 1
        
        winner_name = "LLM" if game.winner == 1 else ("Greedy" if game.winner == 2 else "Draw")
        if game.winner == 1:
            results['ollama'] += 1
        elif game.winner == 2:
            results['greedy'] += 1
        else:
            results['draw'] += 1
        
        print(f"  Winner = {winner_name}, Moves = {moves}\n")
    
    print(f"Results: LLM={results['ollama']}, Greedy={results['greedy']}, Draws={results['draw']}")
    return results

def visualize_ollama_game(model="llama3.2"):
    """Run one visualized game between LLM and greedy (optional)"""
    print("\n" + "="*60)
    print("Watch LLM vs Greedy with visualization (optional)")
    print("="*60)
    
    from tron_agents import GreedyAgent, OllamaAgent
    
    ollama = OllamaAgent(model=model)
    greedy = GreedyAgent()
    game_viz = TronGame(width=10, height=10, visualize=True, cell_size=50)
    state = game_viz.reset()
    move_count = 0
    
    while not game_viz.game_over and move_count < 50:
        a1 = ollama.get_action(state, 1)
        a2 = greedy.get_action(state, 2)
        state, reward, done = game_viz.step(a1, a2)
        move_count += 1
    
    winner = "LLM" if game_viz.winner == 1 else ("Greedy" if game_viz.winner == 2 else "Draw")
    print(f"Visualized game: {winner} wins!")
    game_viz.close()

# Test code when run directly
if __name__ == "__main__":
    run_ollama_tournament(3)
    # Uncomment to visualize (slower due to LLM):
    # visualize_ollama_game()
```

### Reflection Questions

**Question 13:** Analyze the trade-offs between using an LLM for game-playing versus traditional algorithms. Consider factors like interpretability, computational cost, and performance ceiling.

**Question 14:** Explain why prompt engineering is crucial for the LLM agent's performance, and describe how changing the prompt structure might improve or degrade decision quality.

**Question 15:** Compare the LLM's "reasoning" (pattern matching from training) to MCTS's statistical reasoning and minimax's logical reasoning. What are the fundamental epistemological differences in how each approach "knows" what move is best?

---

## Exercise 6: Head-to-Head Algorithm Comparison

### Description
This exercise runs a round-robin tournament between all implemented agents to directly compare their performance. Observing which agents dominate reveals the relative strengths of different AI paradigms.

### Key Concepts
- **Round-robin tournament**: Each agent plays every other agent multiple times
- **Transitive vs intransitive dominance**: Whether "A beats B" and "B beats C" implies "A beats C"
- **Performance stability**: Consistency of results across multiple trials
- **Algorithm scaling**: How performance changes with computational budget

### Task
Run the tournament and analyze the results. Pay attention to:
- Which algorithms have the highest win rates?
- Are there any surprises (weak algorithms beating strong ones)?
- How does computational time correlate with performance?

```python
# Import all components from our modules
from tron_base import TronGame, RandomAgent
from greedy import GreedyAgent
from minimax import MinimaxAgent
from mcts import MCTSAgent
import time

# Define all agents using our imported classes
agents = {
    'Random': RandomAgent(),
    'Greedy': GreedyAgent(),
    'Minimax-2': MinimaxAgent(depth=2),
    'Minimax-3': MinimaxAgent(depth=5),
    'MCTS-200': MCTSAgent(simulations=200),
    'MCTS-500': MCTSAgent(simulations=500)
}

# Tournament function
def run_round_robin_tournament(games_per_matchup=3):
    """Run round-robin tournament between all agents"""
    print("\n=== ROUND-ROBIN TOURNAMENT ===")
    print("(Each matchup: {} games, 10x10 grid)\n".format(games_per_matchup))
    
    results = {name: {'wins': 0, 'losses': 0, 'draws': 0, 'time': 0} for name in agents}
    
    agent_names = list(agents.keys())
    for i, name1 in enumerate(agent_names):
        for name2 in agent_names[i+1:]:
            print(f"\n{name1} vs {name2}:")
            
            for game_num in range(games_per_matchup):
                game = TronGame(width=10, height=10)
                state = game.reset()
                moves = 0
                
                start_time = time.time()
                while not game.game_over and moves < 100:
                    a1 = agents[name1].get_action(state, 1)
                    a2 = agents[name2].get_action(state, 2)
                    state, reward, done = game.step(a1, a2)
                    moves += 1
                elapsed = time.time() - start_time
                
                if game.winner == 1:
                    results[name1]['wins'] += 1
                    results[name2]['losses'] += 1
                    print(f"  Game {game_num + 1}: {name1} wins ({moves} moves, {elapsed:.2f}s)")
                elif game.winner == 2:
                    results[name2]['wins'] += 1
                    results[name1]['losses'] += 1
                    print(f"  Game {game_num + 1}: {name2} wins ({moves} moves, {elapsed:.2f}s)")
                else:
                    results[name1]['draws'] += 1
                    results[name2]['draws'] += 1
                    print(f"  Game {game_num + 1}: Draw ({moves} moves, {elapsed:.2f}s)")
                
                results[name1]['time'] += elapsed / 2
                results[name2]['time'] += elapsed / 2
    
    # Display final standings
    print("\n" + "="*60)
    print("FINAL STANDINGS")
    print("="*60)
    print(f"{'Agent':<15} {'Wins':>6} {'Losses':>6} {'Draws':>6} {'Win%':>6} {'Avg Time':>10}")
    print("-"*60)
    
    sorted_agents = sorted(results.items(), key=lambda x: x[1]['wins'], reverse=True)
    for name, stats in sorted_agents:
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        win_pct = (stats['wins'] / total_games * 100) if total_games > 0 else 0
        avg_time = stats['time'] / total_games if total_games > 0 else 0
        print(f"{name:<15} {stats['wins']:>6} {stats['losses']:>6} {stats['draws']:>6} {win_pct:>5.1f}% {avg_time:>9.3f}s")
    
    return results

# Test code when run directly
if __name__ == "__main__":
    run_round_robin_tournament(3)
```

### Reflection Questions

**Question 16:** Analyze the tournament results to identify which algorithmic properties (lookahead depth, simulation count, heuristic quality) most strongly correlate with winning performance in Tron.

**Question 17:** Explain why certain matchups might produce unexpected results (e.g., a weaker agent occasionally beating a stronger one), and what this reveals about the relationship between algorithm design and opponent behavior.

**Question 18:** Discuss the time-performance trade-off observed in the results. If this were a real-time game with a 1-second move limit, how would you balance algorithm sophistication against time constraints?

---

## Exercise 7: Advanced Heuristic - Articulation Points

### Description
This exercise implements a sophisticated evaluation function based on **articulation points** - moves that would divide the board into disconnected regions, trapping the opponent. This heuristic detects "cut-off" moves that greedy space-counting misses.

### Key Concepts
- **Articulation point**: A position whose removal disconnects a graph
- **Board segmentation**: Dividing the game space into isolated regions
- **Strategic vs tactical heuristics**: Detecting patterns vs counting resources
- **Voronoi regions**: Areas of the board "owned" by each player

### Task
Run the improved minimax with articulation-point detection and compare to both standard minimax and greedy. Observe:
- Does the better heuristic improve performance at all depths?
- How much more expensive is it to compute?
- Does it change the strategic behavior visibly?

```python
# advanced_heuristic.py
from tron_base import TronGame, flood_fill
from tron_agents import GreedyAgent
from copy import deepcopy
import time

def find_articulation_points(board, start_pos, player_id):
    """
    Find articulation points - moves that would split opponent's space.
    Returns the number of separate regions opponent would be split into.
    """
    opponent_id = 3 - player_id
    
    # Find opponent's reachable space
    opponent_space = set()
    stack = [start_pos]
    visited = set()
    height, width = board.shape
    
    # First, identify all cells reachable by opponent
    opp_pos = None
    for y in range(height):
        for x in range(width):
            if board[y, x] == opponent_id:
                opp_pos = (y, x)
                break
        if opp_pos:
            break
    
    if not opp_pos:
        return 1
    
    # Flood fill from opponent position to find their territory
    stack = [opp_pos]
    visited = set()
    while stack:
        pos = stack.pop()
        if pos in visited:
            continue
        y, x = pos
        if not (0 <= y < height and 0 <= x < width):
            continue
        if board[y, x] != 0 and board[y, x] != opponent_id:
            continue
        
        visited.add(pos)
        opponent_space.add(pos)
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            stack.append((y + dy, x + dx))
    
    # Now check if our move at start_pos splits opponent's space
    # Count connected components in opponent_space excluding start_pos
    remaining_space = opponent_space - {start_pos}
    
    if not remaining_space:
        return 1
    
    # Count connected components
    components = 0
    unvisited = remaining_space.copy()
    
    while unvisited:
        components += 1
        start = unvisited.pop()
        stack = [start]
        component_visited = set()
        
        while stack:
            pos = stack.pop()
            if pos in component_visited:
                continue
            if pos not in remaining_space:
                continue
            
            component_visited.add(pos)
            unvisited.discard(pos)
            
            y, x = pos
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (y + dy, x + dx)
                if neighbor in remaining_space and neighbor not in component_visited:
                    stack.append(neighbor)
    
    return components

def advanced_evaluate(board, p1_pos, p2_pos):
    """
    Advanced evaluation considering:
    1. Space control (like greedy)
    2. Articulation points (cutting off opponent)
    3. Voronoi territory (cells closer to you)
    """
    # Basic space control
    p1_space = flood_fill(board, p1_pos, 1)
    p2_space = flood_fill(board, p2_pos, 2)
    space_diff = p1_space - p2_space
    
    # Articulation bonus: does our position split opponent's space?
    p1_splits = find_articulation_points(board, p1_pos, 1)
    p2_splits = find_articulation_points(board, p2_pos, 2)
    articulation_bonus = (p1_splits - p2_splits) * 10  # Weight articulation points heavily
    
    # Voronoi territory - cells closer to us than opponent
    height, width = board.shape
    voronoi_score = 0
    
    for y in range(height):
        for x in range(width):
            if board[y, x] == 0:  # Empty cell
                dist_p1 = abs(y - p1_pos[0]) + abs(x - p1_pos[1])
                dist_p2 = abs(y - p2_pos[0]) + abs(x - p2_pos[1])
                if dist_p1 < dist_p2:
                    voronoi_score += 1
                elif dist_p2 < dist_p1:
                    voronoi_score -= 1
    
    # Combined score
    return space_diff + articulation_bonus + voronoi_score * 0.5

class AdvancedMinimaxAgent:
    """Minimax with articulation-point aware evaluation"""
    
    def __init__(self, depth=5):
        self.depth = depth
        self.nodes_evaluated = 0
    
    def evaluate_state(self, board, p1_pos, p2_pos):
        """Use advanced evaluation function"""
        return advanced_evaluate(board, p1_pos, p2_pos)
    
    def minimax(self, state, depth, alpha, beta, maximizing_player):
        """Minimax with alpha-beta pruning"""
        self.nodes_evaluated += 1
        if depth == 0 or not state['p1_moves'] or not state['p2_moves']:
            return self.evaluate_state(state['board'], state['p1_pos'], state['p2_pos'])
        
        if maximizing_player:
            max_eval = float('-inf')
            for action in state['p1_moves']:
                new_state = self.simulate_move(state, action, None, 1)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in state['p2_moves']:
                new_state = self.simulate_move(state, None, action, 2)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def simulate_move(self, state, p1_action, p2_action, player):
        """Create new state after hypothetical move"""
        new_state = deepcopy(state)
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        
        if player == 1 and p1_action:
            dy, dx = directions[p1_action]
            new_pos = (state['p1_pos'][0] + dy, state['p1_pos'][1] + dx)
            new_state['board'][new_pos] = 1
            new_state['p1_pos'] = new_pos
            new_state['p1_moves'] = self.get_valid_moves_from_board(new_state['board'], new_pos)
        
        if player == 2 and p2_action:
            dy, dx = directions[p2_action]
            new_pos = (state['p2_pos'][0] + dy, state['p2_pos'][1] + dx)
            new_state['board'][new_pos] = 2
            new_state['p2_pos'] = new_pos
            new_state['p2_moves'] = self.get_valid_moves_from_board(new_state['board'], new_pos)
        
        return new_state
    
    def get_valid_moves_from_board(self, board, pos):
        """Helper to get valid moves from board state"""
        moves = []
        directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        height, width = board.shape
        
        for action, (dy, dx) in directions.items():
            new_y, new_x = pos[0] + dy, pos[1] + dx
            if (0 <= new_y < height and 0 <= new_x < width and board[new_y, new_x] == 0):
                moves.append(action)
        return moves
    
    def get_action(self, state, player):
        """Select best action using minimax"""
        self.nodes_evaluated = 0
        moves = state['p1_moves'] if player == 1 else state['p2_moves']
        if not moves:
            return None
        
        best_action = moves[0]
        best_value = float('-inf') if player == 1 else float('inf')
        
        for action in moves:
            new_state = self.simulate_move(state, action if player == 1 else None, 
                                          action if player == 2 else None, player)
            value = self.minimax(new_state, self.depth - 1, float('-inf'), float('inf'), 
                               player == 2)
            
            if player == 1 and value > best_value:
                best_value = value
                best_action = action
            elif player == 2 and value < best_value:
                best_value = value
                best_action = action
        
        return best_action

def compare_heuristics(num_games=15, depth=5, board_size=10):
    """Compare standard minimax vs advanced minimax"""
    print(f"\n{'='*70}")
    print(f"HEURISTIC COMPARISON: Standard vs Advanced Evaluation")
    print(f"Board: {board_size}x{board_size}, Depth: {depth}")
    print(f"{'='*70}\n")
    
    from tron_agents import MinimaxAgent
    
    standard = MinimaxAgent(depth=depth)
    advanced = AdvancedMinimaxAgent(depth=depth)
    greedy = GreedyAgent()
    
    matchups = [
        ("Advanced Minimax", advanced, "Standard Minimax", standard),
        ("Advanced Minimax", advanced, "Greedy", greedy),
        ("Standard Minimax", standard, "Greedy", greedy)
    ]
    
    results = {}
    
    for name1, agent1, name2, agent2 in matchups:
        print(f"\n--- {name1} vs {name2} ({num_games} games) ---")
        
        wins1 = 0
        wins2 = 0
        draws = 0
        total_time = 0
        
        for game_num in range(num_games):
            game = TronGame(width=board_size, height=board_size)
            state = game.reset()
            moves = 0
            
            start = time.time()
            
            while not game.game_over and moves < 150:
                a1 = agent1.get_action(state, 1)
                a2 = agent2.get_action(state, 2)
                state, reward, done = game.step(a1, a2)
                moves += 1
            
            elapsed = time.time() - start
            total_time += elapsed
            
            if game.winner == 1:
                wins1 += 1
            elif game.winner == 2:
                wins2 += 1
            else:
                draws += 1
        
        win_rate1 = wins1 / num_games * 100
        avg_time = total_time / num_games
        
        results[f"{name1}_vs_{name2}"] = {
            'wins1': wins1,
            'wins2': wins2,
            'draws': draws,
            'win_rate': win_rate1,
            'avg_time': avg_time
        }
        
        print(f"  {name1}: {wins1}, {name2}: {wins2}, Draws: {draws}")
        print(f"  {name1} Win Rate: {win_rate1:.1f}%")
        print(f"  Avg Time/Game: {avg_time:.2f}s")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for matchup, stats in results.items():
        print(f"\n{matchup}:")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg Time: {stats['avg_time']:.2f}s")
    
    return results

if __name__ == "__main__":
    # Test on standard board
    print("\n" + "="*70)
    print("TEST 1: Standard 10x10 Board")
    print("="*70)
    compare_heuristics(num_games=15, depth=5, board_size=10)
    
    # Test on small board where tactics matter more
    print("\n\n" + "="*70)
    print("TEST 2: Small 6x6 Board (Tactics-Heavy)")
    print("="*70)
    compare_heuristics(num_games=15, depth=4, board_size=6)
    
    # Optional: Visualize 
    print("\n" + "="*60)
    print("Watch MCTS vs Greedy - see the exploration!")
    print("="*60)

    game_viz = TronGame(width=20, height=20, visualize=True, cell_size=50)
    state = game_viz.reset()
    move_count = 0

    depth = 7
    advanced = AdvancedMinimaxAgent(depth=depth)
    greedy = GreedyAgent()

    while not game_viz.game_over and move_count < 100:
        a1 = advanced.get_action(state, 1)
        a2 = greedy.get_action(state, 2)
        state, reward, done = game_viz.step(a1, a2)
        move_count += 1

    winner = "MCTS" if game_viz.winner == 1 else ("Greedy" if game_viz.winner == 2 else "Draw")
    print(f"Visualized game: {winner} wins!")
    game_viz.close()
```

### Reflection Questions

**Question 19:** Does the advanced heuristic improve minimax's performance against greedy? If so, by how much? If not, why might even sophisticated pattern detection fail to help?

**Question 20:** Compare the computational cost of advanced vs standard minimax. Is the articulation-point detection worth the extra computation time given the performance improvement (or lack thereof)?

**Question 21:** The advanced heuristic looks for "cut-off" moves that split opponent's space. In what game situations would this strategic insight matter most, and do these situations occur frequently enough to justify the complexity?

---

## Exercise 8: Performance Analysis and Reflection

### Description
This final exercise asks you to synthesize your observations across all previous exercises and conduct a deeper analysis of what makes certain AI approaches effective for adversarial games.

### Key Concepts
- **Branching factor**: Number of legal moves at each position (affects search complexity)
- **Horizon effect**: Inability to see beyond search depth leading to short-sighted decisions
- **Anytime algorithms**: Algorithms that can return progressively better answers given more time
- **No free lunch theorem**: No single algorithm is best for all problems

### Task
Review all previous exercise outputs and answer the reflection questions below. No new code to run—this is pure analysis and synthesis.

### Reflection Questions

**Question 22:** Add `AdvancedMinimax` to the agent tournament and run it again a few times. Create a ranking of all tested algorithms from weakest to strongest, justifying your ranking with specific evidence from tournament results. Then, describe a hypothetical Tron variant (different board size, different rules) where your ranking might change, explaining why.

**Question 23:** Compare the "intelligence" exhibited by minimax (logical reasoning), MCTS (statistical sampling), and LLM (pattern matching). Are these fundamentally different types of intelligence, or are they all reducible to the same underlying computational process?

**Question 24:** Imagine you're building a Tron AI for a competition with a strict 100ms time limit per move. Describe your design choices (which algorithm(s), what parameters, any hybrid approaches) and justify each decision with reference to the time-performance trade-offs you observed.

---

## Summary and Key Takeaways

Through this lab, you've observed multiple AI paradigms applied to the same adversarial game:

**Core Insights:**
- **Heuristics matter**: Even simple evaluation functions (space control) dramatically outperform random play
- **Lookahead trades time for quality**: Minimax's systematic search finds better moves than greedy, but at computational cost
- **Sampling approximates exhaustive search**: MCTS achieves good performance without examining every possibility
- **Domain knowledge vs learning**: Traditional algorithms use hand-crafted strategies; LLMs leverage patterns from training
- **No universal best**: Algorithm choice depends on time constraints, opponent behavior, and game complexity

**Connections to Broader AI:**
- Tron demonstrates core adversarial search concepts (zero-sum, minimax principle, evaluation functions)
- MCTS's success here extends to Go, where exhaustive search is impossible
- Hybrid architectures mirror real-world AI systems that combine multiple techniques
- The exploration-exploitation trade-off appears in reinforcement learning, multi-armed bandits, and optimization

**Strategic Depth:**
- Tron's simplicity belies rich strategy: space control, wall-building, forcing opponents into traps
- Good AI must balance immediate survival (don't crash) with long-term positioning (control territory)
- Endgame requires exact calculation (minimax strength), while opening favors fast heuristics

---

## Submission Instructions

Submit a document containing your responses to all **27 reflection questions** (24 main + 3 visualization). Each response should be 2-5 complete sentences demonstrating critical thinking and connecting to textbook concepts.


## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including all code from this lab and:

Create `lab_ch5_results.md`:

```markdown
# Names: Your names here
# Lab: lab3 (Adversarial Search)
# Date: Today's date
```

And your answers to all reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the third lab!

---