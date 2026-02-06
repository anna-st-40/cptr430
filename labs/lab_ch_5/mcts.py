# mcts.py - Add to this file

from tron_base import TronGame, flood_fill
from greedy import GreedyAgent
import math
import random
from copy import deepcopy

# MCTS implementation
class MCTSNode:
    """Node in the Monte Carlo search tree"""
    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state['p1_moves'][:] if parent is None or parent.player == 2 else state['p2_moves'][:]
        self.player = 1 if parent is None or parent.player == 2 else 2
    
    def ucb1(self, exploration=1.41):
        """UCB1 formula for balancing exploitation/exploration"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self):
        """Select child with highest UCB1 value"""
        return max(self.children, key=lambda c: c.ucb1())

class MCTSAgent:
    """Monte Carlo Tree Search agent"""
    
    def __init__(self, simulations=500):
        self.simulations = simulations
    
    def get_action(self, state, player):
        """Run MCTS to select best action"""
        if not state['p1_moves'] and player == 1:
            return None
        if not state['p2_moves'] and player == 2:
            return None
        
        root = MCTSNode(state)
        root.player = 3 - player  # Opposite player just moved
        
        # Run simulations
        for _ in range(self.simulations):
            node = self.select(root)
            result = self.simulate(node.state, node.player)
            self.backpropagate(node, result)
        
        # Choose most visited child
        if not root.children:
            moves = state['p1_moves'] if player == 1 else state['p2_moves']
            return moves[0] if moves else None
        
        best = max(root.children, key=lambda c: c.visits)
        return best.action
    
    def select(self, node):
        """Selection phase: traverse tree using UCB1"""
        while node.untried_actions == [] and node.children:
            node = node.best_child()
        
        # Expansion
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state = self.make_move(node.state, action, node.player)
            child = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child)
            return child
        
        return node
    
    def simulate(self, state, player):
        """Simulation phase: random playout"""
        sim_state = deepcopy(state)
        current_player = player
        
        for _ in range(50):  # Max simulation depth
            moves = sim_state['p1_moves'] if current_player == 1 else sim_state['p2_moves']
            if not moves:
                return -1 if current_player == 1 else 1
            
            action = random.choice(moves)
            sim_state = self.make_move(sim_state, action, current_player)
            current_player = 3 - current_player
        
        # Heuristic evaluation if no winner
        p1_space = flood_fill(sim_state['board'], sim_state['p1_pos'], 1)
        p2_space = flood_fill(sim_state['board'], sim_state['p2_pos'], 2)
        return 1 if p1_space > p2_space else -1
    
    def backpropagate(self, node, result):
        """Backpropagation phase: update statistics"""
        while node:
            node.visits += 1
            node.value += result if node.player == 1 else -result
            node = node.parent
        
    def make_move(self, state, action, player):
        new_state = deepcopy(state)

        directions = {
            'UP': (-1, 0), 'DOWN': (1, 0),
            'LEFT': (0, -1), 'RIGHT': (0, 1)
        }

        pos = new_state['p1_pos'] if player == 1 else new_state['p2_pos']
        dy, dx = directions[action]
        new_pos = (pos[0] + dy, pos[1] + dx)

        # --- SAFETY CHECK ---
        h, w = new_state['board'].shape
        if not (0 <= new_pos[0] < h and 0 <= new_pos[1] < w):
            return new_state   # invalid move -> treat as dead end

        if new_state['board'][new_pos] != 0:
            return new_state

        new_state['board'][new_pos] = player

        if player == 1:
            new_state['p1_pos'] = new_pos
        else:
            new_state['p2_pos'] = new_pos

        # Recompute BOTH players' moves every time
        new_state['p1_moves'] = self.get_valid_moves(new_state['board'], new_state['p1_pos'])
        new_state['p2_moves'] = self.get_valid_moves(new_state['board'], new_state['p2_pos'])

        return new_state

    
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
input()
game_viz.close()