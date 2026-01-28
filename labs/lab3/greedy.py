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
input()
game_viz.close()