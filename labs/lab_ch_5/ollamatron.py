# ollamatron.py - Add to this file

from greedy import GreedyAgent
from tron_base import TronGame, flood_fill
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

            print(response._content)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').upper()

                print(f"  [LLM Response: {text.strip()}]")
                
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
        print("  [Using Greedy Fallback]")
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
    visualize_ollama_game()