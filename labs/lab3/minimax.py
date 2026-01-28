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