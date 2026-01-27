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