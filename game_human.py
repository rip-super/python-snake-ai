import pygame
import random
import time
from enum import Enum
from collections import namedtuple, deque

pygame.init()
font = pygame.font.SysFont('SEGUIVAR', 36)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10
MOVE_INTERVAL = 0.001

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.is_started = False
        self.move_queue = deque()
        self.last_move_time = time.time()
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                self.is_started = True  # Start the game on first key press
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.move_queue.append(Direction.LEFT)
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.move_queue.append(Direction.RIGHT)
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.move_queue.append(Direction.UP)
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.move_queue.append(Direction.DOWN)

        # Update the UI even if the game hasn't started
        self._update_ui()

        # Return if not started yet
        if not self.is_started:
            return False, self.score

        # Process move queue based on timing
        current_time = time.time()
        if current_time - self.last_move_time >= MOVE_INTERVAL:
            if self.move_queue:
                self.direction = self.move_queue.popleft()  # Get the next move from the queue
            self._move(self.direction)  # Move the snake
            self.snake.insert(0, self.head)  # Insert the new head position
            self.last_move_time = current_time

            # Check if game over
            if self._is_collision():
                return True, self.score

            # Place new food or just move
            if self.head == self.food:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return False, self.score
    
    def _is_collision(self):
        # Hits boundary
        if (self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or
                self.head.y > self.h - BLOCK_SIZE or self.head.y < 0):
            return True
        # Hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGame()
    
    # Game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            break
        
    print('Final Score:', score)
    pygame.quit()