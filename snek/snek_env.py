import random
import numpy as np

# Reward Values
REWARD_EAT_FOOD = 30            # reward for eating food
REWARD_STEP = -1                # penalty for each step
REWARD_OUT_OF_BOUNDS = -60      # Heavy penalty for going out of bounds

MAX_STEPS = 100

class SnakeEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.max_foods = max(1, grid_size)  # Number of food pieces based on grid size
        self.episode = 1
        self.end_condition = None
        self.current_reward = 0
        self.reset()
    
    def reset(self):
        # Initialize snake position at the center of the grid
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

        # Initialize the food list before placing food
        self.food = []
        self.food = self._place_multiple_food(self.max_foods)

        # Initialize other variables
        self.score = 0
        self.steps = 0
        self.done = False
        self.end_condition = None
        self.current_reward = 0

        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake and food not in self.food:
                return food
    
    def _place_multiple_food(self, number_of_foods):
        food_positions = []
        for _ in range(number_of_foods):
            food_positions.append(self._place_food())
        return food_positions
    
    def _get_state(self):
        head_x, head_y = self.snake[0]
        
        # Normalize positions between 0 and 1
        head_x_norm = head_x / (self.grid_size - 1)
        head_y_norm = head_y / (self.grid_size - 1)
        
        food_positions = []
        for food_x, food_y in self.food:
            food_x_norm = food_x / (self.grid_size - 1)
            food_y_norm = food_y / (self.grid_size - 1)
            food_positions.extend([food_x_norm, food_y_norm])
        
        # Pad food positions to always have 5 food items (10 values)
        food_positions.extend([0] * (10 - len(food_positions)))
        
        # Calculate distances to walls and normalize
        dist_up = head_y / (self.grid_size - 1)
        dist_down = (self.grid_size - 1 - head_y) / (self.grid_size - 1)
        dist_left = head_x / (self.grid_size - 1)
        dist_right = (self.grid_size - 1 - head_x) / (self.grid_size - 1)
        
        state = np.array([
            head_x_norm,
            head_y_norm,
            dist_up,
            dist_down,
            dist_left,
            dist_right,
            *food_positions
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        self.steps += 1
        head_x, head_y = self.snake[0]
        
        # Define movement based on action
        if action == 0:  # UP
            head_y -= 1
        elif action == 1:  # DOWN
            head_y += 1
        elif action == 2:  # LEFT
            head_x -= 1
        elif action == 3:  # RIGHT
            head_x += 1
        
        # Check for out of bounds
        if head_x < 0 or head_x >= self.grid_size or head_y < 0 or head_y >= self.grid_size:
            self.done = True
            reward = REWARD_OUT_OF_BOUNDS
            self.current_reward += reward  # Accumulate reward for the episode
            self.end_condition = "Out of Bounds"
            return self._get_state(), reward, self.done, {}
        
        # Move snake
        self.snake = [(head_x, head_y)]
        
        # Check if food is eaten
        if (head_x, head_y) in self.food:
            self.score += 1
            reward = REWARD_EAT_FOOD
            self.current_reward += reward  # Accumulate reward for the episode
            
            # Remove the eaten food and place a new one
            self.food.remove((head_x, head_y))
            if self.score >= self.max_foods:
                self.done = True
                self.end_condition = "All Foods Eaten"
            else:
                self.food.append(self._place_food())
        else:
            reward = REWARD_STEP  # Penalty for each step
            self.current_reward += reward  # Accumulate reward for the episode
        
        # Check if max steps is reached
        if self.steps >= MAX_STEPS:
            self.done = True
            self.end_condition = "Max Steps Reached"
        
        return self._get_state(), reward, self.done, {}

    def render(self):
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Draw food
        for food_x, food_y in self.food:
            grid[food_y][food_x] = '#'
        
        # Draw snake head
        head_x, head_y = self.snake[0]
        grid[head_y][head_x] = '@'
        
        # Clear screen and print grid with boundaries
        print("\033[H\033[J")  # Clear console
        
        # Print column numbers
        print('  ' + ' '.join(f'{i:2}' for i in range(self.grid_size)))
        
        # Print top boundary
        print('  +' + '-' * (self.grid_size * 2 + 1) + '+')
        
        for i, row in enumerate(grid):
            # Print each row with left and right boundaries, and row numbers
            print(f'{i:2}|{" ".join(f"{cell:2}" for cell in row)}|')
        
        # Print bottom boundary
        print('  +' + '-' * (self.grid_size * 2 + 1) + '+')
    
    def update_episode(self, episode):
        self.episode = episode

    def is_game_over(self):
        return self.done and self.end_condition == "All Foods Eaten"

    def close(self):
        pass